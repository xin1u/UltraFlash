"""
Ultra Flash Long Video SR Inference
Pipeline: LongLive (4-step, LoRA) -> Latent Upsampler (2x) -> SR DiT (1-step) -> Tiny Decoder (HR)

Supports:
  - Sparse SR DiT (--use_sparse_sr)
  - DCM acceleration (--dcm_*)
  - Tiny Decoder (--tiny_decoder)
  - FP8 quantization (--fp8, --fp8_sr)
  - torch.compile (--torch_compile, --compile_sr_dit)
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.io import write_video
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import CausalInferencePipeline
from pipeline.causal_cascade_streaming import CausalCascadeStreamingPipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


parser = argparse.ArgumentParser(description="Ultra Flash Long Video Inference")
parser.add_argument("--config_path", type=str, required=True)
# SR overrides
parser.add_argument("--sr_config", type=str, default=None)
parser.add_argument("--sr_dit_ckpt", type=str, default=None)
parser.add_argument("--sf_chunks_per_sr", type=int, default=None)
parser.add_argument("--sr_timestep", type=float, default=None)
parser.add_argument("--sr_kv_len", type=int, default=None)
parser.add_argument("--condition_noise_scale", type=float, default=None)
parser.add_argument("--lr_only", action="store_true")
parser.add_argument("--save_lr_video", action="store_true")
# Sparse SR
parser.add_argument("--use_sparse_sr", action="store_true")
# Acceleration
parser.add_argument("--torch_compile", action="store_true")
parser.add_argument("--fp8", action="store_true")
parser.add_argument("--compile_sr_dit", action="store_true")
parser.add_argument("--fp8_sr", action="store_true")
# Tiny Decoder
parser.add_argument("--tiny_decoder", action="store_true")
parser.add_argument("--decoder_ckpt", type=str, default="checkpoints/tiny_decoder.pth")
# DCM
parser.add_argument("--dcm_lr_steps_subsequent", type=int, default=4)
parser.add_argument("--dcm_adaptive_refresh", action="store_true", default=False)
parser.add_argument("--dcm_iqa_threshold", type=float, default=0.62)
parser.add_argument("--dcm_sr_cache_adapt", action="store_true", default=False)
parser.add_argument("--dcm_sr_kv_len_min", type=int, default=1)
parser.add_argument("--dcm_sr_cache_warmup_groups", type=int, default=2)
parser.add_argument("--no_dcm", action="store_true")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Merge CLI overrides
for key in ["sr_config", "sr_dit_ckpt", "sf_chunks_per_sr", "sr_timestep", "sr_kv_len", "condition_noise_scale"]:
    cli_val = getattr(args, key)
    if cli_val is not None:
        OmegaConf.update(config, key, cli_val)

use_sparse_sr = args.use_sparse_sr or getattr(config, "use_sparse_sr", False)
if not hasattr(config, "sr_config"):
    config.sr_config = "configs/sr_sparse_ultraforcing.py"
if not hasattr(config, "sr_dit_ckpt"):
    config.sr_dit_ckpt = "checkpoints/sr_dit_sparse.pth"
if not hasattr(config, "sf_chunks_per_sr"):
    config.sf_chunks_per_sr = 2
if not hasattr(config, "sr_timestep"):
    config.sr_timestep = 1000.0
if not hasattr(config, "sr_kv_len"):
    config.sr_kv_len = 3
if not hasattr(config, "condition_noise_scale"):
    config.condition_noise_scale = 0.0

lr_only = args.lr_only or getattr(config, "lr_only", False)
dcm_enabled = not args.no_dcm

# ========================= Device =========================
device = torch.device("cuda")
set_seed(config.seed)
print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)
use_fast_path = args.torch_compile or args.fp8
runtime_dtype = torch.float16 if use_fast_path else torch.bfloat16
print(f"Runtime dtype: {runtime_dtype}")
print(f"Sparse SR: {use_sparse_sr}")

# ========================= Load pipeline =========================
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")
        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"[Warning] {len(missing)} missing keys: {missing[:5]} ...")
        if unexpected:
            print(f"[Warning] {len(unexpected)} unexpected keys: {unexpected[:5]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# LoRA support
pipeline.is_lora_enabled = False
if getattr(config, "adapter", None):
    from utils.lora_utils import configure_lora_for_model
    import peft
    print(f"LoRA enabled: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=True,
    )
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        print("LoRA weights loaded")
    pipeline.is_lora_enabled = True

pipeline = pipeline.to(dtype=runtime_dtype)

# Inductor optimizations
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.fx_graph_cache = True
torch._inductor.config.triton.unique_kernel_names = True
torch.set_float32_matmul_precision("high")

# Optional FP8
if args.fp8:
    print("Applying FP8 to SF DiT...")
    from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
    quantize_(pipeline.generator, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

if args.torch_compile:
    print("Compiling SF DiT...")
    pipeline.generator.compile(mode="max-autotune-no-cudagraphs")

# ========================= Tiny Decoder =========================
if args.tiny_decoder:
    from sr.tiny_decoder import TAEHV, MemBlock

    class TinyDecoderWrapper(torch.nn.Module):
        def __init__(self, ckpt_path, dtype=torch.float16):
            super().__init__()
            self.decoder = TAEHV(checkpoint_path=ckpt_path).to(dtype)
            self._dtype = dtype
            self._mem_cache = None

        @property
        def model(self):
            return self

        def parameters(self, recurse=True):
            return self.decoder.parameters(recurse=recurse)

        def clear_cache(self):
            self._mem_cache = None

        def _warmup_decode(self, x_ntchw):
            """First group: parallel decode + capture MemBlock states."""
            N = x_ntchw.shape[0]
            cache = {}
            hooks = []
            for idx, b in enumerate(self.decoder.decoder):
                if isinstance(b, MemBlock):
                    def _make_hook(idx_):
                        def fn(module, args, output):
                            inp = args[0]
                            T_ = inp.shape[0] // N
                            cache[idx_] = inp.reshape(N, T_, *inp.shape[1:])[:, -1].clone()
                        return fn
                    hooks.append(b.register_forward_hook(_make_hook(idx)))

            pixels = self.decoder.decode_video(x_ntchw, parallel=True, show_progress_bar=False)

            for h in hooks:
                h.remove()
            self._mem_cache = cache
            return pixels

        def _cached_decode(self, x_ntchw):
            """Subsequent groups: sequential decode using cached MemBlock states."""
            N, T, C, H, W = x_ntchw.shape
            x = x_ntchw.reshape(N * T, C, H, W)
            new_cache = {}

            for idx, b in enumerate(self.decoder.decoder):
                if isinstance(b, MemBlock):
                    NT, C_, H_, W_ = x.shape
                    T_ = NT // N
                    _x = x.reshape(N, T_, C_, H_, W_)
                    mem = torch.cat([self._mem_cache[idx].unsqueeze(1), _x[:, :-1]], dim=1)
                    new_cache[idx] = _x[:, -1].clone()
                    x = b(x, mem.reshape(NT, C_, H_, W_))
                else:
                    x = b(x)

            self._mem_cache = new_cache
            NT, C_, H_, W_ = x.shape
            T_ = NT // N
            return x.reshape(N, T_, C_, H_, W_)

        @torch.inference_mode()
        def decode_to_pixel(self, x, use_cache=False, **kwargs):
            x = x.to(dtype=self._dtype)

            if self._mem_cache is None:
                pixels = self._warmup_decode(x)
                pixels = pixels[:, self.decoder.frames_to_trim:]
            else:
                pixels = self._cached_decode(x)

            pixels = pixels.mul_(2).sub_(1).float().clamp_(-1, 1)
            return pixels

    pipeline.vae = TinyDecoderWrapper(args.decoder_ckpt).eval().requires_grad_(False).to(device)
    print(f"[TinyDecoder] HR Decoder loaded: {args.decoder_ckpt}")

# ========================= Load SR =========================
cascade = None
if not lr_only:
    from sr.config import load_config_class_from_pyfile
    from sr.load_utils import load_sr_dit, load_latent_upsampler
    from sr.stream_forward import _stream_forward_causal, _chunk_indices

    print(f"\n--- Loading {'sparse' if use_sparse_sr else 'dense'} SR stack ---")
    sr_cfg = load_config_class_from_pyfile(config.sr_config)()
    sr_cfg.dit_ckpt = config.sr_dit_ckpt
    sr_cfg.dit_ckpt_type = "pt"
    sr_cfg.strict_load = False
    sr_cfg.val_seed = config.seed

    sr_overrides = {
        "hsdp_shard_dim": 1,
        "sp_size": 1,
        "reshard_after_forward": False,
        "enable_activation_checkpointing": False,
    }
    for k, v in sr_overrides.items():
        setattr(sr_cfg, k, v)

    chunk_size = int(getattr(sr_cfg, "stream_chunk_size", 2))
    sr_kv_len = config.sr_kv_len
    if sr_kv_len is not None:
        sr_cfg.stream_kv_len = sr_kv_len
    elif getattr(sr_cfg, "stream_kv_len", None) is None:
        sr_cfg.stream_kv_len = 3
    print(f"[SR] chunk_size={chunk_size}, kv_len={sr_cfg.stream_kv_len}")

    sr_dit = load_sr_dit(sr_cfg, device=device)
    sr_dit.to(device).eval().requires_grad_(False)

    latent_upsampler = load_latent_upsampler(sr_cfg, device=device)
    latent_upsampler.to(device).eval().requires_grad_(False)
    print(f"[SR] SR DiT and latent upsampler loaded")

    if args.fp8_sr:
        print("Applying FP8 to SR DiT...")
        from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
        quantize_(sr_dit, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

    if args.compile_sr_dit:
        print("Compiling SR DiT...")
        sr_dit.compile(mode="max-autotune-no-cudagraphs")

    cascade = CausalCascadeStreamingPipeline(
        sf_pipeline=pipeline,
        latent_upsampler=latent_upsampler,
        sr_dit=sr_dit,
        sr_cfg=sr_cfg,
        sf_device=device,
        sr_device=device,
        sf_chunks_per_sr=config.sf_chunks_per_sr,
        sr_chunk_size=chunk_size,
        sr_kv_len=int(sr_cfg.stream_kv_len),
        sr_timestep=config.sr_timestep,
        condition_noise_scale=config.condition_noise_scale,
        decode_use_cache=True,
    )

# ========================= Dataset =========================
dataset = TextDataset(prompt_path=config.data_path)
print(f"Number of prompts: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=0)
os.makedirs(config.output_folder, exist_ok=True)

# ========================= Inference =========================
for i, batch_data in tqdm(enumerate(dataloader)):
    idx = batch_data["idx"].item()
    prompt = batch_data["prompts"][0]
    prompts = [prompt] * config.num_samples

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104],
        device=device, dtype=runtime_dtype,
    )
    print(f"\n[Prompt {idx}] {prompt[:120]}")

    if lr_only or cascade is None:
        pipeline.kv_cache1 = None
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
        )
        video_out = rearrange(video, "b t c h w -> b t h w c").cpu()
        video_uint8 = (video_out * 255).to(torch.uint8)
        for seed_idx in range(config.num_samples):
            fname = f"{idx}-{seed_idx}_lr.mp4"
            write_video(os.path.join(config.output_folder, fname), video_uint8[seed_idx], fps=16)
            print(f"  Saved LR: {fname}")
    else:
        hr_video, lr_lat, hr_lat = cascade.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
        )
        for seed_idx in range(config.num_samples):
            hr_out = rearrange(hr_video[seed_idx:seed_idx+1], "b t c h w -> b t h w c").cpu()
            hr_uint8 = (hr_out * 255).to(torch.uint8)
            fname = f"{idx}-{seed_idx}_sr.mp4"
            write_video(os.path.join(config.output_folder, fname), hr_uint8[0], fps=16)
            print(f"  Saved HR: {fname}")

            if args.save_lr_video and lr_lat is not None:
                if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
                    pipeline.vae.model.clear_cache()
                lr_video = pipeline.vae.decode_to_pixel(lr_lat[seed_idx:seed_idx+1].to(device), use_cache=False)
                lr_video = (lr_video * 0.5 + 0.5).clamp(0, 1)
                lr_out = rearrange(lr_video, "b t c h w -> b t h w c").cpu()
                lr_uint8 = (lr_out * 255).to(torch.uint8)
                lr_fname = f"{idx}-{seed_idx}_lr.mp4"
                write_video(os.path.join(config.output_folder, lr_fname), lr_uint8[0], fps=16)
                print(f"  Saved LR: {lr_fname}")

    pipeline.kv_cache1 = None
    if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
        pipeline.vae.model.clear_cache()

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
