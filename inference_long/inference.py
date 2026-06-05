"""
Ultra Flash: Real-Time High-Resolution Streaming Video Generation
=================================================================
Single-GPU cascaded inference pipeline:
  LR Streaming Generator (Self-Forcing) -> Latent Upsampler -> Sparse SR DiT -> HR Decoder

Features:
  - Dynamic Cache Management (DCM):
    (i)   LR denoising step reduction
    (ii)  Adaptive cache refresh via lightweight IQA
    (iii) SR KV cache length adaptation
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import CausalInferencePipeline
from pipeline.causal_cascade_streaming import CausalCascadeStreamingPipeline
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from sr.config import load_config_class_from_pyfile
from sr.load_utils import load_sr_dit, load_latent_upsampler
from sr.stream_forward import _stream_forward_causal, _chunk_indices

# ====================================================================
# Lightweight IQA Evaluator (for DCM adaptive cache refresh)
# ====================================================================

class LightweightIQAEvaluator:
    """CLIP-IQA+ single-frame evaluator for adaptive cache refresh decisions."""

    def __init__(self, device: torch.device, threshold: float = 0.62,
                 decoder_ckpt: str = None):
        try:
            os.environ.setdefault('PYIQA_HOME', str(SCRIPT_DIR / 'checkpoints'))
            import pyiqa
            self.metric = pyiqa.create_metric(
                "clipiqa+", backbone='RN50', model_type='clipiqa',
            ).to(device).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP-IQA+: {e}. Install pyiqa for DCM adaptive refresh.")
        self.threshold = threshold
        self.device = device

        if decoder_ckpt:
            from sr.tiny_decoder import TAEHV
            self._tiny_dec = TAEHV(checkpoint_path=decoder_ckpt).to(torch.float16).to(device).eval()
        else:
            self._tiny_dec = None

    @torch.inference_mode()
    def score_latent(self, denoised_pred: torch.Tensor) -> float:
        """Score the last frame of denoised latent prediction."""
        if self._tiny_dec is None:
            return 0.0
        last_frame_lat = denoised_pred[:, -1:, :, :, :].to(torch.float16)
        pixels = self._tiny_dec.decode_video(last_frame_lat, parallel=True, show_progress_bar=False)
        pixels = pixels.float().clamp(0, 1)
        frame = pixels[:, -1]
        frame_resized = F.interpolate(frame, size=(224, 224), mode="bilinear", align_corners=False)
        frame_resized = frame_resized.to(self.device)
        return float(self.metric(frame_resized).mean().item())


# ====================================================================
# DCM Cascade Pipeline
# ====================================================================

class DCMSparseCascadePipeline(CausalCascadeStreamingPipeline):
    """Dynamic Cache Management + Sparse SR cascaded streaming pipeline.

    Three optimizations:
      (i)   LR step reduction: subsequent chunks skip one denoising step
      (ii)  Adaptive cache refresh: skip context rerun when IQA score is high
      (iii) SR cache length adaptation: shrink SR KV window as generation progresses
    """

    def __init__(self, *a,
                 dcm_lr_steps_subsequent: int = 4,
                 dcm_adaptive_refresh: bool = False,
                 dcm_iqa_evaluator=None,
                 dcm_sr_cache_adapt: bool = True,
                 dcm_sr_kv_len_min: int = 1,
                 dcm_sr_cache_warmup_groups: int = 2,
                 **kw):
        super().__init__(*a, **kw)
        self.dcm_lr_steps_subsequent = dcm_lr_steps_subsequent
        self.dcm_adaptive_refresh = dcm_adaptive_refresh
        self.dcm_iqa_evaluator = dcm_iqa_evaluator
        self.dcm_sr_cache_adapt = dcm_sr_cache_adapt
        self.dcm_sr_kv_len_min = dcm_sr_kv_len_min
        self.dcm_sr_cache_warmup_groups = dcm_sr_cache_warmup_groups

    def _get_sr_kv_len_for_group(self, group_idx: int) -> int:
        if not self.dcm_sr_cache_adapt:
            return self.sr_kv_len
        if group_idx < self.dcm_sr_cache_warmup_groups:
            return self.sr_kv_len
        shrink = group_idx - self.dcm_sr_cache_warmup_groups
        return max(self.dcm_sr_kv_len_min, self.sr_kv_len - shrink)


# ====================================================================
# Argument Parser
# ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Ultra Flash Inference")
    # Self-Forcing params
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd_4step.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--data_path", type=str, default="prompts/examples.txt")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    # Acceleration
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_sr_dit", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    # SR params
    parser.add_argument("--sr_config", type=str, default="configs/sr_sparse_ultraforcing.py")
    parser.add_argument("--sr_dit_ckpt", type=str, default="checkpoints/sr_dit_sparse.pth")
    parser.add_argument("--sf_chunks_per_sr", type=int, default=2)
    parser.add_argument("--sr_timestep", type=float, default=1000.0)
    parser.add_argument("--sr_kv_len", type=int, default=3)
    parser.add_argument("--condition_noise_scale", type=float, default=0.0)
    parser.add_argument("--save_lr_video", action="store_true")
    parser.add_argument("--lr_only", action="store_true")
    # HR Decoder
    parser.add_argument("--tiny_decoder", action="store_true", default=True)
    parser.add_argument("--decoder_ckpt", type=str, default="checkpoints/tiny_decoder.pth")
    # DCM params
    parser.add_argument("--dcm_lr_steps_subsequent", type=int, default=4)
    parser.add_argument("--dcm_adaptive_refresh", action="store_true", default=False)
    parser.add_argument("--dcm_iqa_threshold", type=float, default=0.62)
    parser.add_argument("--dcm_sr_cache_adapt", action="store_true", default=True)
    parser.add_argument("--dcm_sr_kv_len_min", type=int, default=1)
    parser.add_argument("--dcm_sr_cache_warmup_groups", type=int, default=2)
    parser.add_argument("--no_dcm", action="store_true")
    return parser.parse_args()


# ====================================================================
# Main
# ====================================================================

def main():
    args = parse_args()
    device = torch.device("cuda")
    set_seed(args.seed)

    free_mem = get_cuda_free_memory_gb(gpu)
    print(f"Free VRAM: {free_mem:.1f} GB")
    low_memory = free_mem < 40

    torch.set_grad_enabled(False)
    use_fast_path = args.torch_compile or args.fp8
    runtime_dtype = torch.float16 if use_fast_path else torch.bfloat16
    print(f"Runtime dtype: {runtime_dtype}")

    dcm_enabled = not args.no_dcm
    if dcm_enabled:
        print(f"[DCM] Dynamic Cache Management ENABLED")
        print(f"  (i)   LR step reduction: subsequent={args.dcm_lr_steps_subsequent}")
        print(f"  (ii)  Adaptive cache refresh: {args.dcm_adaptive_refresh}")
        print(f"  (iii) SR cache adaptation: {args.dcm_sr_cache_adapt}")

    # ---- Load Self-Forcing Pipeline ----
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    print(f"Denoising steps: {list(config.denoising_step_list)}")

    pipeline = CausalInferencePipeline(config, device=device)

    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        key = "generator_ema" if args.use_ema else "generator"
        pipeline.generator.load_state_dict(state_dict[key])
        print(f"[SF] Loaded: {args.checkpoint_path} ({key})")

    pipeline = pipeline.to(dtype=runtime_dtype)
    pipeline.generator.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)
    pipeline.vae.eval().requires_grad_(False)

    # Inductor optimizations
    import torch._inductor.config
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.fx_graph_cache = True
    torch.set_float32_matmul_precision("high")

    if args.fp8:
        from torchao.quantization.quant_api import (
            quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor,
        )
        quantize_(pipeline.generator, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    if args.torch_compile:
        print("Compiling SF DiT...")
        pipeline.generator.compile(mode="max-autotune-no-cudagraphs")

    # ---- Load SR Stack ----
    if not args.lr_only:
        print("\n--- Loading SR Stack ---")
        sr_cfg = load_config_class_from_pyfile(args.sr_config)()
        sr_cfg.dit_ckpt = args.sr_dit_ckpt
        sr_cfg.strict_load = False

        chunk_size = int(getattr(sr_cfg, "stream_chunk_size", 2))
        sr_cfg.stream_kv_len = args.sr_kv_len
        print(f"[SR] chunk_size={chunk_size}, kv_len={sr_cfg.stream_kv_len}")

        sr_dit = load_sr_dit(sr_cfg, device=device)
        latent_upsampler = load_latent_upsampler(sr_cfg, device=device)

        if args.compile_sr_dit:
            print("Compiling SR DiT...")
            sr_dit.compile(mode="max-autotune-no-cudagraphs")

    # ---- Tiny Decoder (HR) ----
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

            @torch.inference_mode()
            def decode_to_pixel(self, x, use_cache=False, **kwargs):
                x = x.to(dtype=self._dtype)
                pixels = self.decoder.decode_video(x, parallel=True, show_progress_bar=False)
                pixels = pixels.mul_(2).sub_(1).float().clamp_(-1, 1)
                return pixels

        pipeline.vae = TinyDecoderWrapper(args.decoder_ckpt).eval().requires_grad_(False).to(device)
        print(f"[TinyDecoder] HR Decoder loaded: {args.decoder_ckpt}")

    # ---- IQA Evaluator (optional) ----
    iqa_evaluator = None
    if dcm_enabled and args.dcm_adaptive_refresh and not args.lr_only:
        try:
            iqa_evaluator = LightweightIQAEvaluator(
                device=device, threshold=args.dcm_iqa_threshold,
                decoder_ckpt=args.decoder_ckpt if args.tiny_decoder else None,
            )
            print(f"[DCM] IQA evaluator loaded (threshold={args.dcm_iqa_threshold})")
        except Exception as e:
            print(f"[DCM] IQA evaluator unavailable ({e}), disabling adaptive refresh")

    # ---- Build Cascade Pipeline ----
    if not args.lr_only:
        cascade = DCMSparseCascadePipeline(
            sf_pipeline=pipeline,
            latent_upsampler=latent_upsampler,
            sr_dit=sr_dit,
            sr_cfg=sr_cfg,
            sf_device=device,
            sr_device=device,
            sf_chunks_per_sr=args.sf_chunks_per_sr,
            sr_chunk_size=chunk_size,
            sr_kv_len=args.sr_kv_len,
            sr_timestep=args.sr_timestep,
            condition_noise_scale=args.condition_noise_scale,
            decode_use_cache=True,
            dcm_lr_steps_subsequent=args.dcm_lr_steps_subsequent if dcm_enabled else 4,
            dcm_adaptive_refresh=args.dcm_adaptive_refresh and dcm_enabled,
            dcm_iqa_evaluator=iqa_evaluator,
            dcm_sr_cache_adapt=args.dcm_sr_cache_adapt and dcm_enabled,
            dcm_sr_kv_len_min=args.dcm_sr_kv_len_min,
            dcm_sr_cache_warmup_groups=args.dcm_sr_cache_warmup_groups,
        )

    # ---- Dataset ----
    dataset = TextDataset(prompt_path=args.data_path)
    print(f"Number of prompts: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                            num_workers=0, drop_last=False)
    os.makedirs(args.output_folder, exist_ok=True)

    # ---- Inference Loop ----
    for i, batch_data in tqdm(enumerate(dataloader), total=len(dataset)):
        prompt = batch_data["prompts"][0]
        prompts = [prompt] * args.num_samples

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104],
            device=device, dtype=runtime_dtype,
        )

        if args.lr_only:
            pipeline.kv_cache1 = None
            if hasattr(pipeline.vae, "clear_cache"):
                pipeline.vae.clear_cache()
            lr_video = pipeline.inference(noise=sampled_noise, text_prompts=prompts)
            for seed_idx in range(args.num_samples):
                base = f"{prompt[:100]}-{seed_idx}"
                lr_uint8 = (lr_video[seed_idx] * 255).to(torch.uint8)
                out_path = os.path.join(args.output_folder, f"{base}_lr.mp4")
                write_video(out_path, lr_uint8.permute(0, 2, 3, 1), fps=16)
                print(f"Saved: {out_path}")
        else:
            hr_video, lr_lat, hr_lat = cascade.inference(
                noise=sampled_noise, text_prompts=prompts,
                initial_latent=None, return_latents=True,
            )
            for seed_idx in range(args.num_samples):
                base = f"{prompt[:100]}-{seed_idx}"
                sr_path = os.path.join(args.output_folder, f"{base}_2k.mp4")
                sr_uint8 = (hr_video[seed_idx] * 255).to(torch.uint8)
                write_video(sr_path, sr_uint8.permute(0, 2, 3, 1), fps=16)
                print(f"Saved: {sr_path}")


if __name__ == "__main__":
    main()
