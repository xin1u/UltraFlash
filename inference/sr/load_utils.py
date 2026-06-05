"""
Utility functions for loading SR DiT and Latent Upsampler models.
"""
import torch
import torch.nn as nn
from pathlib import Path


def load_sr_dit(sr_cfg, device: torch.device) -> nn.Module:
    """Load the sparse causal SR DiT model from config and checkpoint."""
    from sr.config import load_config_class_from_pyfile
    from sr.dit_sparse import Transformer3DModel

    arch_config = sr_cfg.dit_arch_config
    params = arch_config.get("params", {})
    model = Transformer3DModel(**params)

    ckpt_path = sr_cfg.dit_ckpt
    if ckpt_path and Path(ckpt_path).exists():
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "generator" in state_dict:
            state_dict = state_dict["generator"]
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        print(f"[SR DiT] Loaded checkpoint: {ckpt_path}")

    precision = getattr(sr_cfg, "dit_precision", "bf16")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    model_dtype = dtype_map.get(precision, torch.bfloat16)
    model.to(device=device, dtype=model_dtype).eval().requires_grad_(False)
    return model


def load_latent_upsampler(sr_cfg, device: torch.device) -> nn.Module:
    """Load the Causal Streaming Latent Upsampler."""
    if not getattr(sr_cfg, "enable_latent_upsampler", False):
        return None

    upsampler_cfg = getattr(sr_cfg, "latent_upsampler_arch_config", None)
    if not upsampler_cfg:
        raise ValueError("SR config must include latent_upsampler_arch_config")

    target = upsampler_cfg.get("target", "sr.latent_upsampler.ultra_latent_up_v1.UltraLatentUpV1")
    params = upsampler_cfg.get("params", {})

    # Dynamic import
    module_path, cls_name = target.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    upsampler = cls(**params)

    precision_key = getattr(sr_cfg, "latent_upsampler_precision", "fp16")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    upsampler_dtype = dtype_map.get(precision_key, torch.float32)
    upsampler = upsampler.to(device=device, dtype=upsampler_dtype)

    ckpt = getattr(sr_cfg, "latent_upsampler_ckpt", None)
    if ckpt and Path(ckpt).exists():
        ckpt_state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "model" in ckpt_state:
            model_sd = ckpt_state["model"]
        else:
            model_sd = ckpt_state
        upsampler.load_state_dict(model_sd, strict=False)
        del ckpt_state, model_sd
        print(f"[Latent Upsampler] Loaded checkpoint: {ckpt}")

    upsampler.eval().requires_grad_(False)
    return upsampler
