"""
Ultra Flash SR Config (Inference Only)
======================================
Sparse causal SR DiT configuration for streaming inference.
"""
from dataclasses import dataclass, field


@dataclass
class SRSparseInferenceConfig:
    """Inference-only config for the sparse causal SR DiT."""

    # Checkpoint paths (relative to repo root)
    dit_ckpt: str = "checkpoints/sr_dit_sparse.pth"
    dit_ckpt_type: str = "pt"
    strict_load: bool = False
    val_seed: int = 0

    # DiT architecture
    dit_arch_config: dict = field(default_factory=lambda: {
        "target": "sr.dit_sparse.Transformer3DModel",
        "params": {
            "model_type": "t2v",
            "patch_size": [1, 2, 2],
            "text_len": 512,
            "in_dim": 32,
            "dim": 1536,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 12,
            "num_layers": 30,
            "window_size": [-1, -1],
            "qk_norm": True,
            "cross_attn_norm": True,
            "eps": 1e-6,
            "use_sparse_attn": True,
            "sparse_causal": True,
            "sparse_block_size": [2, 8, 8],
            "sparse_top_k": None,
            "sparse_top_k_ratio": 1.0,
            "sparse_kv_ratio": 3.0,
            "sparse_local_range": 9,
            "sparse_local_num": None,
            "sparse_use_kernel": True,
            "stream_chunk_size": 2,
            "rope_max_seq_len": 1024,
            "rope_theta": 10000.0,
            "rope_cache_multiple": 1024,
            "sparse_ref_spatial_tokens": 1560,
        },
    })
    dit_precision: str = "bf16"

    # Streaming parameters
    stream_chunk_size: int = 2
    stream_kv_len: int = 3

    # Latent upsampler
    enable_latent_upsampler: bool = True
    latent_upsampler_ckpt: str = "checkpoints/latent_upsampler.pth"
    latent_upsampler_precision: str = "bf16"
    latent_upsampler_arch_config: dict = field(
        default_factory=lambda: {
            "target": "sr.latent_upsampler.ultra_latent_up_v1.UltraLatentUpsampler",
            "params": {
                "in_channels": 16,
                "out_channels": 16,
                "mid_channels": 128,
                "num_blocks": 8,
                "activation": "relu",
                "residual": False,
                "residual_scale": 1.0,
                "memory_init": "replicate",
                "zero_init_final": True,
                "default_parallel": False,
            },
        }
    )

    # Inference overrides
    hsdp_shard_dim: int = 1
    sp_size: int = 1
    reshard_after_forward: bool = False
    enable_activation_checkpointing: bool = False
    enable_flash_attention_3: bool = True
    vae_precision: str = "fp16"
