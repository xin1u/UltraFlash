<p align="center">
<h1 align="center">Ultra Flash ⚡</h1>
<h3 align="center">Scaling Real-Time Streaming Video Generation to High Resolutions</h3>
</p>

<p align="center">
  <a href="#"><b>Paper</b></a> |
  <a href="https://xin1u.github.io/UltraFlash/"><b>Project Page</b></a> |
  <a href="https://github.com/xin1u/UltraFlash"><b>Code (GitHub)</b></a> |
  <a href="#"><b>Models (HuggingFace)</b></a>
</p>

---

<p align="center">
  <img src="assets/show1.png" width="100%">
</p>
<p align="center">
  <img src="assets/show2.png" width="100%">
</p>

**Ultra Flash** is the first framework to achieve **real-time high-resolution streaming video generation**, producing **1K video at ~30 FPS** and **2K video at ~18 FPS** on a single GPU. It cascades three key components after a low-resolution streaming generator:

1. **Architecture-Preserving T2V-to-TV2V SR Training** with AIGC-oriented degradation
2. **Causal Streaming Latent Upsampler** (~2M params, <5% overhead)
3. **Cascaded Streaming Optimization** (sparse distillation + DPO + dynamic cache management)

---

## Requirements

- NVIDIA GPU with **24+ GB** memory (RTX 4090, H200, B200 tested)
- Linux operating system
- 64 GB RAM
- Python 3.10+

## Installation

```bash
conda create -n ultraflash python=3.10 -y
conda activate ultraflash
cd inference
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Block Sparse Attention (CUDA kernel, required for SR DiT)
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
pip install -e .
cd ..

python setup.py develop
```

## Download Checkpoints

```bash
# Wan2.1-T2V-1.3B base model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir inference/wan_models/Wan2.1-T2V-1.3B

# Ultra Flash checkpoints (short video)
huggingface-cli download YOUR_ORG/UltraFlash --local-dir inference/checkpoints

# Ultra Flash checkpoints (long video, optional)
huggingface-cli download YOUR_ORG/UltraFlash-Long --local-dir inference_long/checkpoints

# Symlink shared checkpoints to avoid duplication (optional)
cd inference_long/checkpoints
ln -s ../../inference/checkpoints/sr_dit_sparse.pth .
ln -s ../../inference/checkpoints/latent_upsampler.pth .
ln -s ../../inference/checkpoints/tiny_decoder.pth .
ln -s ../../inference/checkpoints/ultra_decoder_v3.pth .
ln -s ../../inference/wan_models wan_models
```

The `inference/checkpoints/` folder should contain:
| File | Description |
|------|-------------|
| `self_forcing_dmd.pt` | Self-Forcing 4-step streaming generator |
| `sr_dit_sparse.pth` | Sparse causal SR DiT (single-step) |
| `latent_upsampler.pth` | Causal streaming latent upsampler |
| `tiny_decoder.pth` | Tiny Decoder for high-resolution decoding |
| `ultra_decoder_v3.pth` | **(Optional)** Ultra Decoder V3 — improved HR decoder with better texture fidelity |

## Quick Start

### One-click inference (2K resolution, ~18 FPS)
```bash
cd inference
bash inference.sh
```

### Custom inference
```bash
cd inference
python inference.py \
    --config_path configs/self_forcing_dmd_4step.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/examples.txt \
    --output_folder outputs/ \
    --use_ema \
    --tiny_decoder \
    --torch_compile \
    --compile_sr_dit
```

### LR-only mode (480P, for comparison)
```bash
cd inference
python inference.py \
    --config_path configs/self_forcing_dmd_4step.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/examples.txt \
    --output_folder outputs_lr/ \
    --use_ema \
    --lr_only
```

## Long Video Generation

The `inference_long/` directory supports long video generation (~10 seconds, 120 latent frames) using **LongLive** with LoRA adapters and the same SR cascade pipeline.

### Additional checkpoints for long video

The `inference_long/checkpoints/` folder requires:

| File | Description |
|------|-------------|
| `longlive_base.pt` | LongLive base generator (Self-Forcing + long video fine-tuning) |
| `lora.pt` | LoRA adapter for long video coherence (rank 256) |
| `sr_dit_sparse.pth` | Same as `inference/` — can be symlinked |
| `latent_upsampler.pth` | Same as `inference/` — can be symlinked |
| `tiny_decoder.pth` | Same as `inference/` — can be symlinked |

### Run long video inference
```bash
cd inference_long
python inference_sr.py \
    --config_path configs/longlive_inference_sr.yaml \
    --use_sparse_sr \
    --tiny_decoder \
    --torch_compile \
    --compile_sr_dit
```

## Key Arguments

### Short video (`inference/inference.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_ema` | False | Load EMA weights (recommended) |
| `--num_output_frames` | 21 | Number of latent frames (~5s video at 21 frames) |
| `--num_samples` | 1 | Number of samples per prompt |
| `--seed` | 0 | Random seed |
| `--torch_compile` | False | Enable torch.compile for SF DiT (~1.5x speedup) |
| `--compile_sr_dit` | False | Enable torch.compile for SR DiT |
| `--fp8` | False | FP8 quantization for SF DiT |
| `--tiny_decoder` | True | Use Tiny Decoder for HR decoding (faster than Wan VAE) |
| `--decoder_ckpt` | `checkpoints/tiny_decoder.pth` | Path to decoder checkpoint |
| `--ultra_decoder_v3` | False | Use Ultra Decoder V3 (better texture, replaces Tiny Decoder) |
| `--ultra_decoder_v3_ckpt` | `checkpoints/ultra_decoder_v3.pth` | Path to V3 decoder checkpoint |
| `--sr_kv_len` | 3 | SR DiT KV cache window length |
| `--condition_noise_scale` | 0.0 | Noise injected into SR condition |
| `--save_lr_video` | False | Also save LR video output |
| `--lr_only` | False | Only generate LR video (skip SR cascade) |
| `--dcm_lr_steps_subsequent` | 4 | Denoising steps for subsequent chunks (4=full) |
| `--dcm_adaptive_refresh` | False | Enable IQA-based adaptive cache refresh |
| `--no_dcm` | False | Disable all DCM optimizations |

### Long video (`inference_long/inference_sr.py`)

Inherits most arguments from the YAML config. Key CLI overrides:

| Argument | Default | Description |
|----------|---------|-------------|
| `--config_path` | (required) | Path to YAML config (e.g. `configs/longlive_inference_sr.yaml`) |
| `--use_sparse_sr` | False | Use sparse attention for SR DiT (recommended) |
| `--tiny_decoder` | False | Use Tiny Decoder for HR decoding |
| `--torch_compile` | False | Compile SF DiT |
| `--compile_sr_dit` | False | Compile SR DiT |
| `--fp8` | False | FP8 quantization for SF DiT |
| `--fp8_sr` | False | FP8 quantization for SR DiT |

## Dynamic Cache Management (DCM)

Ultra Flash introduces three inference-time optimizations:

| Optimization | Effect | Quality Impact |
|---|---|---|
| **(i) LR Step Reduction** | Saves 1 forward pass per chunk | Negligible |
| **(ii) Adaptive Cache Refresh** | Skips context rerun when quality is sufficient | Negligible |
| **(iii) SR Cache Adaptation** | Reduces SR KV cache memory over time | Minimal |

## Architecture

```
Text Prompt
    │
    ▼
┌─────────────────────────┐
│  Self-Forcing Generator  │  480P, 4 denoising steps, streaming
│  (Wan2.1-1.3B causal)    │
└───────────┬─────────────┘
            │ LR latents (16×60×104)
            ▼
┌─────────────────────────┐
│  Causal Latent Upsampler │  2× spatial upsampling in latent space
│  (~2M params, Conv2D)    │  Causal memory for temporal coherence
└───────────┬─────────────┘
            │ HR latents (16×120×208)
            ▼
┌─────────────────────────┐
│  Sparse SR DiT           │  Single-step denoising, block-sparse attention
│  (1.3B, causal)          │  Adaptive top-k with local window
└───────────┬─────────────┘
            │ Refined HR latents
            ▼
┌─────────────────────────┐
│  Tiny Decoder (HR)       │  Latent → Pixel (960×1664 or 1440×2496)
│  (Causal Memory Network) │  Sequential streaming mode
└───────────┬─────────────┘
            │
            ▼
       2K Video Output (~18 FPS)
```

## Training

Training code is coming soon. See the `train/` directory for updates.

## Citation

```bibtex
@inproceedings{luxury2026ultraflash,
  title={Ultra Flash: Scaling Real-Time Streaming Video Generation to High Resolutions},
  author={Luxury and Huang, Jie and Fan, Zihao and Ma, Xiaoxiao and Li, Yuming and Fu, Siming and Zhuang, Jun-hao and Xue, Zeyue and Li, Haoran and Huang, Haoyang and Duan, Nan},
  booktitle={arXiv preprint},
  year={2026}
}
```

## Acknowledgements

This work builds upon the following excellent open-source projects:

- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) — Block-sparse attention for streaming video super-resolution
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) — Real-time autoregressive video generation with self-rollout training
- [TAEHV](https://github.com/madebyollin/taehv) — Tiny Autoencoder for high-resolution video decoding
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — Video DiT foundation model

We thank the authors for their outstanding contributions to the community.

## License

This project is released under the [Apache 2.0 License](inference/LICENSE).
