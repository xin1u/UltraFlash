"""
LongLive (LR causal generator) -> UltraForcing (latent upsampler + causal SR DiT)
Streaming cascade: every N SF chunks (default 6 latent frames) are sent to the SR pipeline.
SR DiT uses warmup (first group joint forward) + subsequent 2-frame sub-chunks (KV cache + temporal_offset).
Latent Upsampler uses sequential mode with cross-group caches, VAE decode uses use_cache=True.

Supports both dense and sparse SR DiT, and Dynamic Cache Management (DCM) optimizations:
  (i)   LR Denoising Step Reduction
  (ii)  Adaptive Cache Refresh (CLIP-IQA+)
  (iii) SR Cache Length Adaptation
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_stream_forward_functions(use_sparse: bool = False):
    """Load the appropriate _stream_forward_causal and _chunk_indices functions."""
    from sr.stream_forward import _stream_forward_causal, _chunk_indices
    return _stream_forward_causal, _chunk_indices


class CausalCascadeStreamingPipeline:
    """SF -> Latent Upsampler -> SR DiT -> WAN VAE, fully streaming (chunked + KV cache)

    With optional DCM (Dynamic Cache Management) optimizations.
    """

    def __init__(
        self,
        sf_pipeline,
        latent_upsampler: torch.nn.Module,
        sr_dit: torch.nn.Module,
        sr_cfg,
        sf_device: torch.device,
        sr_device: torch.device,
        sf_chunks_per_sr: int = 2,
        sr_chunk_size: int = 2,
        sr_kv_len: int = 3,
        sr_timestep: float = 1000.0,
        latent_upsample_scale: float = 2.0,
        condition_noise_scale: Optional[float] = None,
        decode_use_cache: bool = True,
        preproject_sr_context: bool = False,
        debug: bool = False,
        # Sparse SR
        use_sparse_sr: bool = False,
        # DCM parameters
        dcm_enabled: bool = False,
        dcm_lr_steps_subsequent: int = 0,
        dcm_adaptive_refresh: bool = False,
        dcm_iqa_evaluator=None,
        dcm_sr_cache_adapt: bool = False,
        dcm_sr_kv_len_min: int = 1,
        dcm_sr_cache_warmup_groups: int = 2,
        no_context_rerun: bool = False,
    ):
        self.sf_pipeline = sf_pipeline
        self.latent_upsampler = latent_upsampler
        self.sr_dit = sr_dit
        self.sr_cfg = sr_cfg
        self.sf_device = sf_device
        self.sr_device = sr_device

        self.sf_chunks_per_sr = int(sf_chunks_per_sr)
        self.sr_chunk_size = int(sr_chunk_size)
        self.sr_kv_len = int(sr_kv_len)
        self.sr_timestep = float(sr_timestep)
        self.latent_upsample_scale = float(latent_upsample_scale)
        self.decode_use_cache = bool(decode_use_cache)
        self.preproject_sr_context = bool(preproject_sr_context)

        if condition_noise_scale is None:
            cmin = float(getattr(sr_cfg, "condition_noise_min", 0.1))
            cmax = float(getattr(sr_cfg, "condition_noise_max", 0.3))
            condition_noise_scale = 0.5 * (cmin + cmax)
        self.condition_noise_scale = float(condition_noise_scale)

        # Sparse SR
        self.use_sparse_sr = bool(use_sparse_sr)
        self._stream_forward_causal, self._chunk_indices = _load_stream_forward_functions(
            use_sparse=self.use_sparse_sr
        )

        # DCM
        self.dcm_enabled = bool(dcm_enabled)
        self.dcm_lr_steps_subsequent = int(dcm_lr_steps_subsequent)
        self.dcm_adaptive_refresh = bool(dcm_adaptive_refresh)
        self.dcm_iqa_evaluator = dcm_iqa_evaluator
        self.dcm_sr_cache_adapt = bool(dcm_sr_cache_adapt)
        self.dcm_sr_kv_len_min = int(dcm_sr_kv_len_min)
        self.dcm_sr_cache_warmup_groups = int(dcm_sr_cache_warmup_groups)
        self.no_context_rerun = bool(no_context_rerun)

        # DCM statistics
        self._dcm_skipped_refresh = 0
        self._dcm_total_refresh = 0
        self._dcm_lr_steps_saved = 0

    def _get_sr_kv_len_for_group(self, group_idx: int) -> int:
        """(DCM-iii) SR cache length adaptation: warmup uses full window, then shrinks."""
        if not self.dcm_sr_cache_adapt:
            return self.sr_kv_len
        if group_idx < self.dcm_sr_cache_warmup_groups:
            return self.sr_kv_len
        shrink = group_idx - self.dcm_sr_cache_warmup_groups
        return max(self.dcm_sr_kv_len_min, self.sr_kv_len - shrink)

    @staticmethod
    def _sf_prompt_embeds_to_context(prompt_embeds: torch.Tensor) -> List[torch.Tensor]:
        """Convert SF text encoder padded embeds [B, L, D] to per-sample list (strip zero padding)."""
        out: List[torch.Tensor] = []
        for emb in prompt_embeds:
            keep = emb.abs().sum(dim=-1) > 0
            out.append(emb[keep] if torch.any(keep) else emb[:1])
        return out

    @torch.no_grad()
    def _preproject_sr_context(self, sf_prompt_context: List[torch.Tensor]) -> torch.Tensor:
        sr_dit = self.sr_dit
        device = sr_dit.patch_embedding.weight.device
        try:
            text_emb_dtype = next(sr_dit.text_embedding.parameters()).dtype
        except (AttributeError, StopIteration):
            text_emb_dtype = sr_dit.patch_embedding.weight.dtype
        text_len = getattr(sr_dit, "text_len", 512)
        stacked = torch.stack([
            torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))])
            for u in sf_prompt_context
        ]).to(device=device, dtype=text_emb_dtype)
        if stacked.size(-1) != sr_dit.dim:
            stacked = sr_dit.text_embedding(stacked)
        return stacked

    @staticmethod
    def _bicubic_upsample_latents_spatial(latents_bcthw: torch.Tensor, scale: float) -> torch.Tensor:
        b, c, t, h, w = latents_bcthw.shape
        up_h = int(round(h * scale))
        up_w = int(round(w * scale))
        x = latents_bcthw.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).to(torch.float32)
        x = F.interpolate(x, size=(up_h, up_w), mode="bicubic", align_corners=False)
        x = x.to(dtype=latents_bcthw.dtype)
        return x.reshape(b, t, c, up_h, up_w).permute(0, 2, 1, 3, 4).contiguous()

    def _build_condition_y(self, lr_hr_latents_bcthw: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        s = self.condition_noise_scale
        if s <= 0:
            return lr_hr_latents_bcthw
        noise = torch.randn(
            lr_hr_latents_bcthw.shape,
            device=lr_hr_latents_bcthw.device,
            dtype=lr_hr_latents_bcthw.dtype,
            generator=generator,
        )
        return (1.0 - s) * lr_hr_latents_bcthw + s * noise

    def _upsample_latents_streaming(
        self,
        lr_latents_btchw: torch.Tensor,
        upsampler_caches: Optional[list],
    ):
        x = lr_latents_btchw.permute(0, 2, 1, 3, 4)  # B,C,T,H,W
        up_param = next(self.latent_upsampler.parameters())
        x = x.to(device=up_param.device, dtype=up_param.dtype)
        with torch.no_grad():
            hr_correction, new_caches = self.latent_upsampler(
                x,
                parallel=False,
                caches=upsampler_caches,
                return_caches=True,
                detach_caches=True,
            )
        return hr_correction.to(self.sr_device), new_caches

    def _run_sr_for_group(
        self,
        cond_y_bcthw: torch.Tensor,
        prompt_context,
        sr_cache_state,
        global_latent_offset: int,
        is_first_group: bool,
        generator: torch.Generator,
        group_idx: int = 0,
    ):
        _stream_forward_causal = self._stream_forward_causal
        _chunk_indices = self._chunk_indices

        device = next(self.sr_dit.parameters()).device
        dtype = next(self.sr_dit.parameters()).dtype
        b, _, t_lat, _, _ = cond_y_bcthw.shape
        cond_y_bcthw = cond_y_bcthw.to(device=device, dtype=dtype)

        if isinstance(prompt_context, (list, tuple)):
            try:
                text_emb_dtype = next(self.sr_dit.text_embedding.parameters()).dtype
            except (AttributeError, StopIteration):
                text_emb_dtype = dtype
            prompt_context = [u.to(device=device, dtype=text_emb_dtype) for u in prompt_context]
        else:
            prompt_context = prompt_context.to(device=device)

        latents = torch.randn(
            cond_y_bcthw.shape, device=device, dtype=dtype, generator=generator
        )

        t_val = torch.full((b,), self.sr_timestep, device=device)
        patch_t = self.sr_dit.patch_size[0]

        # DCM-iii: adaptive KV length
        effective_kv_len = self._get_sr_kv_len_for_group(group_idx)

        if is_first_group:
            self.sr_dit.clear_cross_kv()

        x_list = [latents[i] for i in range(b)]
        y_list = [cond_y_bcthw[i] for i in range(b)]
        out_list, sr_cache_state = _stream_forward_causal(
            model=self.sr_dit,
            x_list=x_list, t=t_val, context=prompt_context,
            y_list=y_list,
            temporal_offset=global_latent_offset // patch_t,
            kv_len=effective_kv_len,
            cache_state=sr_cache_state,
        )
        flow_pred = torch.stack(out_list, dim=0)

        clean = latents - flow_pred
        return clean, sr_cache_state

    @torch.no_grad()
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ):
        sf = self.sf_pipeline
        sf_device = self.sf_device
        sr_device = self.sr_device
        b, num_frames, c_lat, _, _ = noise.shape

        independent_first_frame = getattr(sf.args, "independent_first_frame", False)
        assert num_frames % sf.num_frame_per_block == 0 or \
            (independent_first_frame and (num_frames - 1) % sf.num_frame_per_block == 0)

        conditional_dict = sf.text_encoder(text_prompts=text_prompts)
        sf_prompt_context = self._sf_prompt_embeds_to_context(conditional_dict["prompt_embeds"])

        if self.preproject_sr_context:
            sr_prompt_context = self._preproject_sr_context(sf_prompt_context)
        else:
            sr_prompt_context = sf_prompt_context

        if sf.kv_cache1 is None:
            sf._initialize_kv_cache(batch_size=b, dtype=noise.dtype, device=sf_device)
            sf._initialize_crossattn_cache(batch_size=b, dtype=noise.dtype, device=sf_device)
        else:
            for i in range(sf.num_transformer_blocks):
                sf.crossattn_cache[i]["is_init"] = False
            for i in range(len(sf.kv_cache1)):
                sf.kv_cache1[i]["global_end_index"].zero_()
                sf.kv_cache1[i]["local_end_index"].zero_()

        current_start_frame = 0
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        if initial_latent is not None:
            zero_t = torch.zeros([b, 1], device=sf_device, dtype=torch.int64)
            if independent_first_frame:
                num_input_blocks = (num_input_frames - 1) // sf.num_frame_per_block
                sf.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict, timestep=zero_t,
                    kv_cache=sf.kv_cache1, crossattn_cache=sf.crossattn_cache,
                    current_start=current_start_frame * sf.frame_seq_length,
                )
                current_start_frame += 1
            else:
                num_input_blocks = num_input_frames // sf.num_frame_per_block
            for _ in range(num_input_blocks):
                ref = initial_latent[:, current_start_frame:current_start_frame + sf.num_frame_per_block]
                sf.generator(
                    noisy_image_or_video=ref,
                    conditional_dict=conditional_dict, timestep=zero_t,
                    kv_cache=sf.kv_cache1, crossattn_cache=sf.crossattn_cache,
                    current_start=current_start_frame * sf.frame_seq_length,
                )
                current_start_frame += sf.num_frame_per_block

        all_num_frames = [sf.num_frame_per_block] * (
            (num_frames - (1 if independent_first_frame and initial_latent is None else 0))
            // sf.num_frame_per_block
        )
        if independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        lr_latent_buffer: List[torch.Tensor] = []
        upsampler_caches: Optional[list] = None
        sr_cache_state = None
        cascade_group_idx = 0
        global_latent_offset = 0

        sr_generator = torch.Generator(device=sr_device).manual_seed(
            int(getattr(self.sr_cfg, "val_seed", getattr(self.sr_cfg, "seed", 0))) + 1
        )

        all_lr_latents: List[torch.Tensor] = []
        all_hr_latents: List[torch.Tensor] = []
        hr_video_chunks: List[torch.Tensor] = []

        if self.decode_use_cache and hasattr(sf.vae, "model") and hasattr(sf.vae.model, "clear_cache"):
            sf.vae.model.clear_cache()
        vae_param = next(sf.vae.model.parameters())

        _t_inference_start = time.time()
        _sf_chunk_idx = 0
        _sf_total_time = 0.0
        _sf_total_latent_frames = 0
        _sr_group_idx_timer = 0
        _sr_total_time = 0.0
        _sr_upsample_total = 0.0
        _sr_dit_total = 0.0
        _sr_vae_total = 0.0
        _sr_total_latent_frames = 0

        # DCM statistics reset
        self._dcm_skipped_refresh = 0
        self._dcm_total_refresh = 0
        self._dcm_lr_steps_saved = 0

        # DCM-i: compute subsequent denoising steps
        full_denoising_steps = sf.denoising_step_list
        num_full_steps = len(full_denoising_steps)
        n_sub = self.dcm_lr_steps_subsequent
        if self.dcm_enabled and 0 < n_sub < num_full_steps:
            if n_sub == 1:
                subsequent_steps = full_denoising_steps[:1]
            else:
                indices = torch.linspace(0, num_full_steps - 1, n_sub).round().long()
                subsequent_steps = full_denoising_steps[indices]
        else:
            subsequent_steps = full_denoising_steps

        def _flush_cascade(force_last: bool = False):
            nonlocal upsampler_caches, sr_cache_state, cascade_group_idx, global_latent_offset
            nonlocal _sr_group_idx_timer, _sr_total_time, _sr_upsample_total, _sr_dit_total, _sr_vae_total, _sr_total_latent_frames
            if not lr_latent_buffer:
                return
            if not force_last and len(lr_latent_buffer) < self.sf_chunks_per_sr:
                return

            _t_sr_group_start = time.time()

            lr_chunk_btchw = torch.cat(lr_latent_buffer, dim=1)
            group_num_frames = lr_chunk_btchw.shape[1]
            all_lr_latents.append(lr_chunk_btchw.detach().cpu())

            # 1) Latent upsampler
            torch.cuda.synchronize()
            _t0 = time.time()
            hr_cond_bcthw, upsampler_caches = self._upsample_latents_streaming(
                lr_chunk_btchw, upsampler_caches
            )
            del lr_chunk_btchw
            torch.cuda.synchronize()
            _t_up = time.time() - _t0
            _sr_upsample_total += _t_up

            cond_y = self._build_condition_y(hr_cond_bcthw, sr_generator)
            del hr_cond_bcthw

            # 2) SR DiT (with adaptive KV length)
            effective_kv = self._get_sr_kv_len_for_group(cascade_group_idx)
            torch.cuda.synchronize()
            _t0 = time.time()
            hr_clean_bcthw, sr_cache_state = self._run_sr_for_group(
                cond_y_bcthw=cond_y,
                prompt_context=sr_prompt_context.to(sr_device) if torch.is_tensor(sr_prompt_context) else [u.to(sr_device) for u in sr_prompt_context],
                sr_cache_state=sr_cache_state,
                global_latent_offset=global_latent_offset,
                is_first_group=(cascade_group_idx == 0),
                generator=sr_generator,
                group_idx=cascade_group_idx,
            )
            del cond_y
            torch.cuda.synchronize()
            _t_dit = time.time() - _t0
            _sr_dit_total += _t_dit

            hr_clean_btchw = hr_clean_bcthw.permute(0, 2, 1, 3, 4).contiguous()
            del hr_clean_bcthw
            all_hr_latents.append(hr_clean_btchw.detach().cpu())

            torch.cuda.empty_cache()

            # 3) HR VAE decode
            torch.cuda.synchronize()
            _t0 = time.time()
            decode_in = hr_clean_btchw.to(device=vae_param.device, dtype=vae_param.dtype)
            del hr_clean_btchw
            video = sf.vae.decode_to_pixel(decode_in, use_cache=self.decode_use_cache)
            del decode_in
            video = (video * 0.5 + 0.5).clamp(0, 1).cpu()
            hr_video_chunks.append(video)
            torch.cuda.synchronize()
            _t_vae = time.time() - _t0
            _sr_vae_total += _t_vae

            torch.cuda.empty_cache()

            _t_sr_group = time.time() - _t_sr_group_start
            _sr_total_time += _t_sr_group
            _sr_total_latent_frames += group_num_frames
            pix_frames = group_num_frames * 4
            sr_fps = pix_frames / _t_sr_group if _t_sr_group > 0 else 0
            print(f"[SR group {_sr_group_idx_timer}] {group_num_frames} lat frames | "
                  f"total {_t_sr_group:.2f}s (upsample {_t_up:.2f}s, DiT {_t_dit:.2f}s, VAE {_t_vae:.2f}s) | "
                  f"{sr_fps:.1f} pix fps | kv_len={effective_kv}")

            global_latent_offset += group_num_frames
            cascade_group_idx += 1
            _sr_group_idx_timer += 1
            lr_latent_buffer.clear()

        is_first_sf_chunk = True
        for current_num_frames in all_num_frames:
            torch.cuda.synchronize()
            _t_sf_chunk_start = time.time()

            noisy_input = noise[
                :,
                current_start_frame - num_input_frames:
                current_start_frame + current_num_frames - num_input_frames,
            ]

            # DCM-i: step reduction for subsequent chunks
            if is_first_sf_chunk or not self.dcm_enabled:
                active_steps = full_denoising_steps
            else:
                active_steps = subsequent_steps

            for index, current_timestep in enumerate(active_steps):
                timestep = torch.ones(
                    [b, current_num_frames], device=sf_device, dtype=torch.int64
                ) * current_timestep
                _, denoised_pred = sf.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict, timestep=timestep,
                    kv_cache=sf.kv_cache1, crossattn_cache=sf.crossattn_cache,
                    current_start=current_start_frame * sf.frame_seq_length,
                )
                if index < len(active_steps) - 1:
                    next_t = active_steps[index + 1]
                    noisy_input = sf.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_t * torch.ones([b * current_num_frames], device=sf_device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])

            if not is_first_sf_chunk and self.dcm_enabled:
                self._dcm_lr_steps_saved += (num_full_steps - len(active_steps))

            # DCM-ii: adaptive cache refresh
            skip_context_rerun = False
            if self.dcm_enabled and self.dcm_adaptive_refresh and not is_first_sf_chunk:
                self._dcm_total_refresh += 1
                if self.dcm_iqa_evaluator is not None:
                    try:
                        score = self.dcm_iqa_evaluator.score_latent(denoised_pred)
                        if score > self.dcm_iqa_evaluator.threshold:
                            skip_context_rerun = True
                            self._dcm_skipped_refresh += 1
                    except Exception:
                        pass

            if not skip_context_rerun and not self.no_context_rerun:
                ctx_t = torch.ones_like(timestep) * sf.args.context_noise
                sf.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict, timestep=ctx_t,
                    kv_cache=sf.kv_cache1, crossattn_cache=sf.crossattn_cache,
                    current_start=current_start_frame * sf.frame_seq_length,
                )

            torch.cuda.synchronize()
            _t_sf_chunk = time.time() - _t_sf_chunk_start
            _sf_total_time += _t_sf_chunk
            _sf_total_latent_frames += current_num_frames
            pix_frames = current_num_frames * 4
            sf_fps = pix_frames / _t_sf_chunk if _t_sf_chunk > 0 else 0
            skip_tag = " [no-rerun]" if self.no_context_rerun else (" [skip-refresh]" if skip_context_rerun else "")
            steps_tag = f" [{len(active_steps)}step]" if len(active_steps) != num_full_steps else ""
            print(f"[SF chunk {_sf_chunk_idx}] {current_num_frames} lat frames ({pix_frames} pix) | "
                  f"{_t_sf_chunk:.2f}s | {sf_fps:.1f} pix fps{steps_tag}{skip_tag}")
            _sf_chunk_idx += 1

            lr_latent_buffer.append(denoised_pred.detach())
            current_start_frame += current_num_frames
            is_first_sf_chunk = False
            _flush_cascade(force_last=False)

        _flush_cascade(force_last=True)

        _t_total = time.time() - _t_inference_start
        _sf_pix_total = _sf_total_latent_frames * 4
        _sr_pix_total = _sr_total_latent_frames * 4
        print("\n" + "=" * 70)
        print(f"[TIMING SUMMARY]")
        print(f"  SF : {_sf_total_time:.2f}s total, {_sf_total_latent_frames} lat frames "
              f"({_sf_pix_total} pix), "
              f"avg {_sf_pix_total / _sf_total_time:.1f} pix fps" if _sf_total_time > 0 else "  SF : 0s")
        print(f"  SR : {_sr_total_time:.2f}s total, {_sr_total_latent_frames} lat frames "
              f"({_sr_pix_total} pix), "
              f"avg {_sr_pix_total / _sr_total_time:.1f} pix fps" if _sr_total_time > 0 else "  SR : 0s")
        if _sr_total_time > 0:
            print(f"    - Upsampler : {_sr_upsample_total:.2f}s ({100*_sr_upsample_total/_sr_total_time:.1f}%)")
            print(f"    - SR DiT    : {_sr_dit_total:.2f}s ({100*_sr_dit_total/_sr_total_time:.1f}%)")
            print(f"    - VAE decode: {_sr_vae_total:.2f}s ({100*_sr_vae_total/_sr_total_time:.1f}%)")
        if self.dcm_enabled:
            print(f"  [DCM] LR steps saved: {self._dcm_lr_steps_saved} forward passes")
            print(f"  [DCM] Cache refresh skipped: {self._dcm_skipped_refresh} / {self._dcm_total_refresh}")
        print("=" * 70 + "\n")

        if self.decode_use_cache and hasattr(sf.vae, "model") and hasattr(sf.vae.model, "clear_cache"):
            sf.vae.model.clear_cache()

        hr_video = torch.cat(hr_video_chunks, dim=1) if hr_video_chunks else torch.empty(0)
        if return_latents:
            lr_lat = torch.cat(all_lr_latents, dim=1) if all_lr_latents else None
            hr_lat = torch.cat(all_hr_latents, dim=1) if all_hr_latents else None
            return hr_video, lr_lat, hr_lat
        return hr_video
