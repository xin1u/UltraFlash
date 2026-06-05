"""
Sparse causal streaming forward pass for the SR DiT.
Extracted from UltraForcing/configs/causal-sparse-dmd-latentSR/train_causal_sparse_dmd.py
"""
import math
from typing import Optional

import torch
import torch.nn as nn


def _chunk_indices(num_frames: int, chunk_size: int) -> list[slice]:
    """Split latent temporal dimension into chunks."""
    slices = []
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        slices.append(slice(start, end))
        start = end
    return slices


def _compute_sparse_params(model, grid_sizes):
    """
    Compute block-sparse attention parameters for the current resolution.
    """
    top_k_eff = None
    local_num_eff = getattr(model, "sparse_local_num", None)
    local_range_eff = getattr(model, "sparse_local_range", None)

    if not getattr(model, "use_sparse_attn", False) or len(grid_sizes) == 0:
        return top_k_eff, local_num_eff, local_range_eff

    f0, h0, w0 = [int(v) for v in grid_sizes[0].tolist()]
    block_size = tuple(int(v) for v in getattr(model, "sparse_block_size", (2, 8, 8)))
    spatial_blocks = max(1, math.ceil(h0 / block_size[1]) * math.ceil(w0 / block_size[2]))

    if getattr(model, "sparse_top_k", None) is not None and int(model.sparse_top_k) > 0:
        top_k_eff = int(model.sparse_top_k)
    else:
        square_num = spatial_blocks * spatial_blocks
        ratio = float(getattr(model, "sparse_top_k_ratio", 2.0))
        ref_spatial = getattr(model, "sparse_ref_spatial_tokens", None)
        if ref_spatial is not None and ref_spatial > 0:
            ratio = ratio * ref_spatial / max(h0 * w0, 1)
        temporal_blocks = max(1, math.ceil(f0 / block_size[0]))
        max_k = max(int(square_num * temporal_blocks) - 1, 1)
        if ratio <= 0:
            top_k_eff = 0
        else:
            top_k_eff = min(max(int(square_num * ratio), 1), max_k)

    return top_k_eff, local_num_eff, local_range_eff


def _stream_forward_causal(
    model: torch.nn.Module,
    x_list: list[torch.Tensor],
    t: torch.Tensor,
    context: list[torch.Tensor],
    y_list: list[torch.Tensor] | None = None,
    temporal_offset: int = 0,
    kv_len: int | None = None,
    cache_state: list[dict] | None = None,
) -> tuple[list[torch.Tensor], list[dict]]:
    """
    Causal sparse DiT streaming forward.
    Core difference from dense: computes and passes sparse params to each block.
    """
    from sr.dit_sparse.models import sinusoidal_embedding_1d

    if y_list is not None:
        x_list = [torch.cat([u, v], dim=0) for u, v in zip(x_list, y_list)]

    device = model.patch_embedding.weight.device

    x = [model.patch_embedding(u.unsqueeze(0)) for u in x_list]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x]
    )
    if hasattr(model, "_ensure_rope_freqs"):
        model._ensure_rope_freqs(grid_sizes, temporal_offset=temporal_offset, device=device)
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
    seq_len = int(seq_lens.max().item())
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with torch.amp.autocast('cuda', dtype=torch.float32):
        emb = sinusoidal_embedding_1d(model.freq_dim, t).float()
        e = model.time_embedding(emb)
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))

    if isinstance(context, (list, tuple)):
        context = torch.stack([
            torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
            for u in context
        ]).to(device=device)
        if context.size(-1) != model.dim:
            context = model.text_embedding(context)

    if not getattr(model, "_cross_kv_initialized", False):
        model.reinit_cross_kv(context)

    if cache_state is None:
        cache_state = [{"k": None, "v": None} for _ in range(len(model.blocks))]

    top_k_eff, local_num_eff, local_range_eff = _compute_sparse_params(model, grid_sizes)

    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
        freqs=model.freqs, context=context, context_lens=None,
        local_num=local_num_eff, top_k=top_k_eff,
        local_range=local_range_eff, temporal_offset=temporal_offset,
        kv_len=kv_len, train_img=False,
    )

    for block_id, block in enumerate(model.blocks):
        x, new_k, new_v = block(
            x, is_stream=True,
            pre_cache_k=cache_state[block_id]["k"],
            pre_cache_v=cache_state[block_id]["v"],
            **kwargs,
        )
        cache_state[block_id]["k"] = new_k
        cache_state[block_id]["v"] = new_v

    x = model.head(x, e)
    x = model.unpatchify(x, grid_sizes)
    outputs = [u.to(dtype=model.patch_embedding.weight.dtype) for u in x]
    return outputs, cache_state
