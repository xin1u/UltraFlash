# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try: # 尝试导入块稀疏注意力算子
    from block_sparse_attn import block_sparse_attn_func
    BLOCK_SPARSE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    BLOCK_SPARSE_ATTN_AVAILABLE = False

import warnings
import math
import torch.nn.functional as F
from einops import rearrange

_SPARSE_KERNEL_WARNED = False
_SPARSE_DEBUG_ONCE = set()


def _sparse_debug_log_once(key: str, msg: str):
    if key in _SPARSE_DEBUG_ONCE:
        return
    print(f"[SparseDebug] {msg}", flush=True)
    _SPARSE_DEBUG_ONCE.add(key)


def _validate_cache_block_alignment(
    cache: torch.Tensor | None,
    one_len: int,
    *,
    name: str,
) -> None:
    if cache is None:
        return
    if cache.ndim != 3:
        raise ValueError(f"{name} cache must be 3D [num_blocks, block_tokens, dim], got shape {tuple(cache.shape)}.")
    if one_len <= 0:
        raise ValueError(f"{name}: one_len must be positive, got {one_len}.")
    if cache.shape[0] % one_len != 0:
        raise ValueError(
            f"{name} cache block count {cache.shape[0]} is not divisible by one_len={one_len}. "
            "This means KV cache is not aligned to whole temporal chunks."
        )


def _warn_sparse_kernel_unavailable():
    global _SPARSE_KERNEL_WARNED
    if not _SPARSE_KERNEL_WARNED:
        warnings.warn(
            "block_sparse_attn_func not found; using dense fallback instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        _SPARSE_KERNEL_WARNED = True


def _window_partition_3d(x, win): # 张量分块-编码
    """Split a 3D (frames, height, width) token grid into non-overlapping blocks.

    Flash/sparse attention kernels expect tokens grouped by fixed-size windows.
    This helper reshapes + permutes the grid so each window becomes one sequence.
    """
    b, f, h, w, c = x.shape
    wf, wh, ww = win # 窗口大小【2 8 8】
    # reshape into block grid so every block is contiguous in memory
    x = x.view(b, f // wf, wf, h // wh, wh, w // ww, ww, c) # 将张量按照窗口大小进行逻辑拆分
    # bring the block indices together before flattening
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous() # 调整维度顺序：把“块的索引”提到前面，把“块内的像素”排到后面
    # flatten batch + block index into a list of blocks
    return x.view(-1, wf * wh * ww, c) # 将 batch 和所有的块索引合并，将块内的像素展平


def _window_reverse_3d(windows, win, orig):#张量还原-解码
    """Undo `_window_partition_3d` using the original (F, H, W) grid dimensions."""
    f, h, w = orig # 原始视频的 F, H, W
    wf, wh, ww = win 
    nf, nh, nw = f // wf, h // wh, w // ww # 计算各个维度上块的数量
    b = windows.size(0) // (nf * nh * nw) # 通过总数反推批大小 b
    # reshape flat windows back into block grid layout
    x = windows.view(b, nf, nh, nw, wf, wh, ww, -1) # 将展平的 Token 重新恢复成 3D 块的结构
    # interleave temporal/spatial axes to recover the full grid
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous() # 维度交错：将“块索引”和“块内坐标”重新交织在一起
    return x.view(b, f, h, w, -1) # 彻底压平还原回 (Batch, Frame, Height, Width, Channel)


def _pad_3d_grid_to_window(x: torch.Tensor, win: tuple[int, int, int]):
    """
    将 `[B, F, H, W, C]` 的 3D token 网格按窗口大小补齐到最近倍数。

    这样稀疏注意力不再依赖“输入尺寸必须整除 block_size”的硬约束，
    训练时遇到任意 bucket 也能直接工作。
    """
    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor [B, F, H, W, C], got {tuple(x.shape)}.")
    b, f, h, w, c = x.shape
    wf, wh, ww = win
    pad_f = (-f) % wf
    pad_h = (-h) % wh
    pad_w = (-w) % ww
    if pad_f == 0 and pad_h == 0 and pad_w == 0:
        return x, (f, h, w), (f, h, w)

    x_cf = x.permute(0, 4, 1, 2, 3).contiguous()
    x_cf = F.pad(x_cf, (0, pad_w, 0, pad_h, 0, pad_f))
    x = x_cf.permute(0, 2, 3, 4, 1).contiguous()
    return x, (f, h, w), (f + pad_f, h + pad_h, w + pad_w)


@torch.no_grad() # 局部窗口块级掩码
def build_local_block_mask_shifted_vec_normal_slide(
    block_h, # # 空间中垂直和水平方向块的数量
    block_w, 
    win_h=6, # 注意力窗口的大小（以块为单位）
    win_w=6,
    include_self=True, # 是否允许自己关注自己
    device=None,
):
    """Create a boolean mask describing which spatial blocks can attend to which.

    The mask is computed by sliding a window (`win_h` x `win_w`) over spatial blocks,
    mimicking Swin-like local attention. A `True` entry means block j attends block i.
    """
    device = device or torch.device("cpu")
    r = torch.arange(block_h, device=device) # 生成 [0, 1, ..., block_h-1]
    c = torch.arange(block_w, device=device) # 生成 [0, 1, ..., block_w-1]
    yy, xx = torch.meshgrid(r, c, indexing="ij") # 生成二维坐标矩阵
    r_all = yy.reshape(-1) # 展平为所有块的行坐标
    c_all = xx.reshape(-1) #这里是在为每一个“块”建立坐标系。如果视频被切成了 $10 \times 10$ 个块，这里会生成 100 个坐标对，代表每个块在空间中的位置。
    r_half = win_h // 2
    c_half = win_w // 2 #对于每一个块，代码计算了一个以它为中心的“势力范围”。比如窗口是 $6 \times 6$，那么中心块左右各看 3 个单位。
    start_r = r_all - r_half # 窗口在行方向的起点
    end_r = start_r + win_h - 1 # 窗口在行方向的终点
    start_c = c_all - c_half # 窗口在列方向的起点
    end_c = start_c + win_w - 1 # 窗口在列方向的终点
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])# 判断所有块的行坐标是否在目标窗口的行范围内
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])# 判断所有块的列坐标是否在目标窗口的列范围内
    mask = in_row & in_col # 只有行列都在范围内的块才互相可见
    if not include_self:
        mask.fill_diagonal_(False) # 如果不需要自关注，把对角线设为 False
    return mask


# @torch.no_grad()
# def generate_draft_block_mask(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None):
#     assert batch_size == 1, "Only batch_size=1 supported for now"
#     assert local_attn_mask is not None, "local_attn_mask must be provided"
#     avgpool_q = torch.mean(q_w, dim=1)
#     avgpool_k = torch.mean(k_w, dim=1)
#     avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
#     avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
#     q_heads = avgpool_q.permute(1, 0, 2)
#     k_heads = avgpool_k.permute(1, 0, 2)
#     d = avgpool_q.shape[-1]
#     scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(d)

#     repeat_head = scores.shape[0]
#     repeat_len = scores.shape[1] // local_attn_mask.shape[0]
#     repeat_num = scores.shape[2] // local_attn_mask.shape[1]
#     local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
#     local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
#     local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
#     local_attn_mask = local_attn_mask.to(torch.float32)
#     local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float('inf'))
#     local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
#     scores = scores + local_attn_mask

#     attn_map = torch.softmax(scores, dim=-1)
#     attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
#     loop_num, s1, s2 = attn_map.shape
#     flat = attn_map.reshape(loop_num, -1)
#     apply_topk = min(flat.shape[1] - 1, topk)
#     thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
#     thresholds = thresholds.unsqueeze(1)
#     mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
#     mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
#     mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
#     return mask


# Block-Sparse Attention 中的“动态掩码预测”逻辑。它的核心思路是：与其计算所有 Token 之间的注意力（太慢），不如先将 Token 按块（Block）平均，通过计算块与块之间的相似度，动态筛选出最重要的 Top-K 个块。
def _expand_spatial_mask_to_block_layout(
    spatial_mask: torch.Tensor,
    num_query_temporal_blocks: int,
    num_key_temporal_blocks: int,
) -> torch.Tensor:
    """
    将单帧空间块可见性 `[spatial_blocks, spatial_blocks]` 扩展到
    `[query_blocks, key_blocks]` 的块级布局。

    query_blocks = num_query_temporal_blocks * spatial_blocks
    key_blocks   = num_key_temporal_blocks * spatial_blocks
    """
    if spatial_mask.ndim != 2:
        raise ValueError(
            f"spatial_mask must be 2D [spatial_blocks, spatial_blocks], got {tuple(spatial_mask.shape)}."
        )
    return rearrange(
        spatial_mask.unsqueeze(1).unsqueeze(3).expand(
            spatial_mask.shape[0],
            num_query_temporal_blocks,
            spatial_mask.shape[1],
            num_key_temporal_blocks,
        ),
        "sq tq sk tk -> (tq sq) (tk sk)",
    )


@torch.no_grad()
def build_chunked_temporal_causal_mask(
    *,
    num_query_frames: int,
    num_key_frames: int,
    spatial_blocks: int,
    device: torch.device | str,
    chunk_size: int = 3,
    key_prefix_frames: int = 0,
    train_img: bool = False,
    temporal_block_size: int = 1,
) -> torch.Tensor:
    """构造块级因果掩码，兼容 temporal_block_size >= 1。

    当 temporal_block_size == 1 时退化为原来的逐帧因果掩码（CausVid 风格）。
    当 temporal_block_size == 2 时与 FlashVSR 的 win=(2,8,8) 对齐：
    - 每 temporal_block_size 帧合并为一个时间块；
    - chunk_size 以帧为单位指定，内部转换为时间块个数；
    - 因果边界按时间块计算，块 i 可以看到块 0..i（含自身）。

    参数全部以帧为单位，函数内部负责换算到块级。
    """
    if num_query_frames <= 0 or num_key_frames <= 0:
        raise ValueError(
            "num_query_frames and num_key_frames must be positive, "
            f"got {num_query_frames}, {num_key_frames}."
        )
    if spatial_blocks <= 0:
        raise ValueError(f"spatial_blocks must be positive, got {spatial_blocks}.")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if int(key_prefix_frames) < 0:
        raise ValueError(f"key_prefix_frames must be non-negative, got {key_prefix_frames}.")

    tb = max(1, int(temporal_block_size))
    num_q_tblocks = math.ceil(num_query_frames / tb)
    num_k_tblocks = math.ceil(num_key_frames / tb)
    prefix_tblocks = key_prefix_frames // tb
    chunk_tblocks = max(1, chunk_size // tb)

    q_idx = torch.arange(num_q_tblocks, device=device, dtype=torch.long)
    k_idx = torch.arange(num_k_tblocks, device=device, dtype=torch.long)

    q_abs = q_idx + prefix_tblocks
    chunk_end = ((q_abs // chunk_tblocks) + 1) * chunk_tblocks
    temporal_mask = k_idx.unsqueeze(0) < chunk_end.unsqueeze(1)

    if train_img:
        temporal_mask = temporal_mask[:1, :]

    temporal_mask = temporal_mask.repeat_interleave(spatial_blocks, dim=0)
    temporal_mask = temporal_mask.repeat_interleave(spatial_blocks, dim=1)
    return temporal_mask


@torch.no_grad()
def build_window_only_block_mask(
    *,
    nheads: int,
    local_attn_mask: torch.Tensor | None,
    allowed_block_mask: torch.Tensor | None,
    num_query_temporal_blocks: int,
    num_key_temporal_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """纯窗口（local + causal）块级掩码，跳过 Q×K top-k。

    返回 [1, nheads, query_blocks, key_blocks] 布尔掩码。
    """
    combined = None
    if local_attn_mask is not None:
        combined = _expand_spatial_mask_to_block_layout(
            local_attn_mask,
            num_query_temporal_blocks=num_query_temporal_blocks,
            num_key_temporal_blocks=num_key_temporal_blocks,
        )
    if allowed_block_mask is not None:
        combined = allowed_block_mask if combined is None else (combined & allowed_block_mask)
    if combined is None:
        raise ValueError("window-only mask requires local_attn_mask or allowed_block_mask")
    combined = combined.to(device=device, dtype=torch.bool)
    empty_rows = ~combined.any(dim=-1)
    if empty_rows.any():
        diag_len = min(combined.shape[0], combined.shape[1])
        eye = torch.eye(diag_len, device=device, dtype=torch.bool)
        if combined.shape == eye.shape:
            combined = combined | eye
        else:
            combined[empty_rows, 0] = True
    return combined.unsqueeze(0).unsqueeze(0).expand(1, nheads, -1, -1).contiguous()


@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                              q_w, k_w, topk=10, local_attn_mask=None,
                              allowed_block_mask: torch.Tensor | None = None):
    """Select top-k relevant temporal blocks per head using pooled Q/K scores.

    Steps:
      1. Average tokens inside a block, so we only reason about block-level similarity.
      2. Compute attention logits between query blocks and key blocks.
      3. (Optional) Add locality bias mask to restrict spatial neighborhood.
      4. Keep only the top-k logits for each (head, query-block) row.
    Resulting mask tells block_sparse kernels which block pairs should interact.
    """
    assert batch_size == 1, "Only batch_size=1 supported for now"
    # 对块内的 Token 求平均 => [块数, 隐藏维度] q_w 的形状通常是 (块数, 块大小, 特征维度)。通过对 dim=1（块内维度）求平均，将每个块浓缩为一个向量。这样我们只需要计算“块级”的相似度，计算量直接减少了“块大小的平方”倍。
    # pool by averaging tokens within a block => [num_blocks, hidden]
    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_k = torch.mean(k_w, dim=1) 
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)  # (h, Lq, d)
    k_heads = avgpool_k.permute(1, 0, 2)  # (h, Lk, d) 将特征维度拆分为“多头（Heads）”，并将头维度（h）放到最前面，以便后续并行计算每个头各自的注意力分布。
    D = avgpool_q.shape[-1] # 使用 einsum 计算矩阵乘法，得到每个头、每个 Query 块对所有 Key 块的得分
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    # 不加 local_range：local_attn_mask=None 时跳过 locality 约束
    # 把“空间上必须离得近”这个硬约束，和“内容上相关”这个软权重相加。如果空间上太远（-inf），那么最终得分就是 -inf，这个块就会被排除。
    combined_allowed_mask = None
    if local_attn_mask is not None:
        num_query_temporal_blocks = max(1, scores.shape[1] // local_attn_mask.shape[0])
        num_key_temporal_blocks = max(1, scores.shape[2] // local_attn_mask.shape[1])
        combined_allowed_mask = _expand_spatial_mask_to_block_layout(
            local_attn_mask,
            num_query_temporal_blocks=num_query_temporal_blocks,
            num_key_temporal_blocks=num_key_temporal_blocks,
        )
    if allowed_block_mask is not None:
        if allowed_block_mask.shape != scores.shape[-2:]:
            raise ValueError(
                "allowed_block_mask shape mismatch: "
                f"expected {tuple(scores.shape[-2:])}, got {tuple(allowed_block_mask.shape)}."
            )
        combined_allowed_mask = (
            allowed_block_mask
            if combined_allowed_mask is None
            else (combined_allowed_mask & allowed_block_mask)
        )

    if combined_allowed_mask is not None:
        combined_allowed_mask = combined_allowed_mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~combined_allowed_mask.unsqueeze(0), -float("inf"))

    # convert logits to probabilities so that the threshold is comparable per row
    attn_map = torch.softmax(scores, dim=-1) # 转化为概率分布
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)

    loop_num, s1, s2 = attn_map.shape 
    flat = attn_map.reshape(loop_num, -1) # 展平每一行

    apply_topk = min(flat.shape[1] - 1, topk) # 找每行top k最大值作为阈值
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)

    mask_new = (flat >= thresholds).reshape(loop_num, s1, s2) # 大于阈值的才设为1
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
    if combined_allowed_mask is not None:
        mask_new = mask_new & combined_allowed_mask.unsqueeze(0)

    # 防止某一行因为 top-k 或合法域过小而变成空行。
    empty_rows = ~mask_new.any(dim=-1)
    if empty_rows.any():
        if combined_allowed_mask is not None:
            fallback = combined_allowed_mask.unsqueeze(0).expand_as(mask_new)
            first_valid = fallback.float().argmax(dim=-1)
            head_idx, row_idx = empty_rows.nonzero(as_tuple=True)
            mask_new[head_idx, row_idx] = False
            mask_new[head_idx, row_idx, first_valid[head_idx, row_idx]] = True
        else:
            # 无约束时至少保留 query 自身对应块。
            diag_len = min(mask_new.shape[-2], mask_new.shape[-1])
            eye = torch.eye(diag_len, device=mask_new.device, dtype=torch.bool)
            mask_new[..., :diag_len, :diag_len] |= eye

    # 训练场景 q/k 通常等长；此时保底保留对角块，避免任何 query 完全失联。
    if mask_new.shape[-2] == mask_new.shape[-1]:
        mask_new.diagonal(dim1=-2, dim2=-1).fill_(True)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1) # 增加b维度的B个复制掩码
    # expand 不会真正分配新内存，只改变元数据视图，更高效
    # mask = mask_new.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return mask # 返回 [Batch, Head, Query_Block, Key_Block] 的布尔掩码



def _block_sparse_attn(reorder_q, reorder_k, reorder_v, attention_mask, num_heads):
    """Call the CUDA block-sparse kernel on already-reordered Q/K/V."""
    orig_dtype = reorder_q.dtype
    b, s_q, c = reorder_q.shape # b: 批大小, s_q/s_k: Query和Key的块序列长度, c: 隐藏层维度
    _, s_k, _ = reorder_k.shape
    head_dim = c // num_heads # 每个头的维度 = 总维度 / 头数
    q = reorder_q.view(b, s_q, num_heads, head_dim).reshape(b * s_q, num_heads, head_dim)
    k = reorder_k.view(b, s_k, num_heads, head_dim).reshape(b * s_k, num_heads, head_dim)
    v = reorder_v.view(b, s_k, num_heads, head_dim).reshape(b * s_k, num_heads, head_dim) #通常不直接处理 4D 张量，而是处理展平后的 3D 张量，通过偏移量（seqlens）来区分不同的 Batch，这样可以更灵活地处理变长序列。
    if q.dtype not in (torch.float16, torch.bfloat16):
        target_dtype = torch.bfloat16 if q.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        _sparse_debug_log_once(
            "kernel_cast",
            f"kernel path cast q/k/v from {orig_dtype} to {target_dtype}; "
            f"reorder_q={tuple(reorder_q.shape)}, reorder_k={tuple(reorder_k.shape)}, mask={tuple(attention_mask.shape)}",
        )
        q = q.to(dtype=target_dtype)
        k = k.to(dtype=target_dtype)
        v = v.to(dtype=target_dtype)
    else:
        _sparse_debug_log_once(
            "kernel_dtype_ok",
            f"kernel path dtype={q.dtype}; reorder_q={tuple(reorder_q.shape)}, "
            f"reorder_k={tuple(reorder_k.shape)}, mask={tuple(attention_mask.shape)}",
        )
    cu_seqlens_q = torch.tensor([0, s_q], device=q.device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, s_k], device=q.device, dtype=torch.int32) # cu_seqlens 指的是 Cumulative Sequence Lengths（累计序列长度）。这里 [0, s_q] 表示第 0 个 Batch 从索引 0 开始，到索引 s_q 结束。这告诉 CUDA 内核在显存的哪个位置切换 Batch
    head_mask_type = torch.tensor([1] * num_heads, device=q.device, dtype=torch.int32) # 定义每个头使用的掩码模式。1 通常代表该头启用传进来的 attention_mask
    x = block_sparse_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        None,
        attention_mask,
        s_q,
        s_k,
        0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False, # 是否为标准因果模式（这里由 mask 处理，所以传 False）
        exact_streaming=False,
        return_attn_probs=False,
    ).unsqueeze(0)
    x = x.view(b, s_q, num_heads, head_dim)
    return x.reshape(b, s_q, c).to(dtype=orig_dtype)


def _block_sparse_attn_fallback(q_w, k_w, v_w, attention_mask, num_heads):
    """Fallback PyTorch implementation when the CUDA kernel or mask is unavailable.
    支持流式路径：q_w 和 k_w 的 block 数可以不同（k_w 含 KV cache）。"""
    block_n_q, block_s, c = q_w.shape
    block_n_k = k_w.shape[0]
    head_dim = c // num_heads
    q_blocks = q_w.view(block_n_q, block_s, num_heads, head_dim)
    k_blocks = k_w.view(block_n_k, block_s, num_heads, head_dim)
    v_blocks = v_w.view(block_n_k, block_s, num_heads, head_dim)
    # attention_mask: [batch=1, heads, query_blocks, key_blocks]
    block_mask = attention_mask[0].any(dim=0)  # [query_blocks, key_blocks]
    out_blocks = torch.zeros_like(q_blocks)
    for blk in range(block_n_q):
        selected = block_mask[blk].nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            selected = torch.tensor([min(blk, block_n_k - 1)], device=q_w.device)
        k_sel = k_blocks[selected].reshape(-1, num_heads, head_dim)
        v_sel = v_blocks[selected].reshape(-1, num_heads, head_dim)
        q_blk = q_blocks[blk]
        q_t = q_blk.permute(1, 0, 2).unsqueeze(0)
        k_t = k_sel.permute(1, 0, 2).unsqueeze(0)
        v_t = v_sel.permute(1, 0, 2).unsqueeze(0)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t)
        out_blocks[blk] = out.squeeze(0).permute(1, 0, 2)
    out_blocks = out_blocks.view(block_n_q, block_s, c)
    return out_blocks.view(1, block_n_q * block_s, c)


def block_sparse_causal_attention(
    q,  # 查询张量，形状[B, L, heads, dim]
    k,  # 键张量，形状同上
    v,  # 值张量
    grid_sizes,  # 每个样本的(F, H, W)尺寸
    block_size=(3, 8, 8),  # 稀疏块的尺寸；时间维默认与 Self-Forcing chunk=3 对齐
    top_k=4,  # 每个块保留的邻居数量
    causal=True,  # 是否启用因果约束
    local_range=9,  # 备用的空间局部窗口
    local_num=None,  # 时间轴可见的块数
    train_img=False,  # 图像训练特判
    local_attn_mask=None,  # 自定义空间掩码
    use_kernel=True,  # 是否使用CUDA核
    causal_chunk_size=3,  # CausVid 风格时间 chunk 大小
):
    """Block-sparse attention that builds masks on-the-fly from Q/K statistics."""
    b, _, n, d = q.shape  # 读取批大小与头维信息
    outputs = []  # 收集每个样本的输出
    for i in range(b):  # 逐样本处理
        f, h, w = grid_sizes[i].tolist()  # 取出当前样本的帧/高/宽
        seq_len = f * h * w  # 计算完整序列长度
        out_i = torch.zeros_like(q[i])  # 预先分配输出缓存
        if seq_len == 0:  # 空序列直接跳过
            outputs.append(out_i)  # 记录零输出
            continue  # 下一个样本

        q_i = q[i, :seq_len]  # 裁剪有效 token
        k_i = k[i, :seq_len]  # 同上
        v_i = v[i, :seq_len]  # 同上
        q_flat = q_i.reshape(seq_len, n * d)  # 合并 head 维，方便窗口重排
        k_flat = k_i.reshape(seq_len, n * d)  # 键同理
        v_flat = v_i.reshape(seq_len, n * d)  # 值同理

        q_5d = q_flat.view(1, f, h, w, n * d)  # 还原回5D体素
        k_5d = k_flat.view(1, f, h, w, n * d)  # 键体素
        v_5d = v_flat.view(1, f, h, w, n * d)  # 值体素

        q_5d, orig_shape, padded_shape = _pad_3d_grid_to_window(q_5d, block_size)
        k_5d, _, _ = _pad_3d_grid_to_window(k_5d, block_size)
        v_5d, _, _ = _pad_3d_grid_to_window(v_5d, block_size)
        f_pad, h_pad, w_pad = padded_shape

        q_w = _window_partition_3d(q_5d, block_size)  # 划分成非重叠窗口
        k_w = _window_partition_3d(k_5d, block_size)  # 键窗口
        v_w = _window_partition_3d(v_5d, block_size)  # 值窗口

        block_n = q_w.shape[0]  # 块数量
        block_s = q_w.shape[1]  # 每块token数
        reorder_q = q_w.view(1, block_n, block_s, n * d).reshape(1, block_n * block_s, n * d)  # 改成块序列
        reorder_k = k_w.view(1, block_n, block_s, n * d).reshape(1, block_n * block_s, n * d)  # 键序列
        reorder_v = v_w.view(1, block_n, block_s, n * d).reshape(1, block_n * block_s, n * d)  # 值序列

        block_h = h_pad // block_size[1]  # 空间上块的行数
        block_w = w_pad // block_size[2]  # 空间上块的列数
        spatial_blocks = block_h * block_w  # 单帧内块数
        temporal_blocks = f_pad // block_size[0]  # 稀疏窗口时间块数量
        query_frames = orig_shape[0]

        causal_block_mask = None
        if causal:
            causal_block_mask = build_chunked_temporal_causal_mask(
                num_query_frames=query_frames,
                num_key_frames=query_frames,
                spatial_blocks=spatial_blocks,
                device=q.device,
                chunk_size=causal_chunk_size,
                key_prefix_frames=0,
                train_img=train_img,
                temporal_block_size=block_size[0],
            )

        if top_k is not None and top_k <= 0:
            attention_mask = build_window_only_block_mask(
                nheads=n,
                local_attn_mask=local_attn_mask,
                allowed_block_mask=causal_block_mask,
                num_query_temporal_blocks=temporal_blocks,
                num_key_temporal_blocks=temporal_blocks,
                device=q.device,
            )
        else:
            attention_mask = generate_draft_block_mask(
                1,  # batch_size 固定为1
                n,  # 头数
                temporal_blocks,  # 时间块数量
                q_w,  # 查询窗口
                k_w,  # 键窗口
                topk=top_k,  # 取top-k 这里是取4，每个块只看四个邻居-显存压缩关键
                local_attn_mask=local_attn_mask,
                allowed_block_mask=causal_block_mask,
            )  # 根据相似度获取候选块对

        kernel_enabled = use_kernel and BLOCK_SPARSE_ATTN_AVAILABLE
        if use_kernel and not BLOCK_SPARSE_ATTN_AVAILABLE:
            _warn_sparse_kernel_unavailable()
            _sparse_debug_log_once(
                "dense_kernel_missing",
                f"dense/sparse train path fallback because kernel missing; "
                f"q_w={tuple(q_w.shape)}, k_w={tuple(k_w.shape)}, block_size={block_size}, mask={tuple(attention_mask.shape)}",
            )
        if not kernel_enabled:
            x = _block_sparse_attn_fallback(q_w, k_w, v_w, attention_mask, num_heads=n)  # 强制使用PyTorch回退
        elif not attention_mask.any(dim=-1).all(): # 这地方有问题，会oom
            _sparse_debug_log_once(
                "dense_mask_empty",
                f"dense/sparse train path fallback because attention_mask has empty rows; "
                f"q_w={tuple(q_w.shape)}, k_w={tuple(k_w.shape)}, block_size={block_size}, mask={tuple(attention_mask.shape)}",
            )
            print(f"⚠️ Warning: Falling back to CPU/PyTorch loop! This will be SLOW and may OOM.")
            x = _block_sparse_attn_fallback(q_w, k_w, v_w, attention_mask, num_heads=n)  # 掩码存在空行也回退
        else:
            _sparse_debug_log_once(
                "dense_kernel_ok",
                f"dense/sparse train path uses CUDA kernel; "
                f"q_w={tuple(q_w.shape)}, k_w={tuple(k_w.shape)}, block_size={block_size}, mask={tuple(attention_mask.shape)}",
            )
            x = _block_sparse_attn(reorder_q, reorder_k, reorder_v, attention_mask, num_heads=n)  # 正常走CUDA核
        # if not kernel_enabled or not attention_mask.any(dim=-1).all():
        #     print(f"⚠️ Warning: Falling back to CPU/PyTorch loop! This will be SLOW and may OOM.")
        x = x.view(1, block_n, block_s, n * d).view(block_n, block_s, n * d)  # 先恢复块结构
        x = _window_reverse_3d(x, block_size, padded_shape)  # 先恢复到补齐后的3D网格
        x = x[:, :orig_shape[0], :orig_shape[1], :orig_shape[2], :]  # 再裁回原始尺寸
        x = x.reshape(1, seq_len, n * d)  # 拉平成序列
        out_i[:seq_len] = x.view(1, seq_len, n, d)[0]  # 写回到输出缓存
        outputs.append(out_i)  # 记录结果

    return torch.stack(outputs, dim=0)  # 拼接所有样本输出


def block_sparse_attention_with_cache(
    q,  # 查询张量
    k,  # 键张量
    v,  # 值张量
    grid_sizes,  # 视频shape列表
    block_size=(3, 8, 8),  # 稀疏块尺寸；时间维默认与 Self-Forcing chunk=3 对齐
    top_k=4,  # 动态top-k
    causal=True,  # 是否因果
    local_range=9,  # 备用空间范围
    local_num=None,  # 时间窗口大小
    train_img=False,  # 图像训练标记
    pre_cache_k=None,  # 传入的键缓存
    pre_cache_v=None,  # 传入的值缓存
    kv_len=None,  # 限制缓存长度
    local_attn_mask=None,  # 额外空间掩码
    use_kernel=True,  # 是否使用CUDA核
    causal_chunk_size=3,  # CausVid 风格时间 chunk 大小
):
    """Streaming variant that supports KV caches and optional truncation."""
    b, _, n, d = q.shape  # 记录批量与头维
    outputs = []  # 保存输出
    cache_k_list = []  # 保存新的K缓存
    cache_v_list = []  # 保存新的V缓存

    def select_cache(cache, idx):
        if cache is None:
            return None  # 无缓存直接返回
        if isinstance(cache, (list, tuple)):
            return cache[idx]  # 若是列表则取对应样本
        return cache  # 否则整个张量复用

    for i in range(b):  # 遍历每个样本
        f, h, w = grid_sizes[i].tolist()  # 当前尺寸
        seq_len = f * h * w  # token 总数
        out_i = torch.zeros_like(q[i])  # 初始化输出
        if seq_len == 0:
            outputs.append(out_i)  # 空样本直接写零
            cache_k_list.append(None)  # 缓存也为空
            cache_v_list.append(None)
            continue  # 处理下一个

        q_i = q[i, :seq_len]  # 截取有效token
        k_i = k[i, :seq_len]
        v_i = v[i, :seq_len]

        q_flat = q_i.reshape(seq_len, n * d)  # flatten head维
        k_flat = k_i.reshape(seq_len, n * d)
        v_flat = v_i.reshape(seq_len, n * d)

        q_5d = q_flat.view(1, f, h, w, n * d)  # 恢复为5D
        k_5d = k_flat.view(1, f, h, w, n * d)
        v_5d = v_flat.view(1, f, h, w, n * d)

        q_5d, orig_shape, padded_shape = _pad_3d_grid_to_window(q_5d, block_size)
        k_5d, _, _ = _pad_3d_grid_to_window(k_5d, block_size)
        v_5d, _, _ = _pad_3d_grid_to_window(v_5d, block_size)
        f_pad, h_pad, w_pad = padded_shape

        q_w = _window_partition_3d(q_5d, block_size)  # 分块
        k_w = _window_partition_3d(k_5d, block_size)
        v_w = _window_partition_3d(v_5d, block_size)

        seqlen = f_pad // block_size[0]  # 时间块数量
        one_len = k_w.shape[0] // seqlen  # 每个时间步包含的空间块数

        cache_k_in = select_cache(pre_cache_k, i)  # 取当前样本的K缓存
        cache_v_in = select_cache(pre_cache_v, i)  # 取V缓存
        _validate_cache_block_alignment(cache_k_in, one_len, name="K")
        _validate_cache_block_alignment(cache_v_in, one_len, name="V")
        if cache_k_in is not None and cache_v_in is not None:
            k_w = torch.cat([cache_k_in, k_w], dim=0)  # 把历史块放在前面
            v_w = torch.cat([cache_v_in, v_w], dim=0)

        block_n = q_w.shape[0]  # 查询块数量
        block_s = q_w.shape[1]  # 单块token数
        block_n_kv = k_w.shape[0]  # 键块数量（含缓存）

        reorder_q = q_w.view(1, block_n, block_s, n * d).reshape(1, block_n * block_s, n * d)  # 重排查询
        reorder_k = k_w.view(1, block_n_kv, block_s, n * d).reshape(1, block_n_kv * block_s, n * d)  # 重排键
        reorder_v = v_w.view(1, block_n_kv, block_s, n * d).reshape(1, block_n_kv * block_s, n * d)  # 重排值

        block_h = h_pad // block_size[1]  # 空间块行数
        block_w = w_pad // block_size[2]  # 空间块列数
        spatial_blocks = block_h * block_w  # 单时间步空间块数
        temporal_blocks = f_pad // block_size[0]  # 当前 query 的稀疏窗口时间块数
        query_frames = orig_shape[0]

        # FlashVSR 的 streaming self-attn 会始终带上空间 local mask；
        # 之前这里被禁掉了，导致我们流式路径和 FlashVSR 分布不一致。
        local_range_eff = local_range if local_range is not None else 9
        if local_attn_mask is None:
            local_attn_mask_i = build_local_block_mask_shifted_vec_normal_slide(
                block_h,
                block_w,
                local_range_eff,
                local_range_eff,
                include_self=True,
                device=q.device,
            )
        else:
            local_attn_mask_i = local_attn_mask.to(device=q.device)


        cache_tblocks = 0 if cache_k_in is None else int(cache_k_in.shape[0] // one_len)
        cache_frames = cache_tblocks * block_size[0]
        key_temporal_blocks = temporal_blocks + cache_tblocks
        causal_block_mask = None
        if causal:
            causal_block_mask = build_chunked_temporal_causal_mask(
                num_query_frames=query_frames,
                num_key_frames=cache_frames + query_frames,
                spatial_blocks=spatial_blocks,
                device=q.device,
                chunk_size=causal_chunk_size,
                key_prefix_frames=cache_frames,
                train_img=train_img,
                temporal_block_size=block_size[0],
            )

        if top_k is not None and top_k <= 0:
            attention_mask = build_window_only_block_mask(
                nheads=n,
                local_attn_mask=local_attn_mask_i,
                allowed_block_mask=causal_block_mask,
                num_query_temporal_blocks=temporal_blocks,
                num_key_temporal_blocks=key_temporal_blocks,
                device=q.device,
            )
        else:
            attention_mask = generate_draft_block_mask(
                1,  # batch=1
                n,  # head数
                temporal_blocks,  # 查询的时间块
                q_w,  # 查询窗口
                k_w,  # 键窗口（含缓存）
                topk=top_k,  # top-k参数
                local_attn_mask=local_attn_mask_i,
                allowed_block_mask=causal_block_mask,
            )  # 根据相似度选出候选

        kernel_enabled = use_kernel and BLOCK_SPARSE_ATTN_AVAILABLE 
        if use_kernel and not BLOCK_SPARSE_ATTN_AVAILABLE:
            _warn_sparse_kernel_unavailable()
            _sparse_debug_log_once(
                "stream_kernel_missing",
                f"stream path fallback because kernel missing; "
                f"q_w={tuple(q_w.shape)}, k_w={tuple(k_w.shape)}, block_size={block_size}, "
                f"mask={tuple(attention_mask.shape)}, kv_len={kv_len}",
            )
        if not kernel_enabled:
            x = _block_sparse_attn_fallback(q_w, k_w, v_w, attention_mask, num_heads=n)  # CPU fallback
        elif not attention_mask.any(dim=-1).all():
            _sparse_debug_log_once(
                "stream_mask_empty",
                f"stream path fallback because attention_mask has empty rows; "
                f"q_w={tuple(q_w.shape)}, k_w={tuple(k_w.shape)}, block_size={block_size}, "
                f"mask={tuple(attention_mask.shape)}, kv_len={kv_len}",
            )
            x = _block_sparse_attn_fallback(q_w, k_w, v_w, attention_mask, num_heads=n)  # 掩码空行 fallback
        else:
            _sparse_debug_log_once(
                "stream_kernel_ok",
                f"stream path uses CUDA kernel; "
                f"q_w={tuple(q_w.shape)}, k_w={tuple(k_w.shape)}, block_size={block_size}, "
                f"mask={tuple(attention_mask.shape)}, kv_len={kv_len}",
            )
            x = _block_sparse_attn(reorder_q, reorder_k, reorder_v, attention_mask, num_heads=n)  # CUDA核

        x = x.view(1, block_n, block_s, n * d).view(block_n, block_s, n * d)  # 还原块结构
        x = _window_reverse_3d(x, block_size, padded_shape)  # 回到补齐后的布局
        x = x[:, :orig_shape[0], :orig_shape[1], :orig_shape[2], :]  # 裁掉补齐区域
        x = x.reshape(1, seq_len, n * d)  # 再展平成序列
        out_i[:seq_len] = x.view(1, seq_len, n, d)[0]  # 写入输出
        outputs.append(out_i)  # 保存样本输出

        if kv_len is None:
            cache_k = k_w  # 不限制则全部保留
            cache_v = v_w
        else:
            kv_len = int(kv_len)
            if kv_len <= 0:
                raise ValueError(f"kv_len must be positive or None, got {kv_len}.")
            cur_block_n = k_w.shape[0]  # 当前块数
            cache_num = cur_block_n // one_len  # 时间轴长度
            if cache_num > kv_len:
                keep_block_n = kv_len * one_len
                cache_k = k_w[-keep_block_n:, :, :]
                cache_v = v_w[-keep_block_n:, :, :]
            else:
                cache_k = k_w  # 否则原样保留
                cache_v = v_w
        cache_k_list.append(cache_k)  # 记录新的缓存
        cache_v_list.append(cache_v)

    out = torch.stack(outputs, dim=0)  # 拼接batch输出
    if b == 1:
        return out, cache_k_list[0], cache_v_list[0]  # 单样本简化返回
    return out, cache_k_list, cache_v_list  # 多样本返回列表

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """Unified wrapper that dispatches to FlashAttention v3, v2, or PyTorch SDPA."""
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    if q.device.type != 'cuda' or q.size(-1) > 256:
        # Fallback to PyTorch SDPA when FlashAttention constraints are not met.
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
        )
        return out.transpose(1, 2).contiguous()

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        # pack padded sequences into a single contiguous buffer
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale  # useful for qk_norm variants

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """User-facing attention helper that prefers FlashAttention if available."""
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None  # PyTorch SDPA currently lacks varlen kernel, so mask is dropped

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
