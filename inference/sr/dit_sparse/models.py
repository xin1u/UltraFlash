# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations
import math
import random

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor
try:
    from torch.distributed.tensor import Replicate, Shard
except Exception:  # pragma: no cover - older torch uses _tensor symbols
    from torch.distributed._tensor import Replicate, Shard


def _is_dtensor(x: object) -> bool:
    return hasattr(x, "to_local") and hasattr(x, "placements") and hasattr(x, "device_mesh")


def _is_replicate_placement(p: object) -> bool:
    return isinstance(p, Replicate) or p.__class__.__name__ == "Replicate"
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import (
    flash_attention,
    block_sparse_causal_attention,
    block_sparse_attention_with_cache,
    build_local_block_mask_shifted_vec_normal_slide,
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_2_AVAILABLE,
)

if FLASH_ATTN_3_AVAILABLE:
    import flash_attn_interface
if FLASH_ATTN_2_AVAILABLE:
    import flash_attn

__all__ = ['WanModel', 'Transformer3DModel', 'WanAttentionBlock', 'sinusoidal_embedding_1d']

_FALLBACK_TEXT_CONTEXT_TOKEN_NUMBER = 512
_FALLBACK_FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2
_FALLBACK_STREAM_CHUNK_SIZE = 8
_FALLBACK_SPARSE_BLOCK_SIZE = (2, 8, 8)


def _resolve_sparse_block_size(sparse_block_size) -> tuple[int, int, int]:
    if sparse_block_size is None:
        sparse_block_size = _FALLBACK_SPARSE_BLOCK_SIZE
    if len(sparse_block_size) != 3:
        raise ValueError(f"sparse_block_size must have 3 elements, got {sparse_block_size}.")
    return tuple(int(v) for v in sparse_block_size)


def _validate_streaming_sparse_alignment(
    *,
    grid_sizes: torch.Tensor,
    sparse_block_size: tuple[int, int, int],
    kv_len: int | None,
    expected_stream_chunk_size: int,
    sparse_causal: bool,
) -> None:
    if grid_sizes.numel() == 0:
        return
    temporal_block = int(sparse_block_size[0])
    if temporal_block <= 0:
        raise ValueError(f"sparse temporal block size must be positive, got {temporal_block}.")
    if int(expected_stream_chunk_size) <= 0:
        raise ValueError(f"stream_chunk_size must be positive, got {int(expected_stream_chunk_size)}.")
    if kv_len is not None and int(kv_len) <= 0:
        raise ValueError(f"stream_kv_len must be positive or None, got {kv_len}.")
    if sparse_causal and temporal_block != 1 and temporal_block != 2:
        raise ValueError(
            "Causal sparse attention supports sparse_block_size[0] in {1, 2}, "
            f"got sparse_block_size={sparse_block_size}."
        )


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000, device=None):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len, device=device),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2, device=device).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def _rope_axis_dims(head_dim: int) -> tuple[int, int, int]:
    """返回 FlashVSR/Wan 风格 3D RoPE 在 T/H/W 三轴上的实数维度。

    head_dim=128 时得到 (44, 42, 42)，对应 complex 频率维度 (22, 21, 21)。
    这种切分方式与 FlashVSR 的 `precompute_freqs_cis_3d()` 保持一致。
    """
    return (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))


def _normalize_temporal_offsets(t_offset, batch_size: int) -> list[int]:
    if t_offset is None:
        return [0] * batch_size
    if isinstance(t_offset, int):
        return [t_offset] * batch_size
    if isinstance(t_offset, torch.Tensor):
        offsets = t_offset.detach().cpu().tolist()
    else:
        offsets = list(t_offset)
    if len(offsets) != batch_size:
        raise ValueError('temporal_offset length must match batch size')
    return [int(v) for v in offsets]


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, t_offset=None):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    t_offsets = _normalize_temporal_offsets(t_offset, grid_sizes.size(0))

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        t_off = int(t_offsets[i])
        if t_off < 0 or t_off + f > freqs[0].size(0):
            raise ValueError(
                f'temporal_offset out of range for rope freqs: offset={t_off}, f={f}, cache={freqs[0].size(0)}. '
                'Call WanModel._ensure_rope_freqs() before rope_apply.'
            )
        if h > freqs[1].size(0) or w > freqs[2].size(0):
            raise ValueError(
                f'spatial size out of range for rope freqs: h={h}, w={w}, cache={freqs[1].size(0)}. '
                'Call WanModel._ensure_rope_freqs() before rope_apply.'
            )

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][t_off:t_off + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(dtype=x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        if _is_dtensor(x):
            orig_dtype = x.dtype
            orig_placements = x.placements
            mesh = x.device_mesh
            if not all(_is_replicate_placement(p) for p in orig_placements):
                x = x.redistribute(placements=[Replicate()] * mesh.ndim)
            x_local = x.to_local()
            x_full = x_local
            expected_dim = self.normalized_shape[0]
            if x_local.shape[-1] != expected_dim and dist.is_initialized():
                mesh_dim = None
                for placement in orig_placements:
                    if placement.__class__.__name__ == "Shard" and hasattr(placement, "dim"):
                        mesh_dim = placement.dim
                        break
                if hasattr(mesh, "get_group"):
                    group = mesh.get_group(mesh_dim=mesh_dim) if mesh_dim is not None else None
                elif hasattr(mesh, "get_all_groups"):
                    all_groups = mesh.get_all_groups()
                    group = all_groups[0] if all_groups else None
                else:
                    group = None
                world_size = dist.get_world_size(group)
                if world_size > 1 and expected_dim % x_local.shape[-1] == 0:
                    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
                    dist.all_gather(gathered, x_local, group=group)
                    x_full = torch.cat(gathered, dim=-1)
            y_local = nn.functional.layer_norm(
                x_full.float(),
                self.normalized_shape,
                weight,
                bias,
                self.eps,
            ).to(dtype=orig_dtype)
            y = DTensor.from_local(
                y_local,
                device_mesh=mesh,
                placements=[Replicate()] * mesh.ndim,
            )
            if y.placements != orig_placements:
                y = y.redistribute(placements=orig_placements)
            return y
        return nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        ).type_as(x)


class WanSelfAttention(nn.Module):
    """MMDiT self-attention layer that can switch between dense FA and sparse mode."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 use_sparse_attn=False,
                 sparse_block_size=None,
                 sparse_top_k=4,
                 sparse_causal=False,  # 预留参数：保持与稀疏因果配置一致，当前类未直接使用
                 sparse_local_range=9,
                 sparse_use_kernel=True,
                 stream_chunk_size=None,
                 text_context_token_number=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_sparse_attn = use_sparse_attn  # 是否启用稀疏注意力（包含因果版本）
        self.sparse_block_size = _resolve_sparse_block_size(sparse_block_size)  # 稀疏块的三维尺寸
        self.sparse_top_k = sparse_top_k  # 每块保留的top-k邻居
        self.sparse_causal = sparse_causal  # 稀疏注意力是否应用因果约束
        self.sparse_local_range = sparse_local_range  # 备用的空间局部窗口
        self.sparse_use_kernel = sparse_use_kernel  # 是否必须使用CUDA稀疏核
        self.stream_chunk_size = int(_FALLBACK_STREAM_CHUNK_SIZE if stream_chunk_size is None else stream_chunk_size)
        self._local_attn_mask = None
        self._local_attn_mask_h = None
        self._local_attn_mask_w = None
        self._local_attn_mask_range = None
        self._sparse_grid_debug_printed = False

        # projection layers for Q/K/V/O
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def reset_sparse_debug_state(self) -> None:
        """重置稀疏调试日志状态；用于每轮 valid/infer 开头重新打印一次。"""
        self._sparse_grid_debug_printed = False

    def _log_sparse_grid_status_once(self, message: str) -> None:
        """同一轮里只打印一次 padding/alignment 状态，避免 validation 刷屏。"""
        if self._sparse_grid_debug_printed:
            return
        print(message, flush=True)
        self._sparse_grid_debug_printed = True

    def forward(self, x, seq_lens, grid_sizes, freqs, top_k=None, local_num=None, local_range=None, temporal_offset=None, train_img=False, is_stream=False, pre_cache_k=None, pre_cache_v=None, kv_len=None):
        """Run attention on variable-resolution videos with optional sparse mode."""
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function (kept as nested fn to share logic)
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        cache_k = None
        cache_v = None

        use_sparse_attn = self.use_sparse_attn and not train_img
        local_attn_mask = None  # 默认不使用空间局部掩码
        if is_stream and use_sparse_attn:
            _validate_streaming_sparse_alignment(
                grid_sizes=grid_sizes,
                sparse_block_size=self.sparse_block_size,
                kv_len=kv_len,
                expected_stream_chunk_size=getattr(self, "stream_chunk_size", _FALLBACK_STREAM_CHUNK_SIZE),
                sparse_causal=self.sparse_causal,
            )
        if use_sparse_attn:  # 根据配置启用稀疏/因果路径
            f0 = grid_sizes[0, 0].item()  # 读取第一条样本的帧数
            h0 = grid_sizes[0, 1].item()  # 高度
            w0 = grid_sizes[0, 2].item()  # 宽度
            needs_padding = (
                f0 % self.sparse_block_size[0] != 0 or
                h0 % self.sparse_block_size[1] != 0 or
                w0 % self.sparse_block_size[2] != 0
            )
            if needs_padding:
                pad_f = (-f0) % self.sparse_block_size[0]
                pad_h = (-h0) % self.sparse_block_size[1]
                pad_w = (-w0) % self.sparse_block_size[2]
                self._log_sparse_grid_status_once(
                    f"[SparseDebug] grid needs padding: is_stream={is_stream}, "
                    f"grid={(f0, h0, w0)}, sparse_block_size={self.sparse_block_size}, "
                    f"pad={(pad_f, pad_h, pad_w)}, use_sparse_attn={self.use_sparse_attn}, "
                    f"sparse_causal={self.sparse_causal}",
                )
            else:
                self._log_sparse_grid_status_once(
                    f"[SparseDebug] grid already aligned: is_stream={is_stream}, "
                    f"grid={(f0, h0, w0)}, sparse_block_size={self.sparse_block_size}",
                )

            # 注意：这里不能再因为“不整除”而提前关闭 sparse。
            # 真正的 padding/crop 逻辑已经放到 attention.py 里做了。
            block_h = math.ceil(h0 / self.sparse_block_size[1])  # padding 后的空间块行数
            block_w = math.ceil(w0 / self.sparse_block_size[2])  # padding 后的空间块列数
            local_range_eff = self.sparse_local_range if local_range is None else local_range  # 生效的局部窗口
            if (self._local_attn_mask is None or  # 缓存为空或
                    self._local_attn_mask_h != block_h or
                    self._local_attn_mask_w != block_w or
                    self._local_attn_mask_range != local_range_eff or
                    self._local_attn_mask.device != x.device):  # 分辨率/设备/窗口变化则重建
                self._local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(
                    block_h,
                    block_w,
                    local_range_eff,
                    local_range_eff,
                    include_self=True,
                    device=x.device,
                )  # 根据 padding 后分辨率生成局部空间掩码
                self._local_attn_mask_h = block_h  # 缓存行数
                self._local_attn_mask_w = block_w  # 缓存列数
                self._local_attn_mask_range = local_range_eff  # 缓存窗口大小
            local_attn_mask = self._local_attn_mask  # 复用缓存的掩码

        if use_sparse_attn:
            top_k_eff = self.sparse_top_k if top_k is None else top_k  # 若外部未指定则使用默认top-k
            if top_k_eff is None or top_k_eff <= 0:
                top_k_eff = 4  # 避免稀疏掩码为空
            local_range_eff = self.sparse_local_range if local_range is None else local_range  # 确定空间窗口
            if is_stream or pre_cache_k is not None or pre_cache_v is not None:  # 推理/缓存模式
                x, cache_k, cache_v = block_sparse_attention_with_cache(
                    q=rope_apply(q, grid_sizes, freqs, t_offset=temporal_offset),  # 应用RoPE后的查询
                    k=rope_apply(k, grid_sizes, freqs, t_offset=temporal_offset),  # RoPE后的键
                    v=v,  # 值不变
                    grid_sizes=grid_sizes,  # 分辨率信息
                    block_size=self.sparse_block_size,  # 稀疏块尺寸
                    top_k=top_k_eff,  # 动态top-k
                    causal=self.sparse_causal,  # 是否使用因果稀疏
                    local_range=local_range_eff,  # 空间局部范围
                    local_num=local_num,  # 时间窗口
                    train_img=train_img,  # 图像模式
                    pre_cache_k=pre_cache_k,  # 传入键缓存
                    pre_cache_v=pre_cache_v,  # 传入值缓存
                    kv_len=kv_len,  # 限制缓存长度
                    local_attn_mask=local_attn_mask,  # 空间掩码
                    use_kernel=self.sparse_use_kernel,  # 是否必须CUDA
                    causal_chunk_size=self.stream_chunk_size,  # 因果 chunk 与 CausVid 保持一致
                )  # 返回注意力输出与更新后的缓存
            else:  # 训练/非缓存场景
                x = block_sparse_causal_attention(  # 直接走静态稀疏因果注意力
                    q=rope_apply(q, grid_sizes, freqs, t_offset=temporal_offset),  # RoPE查询
                    k=rope_apply(k, grid_sizes, freqs, t_offset=temporal_offset),  # RoPE键
                    v=v,  # 值
                    grid_sizes=grid_sizes,  # 分辨率
                    block_size=self.sparse_block_size,  # 稀疏块尺寸
                    top_k=top_k_eff,  # top-k
                    causal=self.sparse_causal,  # 是否因果
                    local_range=local_range_eff,  # 空间窗口
                    local_num=local_num,  # 时间窗口
                    train_img=train_img,  # 图像标记
                    local_attn_mask=local_attn_mask,  # 空间掩码
                    use_kernel=self.sparse_use_kernel,  # 是否用CUDA核
                    causal_chunk_size=self.stream_chunk_size,  # 因果 chunk 与 CausVid 保持一致
                )  # 直接计算稀疏因果注意力
        else:
            # ---- dense path：chunk-level 因果 + 流式 KV cache ----
            roped_q = rope_apply(q, grid_sizes, freqs, t_offset=temporal_offset).type_as(v)
            roped_k = rope_apply(k, grid_sizes, freqs, t_offset=temporal_offset).type_as(v)

            if is_stream:
                # 流式推理：拼历史 KV → flash_attention（KV cache 天然因果）
                if pre_cache_k is not None and pre_cache_v is not None:
                    full_k = torch.cat([pre_cache_k, roped_k], dim=1)
                    full_v = torch.cat([pre_cache_v, v], dim=1)
                else:
                    full_k = roped_k
                    full_v = v
                x = flash_attention(roped_q, full_k, full_v)
                cache_k = full_k
                cache_v = full_v
                if kv_len is not None and kv_len > 0:
                    chunk_tokens = roped_k.shape[1]
                    max_tokens = kv_len * chunk_tokens
                    if cache_k.shape[1] > max_tokens:
                        cache_k = cache_k[:, -max_tokens:]
                        cache_v = cache_v[:, -max_tokens:]

            elif self.sparse_causal:
                # 训练因果：chunk-level causal（CausVid 风格），纯 flash_attn 实现
                # 同 chunk 内双向可见，跨 chunk 因果 — 与 KV cache 推理语义一致。
                # 直接调 flash_attn_varlen_func，绕过 flash_attention wrapper 的 flatten 拷贝。
                f0 = int(grid_sizes[0, 0].item())
                frame_seqlen = int(grid_sizes[0, 1].item()) * int(grid_sizes[0, 2].item())
                chunk_frames = self.stream_chunk_size
                chunk_tokens = chunk_frames * frame_seqlen
                total_tokens = f0 * frame_seqlen
                num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens

                half_dtype = torch.bfloat16
                q_3d = roped_q[0, :total_tokens].to(half_dtype)  # [L, H, D] contiguous
                k_3d = roped_k[0, :total_tokens].to(half_dtype)
                v_3d = v[0, :total_tokens].to(half_dtype)

                chunks_out = []
                for ci in range(num_chunks):
                    qs = ci * chunk_tokens
                    qe = min(qs + chunk_tokens, total_tokens)
                    kve = qe  # chunk i 可见 [0, qe)

                    q_chunk = q_3d[qs:qe]       # contiguous slice
                    k_chunk = k_3d[:kve]         # contiguous slice
                    v_chunk = v_3d[:kve]         # contiguous slice

                    q_len = qe - qs
                    kv_len_i = kve
                    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=q_chunk.device)
                    cu_k = torch.tensor([0, kv_len_i], dtype=torch.int32, device=k_chunk.device)

                    if FLASH_ATTN_3_AVAILABLE:
                        o_chunk = flash_attn_interface.flash_attn_varlen_func(
                            q=q_chunk, k=k_chunk, v=v_chunk,
                            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                            seqused_q=None, seqused_k=None,
                            max_seqlen_q=q_len, max_seqlen_k=kv_len_i,
                            softmax_scale=None, causal=False,
                            deterministic=False,
                        )
                    else:
                        o_chunk = flash_attn.flash_attn_varlen_func(
                            q=q_chunk, k=k_chunk, v=v_chunk,
                            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                            max_seqlen_q=q_len, max_seqlen_k=kv_len_i,
                            dropout_p=0.0, softmax_scale=None, causal=False,
                            deterministic=False,
                        )
                    chunks_out.append(o_chunk)

                x_cat = torch.cat(chunks_out, dim=0)  # [total_tokens, H, D]
                if total_tokens < roped_q.shape[1]:
                    x = roped_q.new_zeros(1, roped_q.shape[1], n, d)
                    x[0, :total_tokens] = x_cat.to(roped_q.dtype)
                else:
                    x = x_cat.unsqueeze(0).to(roped_q.dtype)
            else:
                # 非因果：标准双向 flash attention
                x = flash_attention(
                    q=roped_q, k=roped_k, v=v,
                    k_lens=seq_lens,
                    window_size=self.window_size)

        x = x.to(dtype=self.o.weight.dtype)

        # output projection expects shape [B, L, C]
        x = x.flatten(2)
        x = self.o(x)
        if is_stream:
            return x, cache_k, cache_v
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Cross-attention block that reuses encoder context across diffusion steps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_k = None
        self.cache_v = None

    def init_cache(self, context):
        b = context.size(0)
        n = self.num_heads
        d = self.head_dim
        self.cache_k = self.norm_k(self.k(context)).view(b, -1, n, d)
        self.cache_v = self.v(context).view(b, -1, n, d)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, context, context_lens):
        """Attend from video tokens (`x`) to cached text/image context."""
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if self.cache_k is not None and self.cache_v is not None:
            # re-use cached context from previous call (saves recomputation)
            k = self.cache_k
            v = self.cache_v
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 use_sparse_attn=False,
                 sparse_block_size=None,
                 sparse_top_k=4,
                 sparse_causal=False,  # Block 级别的因果稀疏开关
                 sparse_local_range=9,
                 sparse_use_kernel=True,
                 text_context_token_number=None):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.text_context_token_number = int(
            _FALLBACK_TEXT_CONTEXT_TOKEN_NUMBER if text_context_token_number is None else text_context_token_number
        )
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.cache_k = None
        self.cache_v = None
        self.cache_k_img = None
        self.cache_v_img = None

    def init_cache(self, context):
        image_context_length = context.shape[1] - self.text_context_token_number
        context_img = context[:, :image_context_length]
        context_txt = context[:, image_context_length:]
        b = context.size(0)
        n = self.num_heads
        d = self.head_dim
        self.cache_k = self.norm_k(self.k(context_txt)).view(b, -1, n, d)
        self.cache_v = self.v(context_txt).view(b, -1, n, d)
        self.cache_k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        self.cache_v_img = self.v_img(context_img).view(b, -1, n, d)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_k_img = None
        self.cache_v_img = None

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - self.text_context_token_number
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if self.cache_k is not None and self.cache_v is not None:
            k = self.cache_k
            v = self.cache_v
            k_img = self.cache_k_img
            v_img = self.cache_v_img
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 use_sparse_attn=False,
                 sparse_block_size=None,
                 sparse_top_k=4,
                 sparse_causal=False,  # Block 构造参数中的因果稀疏开关
                 sparse_local_range=9,
                 sparse_use_kernel=True,
                 text_context_token_number=None,
                 stream_chunk_size=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_sparse_attn = use_sparse_attn  # 当前Block是否启用稀疏/因果注意力
        self.sparse_block_size = _resolve_sparse_block_size(sparse_block_size)  # 稀疏块大小
        self.sparse_top_k = sparse_top_k  # 稀疏top-k
        self.sparse_causal = sparse_causal  # 配置是否因果
        self.sparse_local_range = sparse_local_range  # 空间局部范围
        self.sparse_use_kernel = sparse_use_kernel  # 是否要求CUDA核
        self.stream_chunk_size = int(_FALLBACK_STREAM_CHUNK_SIZE if stream_chunk_size is None else stream_chunk_size)
        self.text_context_token_number = int(
            _FALLBACK_TEXT_CONTEXT_TOKEN_NUMBER if text_context_token_number is None else text_context_token_number
        )

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            use_sparse_attn=use_sparse_attn,
            sparse_block_size=sparse_block_size,
            sparse_top_k=sparse_top_k,
            sparse_causal=sparse_causal,  # 将因果稀疏开关传入自注意力
            sparse_local_range=sparse_local_range,
            sparse_use_kernel=sparse_use_kernel,
            stream_chunk_size=self.stream_chunk_size,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps,
                                                                      text_context_token_number=self.text_context_token_number)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        local_num=None,
        top_k=None,
        local_range=None,
        temporal_offset=None,
        train_img=False,
        is_stream=False,
        pre_cache_k=None,
        pre_cache_v=None,
        kv_len=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        modulation = self.modulation
        if isinstance(e, DTensor) and not isinstance(modulation, DTensor):
            if modulation.device != e.device:
                modulation = modulation.to(e.device)
            modulation = DTensor.from_local(
                modulation,
                device_mesh=e.device_mesh,
                placements=e.placements,
            )
        elif isinstance(e, torch.Tensor) and isinstance(modulation, DTensor):
            modulation = modulation.to_local()
        with amp.autocast(dtype=torch.float32):
            e = (modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32
        orig_dtype = x.dtype

        # self-attention
        attn_in = self.norm1(x).float() * (1 + e[1]) + e[0]
        if is_stream:
            y, cache_k, cache_v = self.self_attn(
                attn_in.to(dtype=orig_dtype),
                seq_lens,
                grid_sizes,
                freqs,
                top_k=top_k,
                local_num=local_num,
                local_range=local_range,
                temporal_offset=temporal_offset,
                train_img=train_img,
                is_stream=True,
                pre_cache_k=pre_cache_k,
                pre_cache_v=pre_cache_v,
                kv_len=kv_len,
            )
        else:
            y = self.self_attn(
                attn_in.to(dtype=orig_dtype),
                seq_lens,
                grid_sizes,
                freqs,
                top_k=top_k,
                local_num=local_num,
                local_range=local_range,
                temporal_offset=temporal_offset,
                train_img=train_img,
            )
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]
        x = x.to(dtype=orig_dtype)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            ffn_in = self.norm2(x).float() * (1 + e[4]) + e[3]
            y = self.ffn(ffn_in.to(dtype=orig_dtype))
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x.to(dtype=orig_dtype)

        x = cross_attn_ffn(x, context, context_lens, e)
        if is_stream:
            return x, cache_k, cache_v
        return x

    def reset_sparse_debug_state(self) -> None:
        """透传到 self-attention，便于外部统一在 valid 前重置。"""
        if hasattr(self.self_attn, "reset_sparse_debug_state"):
            self.self_attn.reset_sparse_debug_state()


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False, first_last_frame_context_token_number=None):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            first_last_frame_context_token_number = int(
                _FALLBACK_FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER
                if first_last_frame_context_token_number is None
                else first_last_frame_context_token_number
            )
            self.emb_pos = nn.Parameter(
                torch.zeros(1, first_last_frame_context_token_number, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 text_context_token_number=None,
                 first_last_frame_context_token_number=None,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 use_sparse_attn=False,
                 sparse_block_size=None,
                 sparse_top_k=4,
                 sparse_top_k_ratio=2.0,
                 sparse_causal=False,  # 全局因果稀疏配置
                 sparse_local_range=9,
                 sparse_kv_ratio=None,
                 sparse_local_num=None,
                 sparse_use_kernel=True,
                 stream_chunk_size=None,
                 rope_max_seq_len=1024,
                 rope_theta=10000.0,
                 rope_cache_multiple=1024,
                 # FlashVSR 分辨率外推支持
                 sparse_ref_spatial_tokens=None,
                 sparse_train_local_num_random=True,
                 sparse_train_kv_ratio_random=True):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.text_context_token_number = int(
            text_len if text_context_token_number is None else text_context_token_number
        )
        self.first_last_frame_context_token_number = int(
            _FALLBACK_FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER
            if first_last_frame_context_token_number is None
            else first_last_frame_context_token_number
        )
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_sparse_attn = use_sparse_attn
        self.sparse_block_size = _resolve_sparse_block_size(sparse_block_size)
        self.sparse_top_k = sparse_top_k
        self.sparse_top_k_ratio = sparse_top_k_ratio
        self.sparse_causal = sparse_causal  # 保存是否在全模型中启用因果稀疏
        self.sparse_local_range = sparse_local_range
        self.sparse_use_kernel = sparse_use_kernel
        self.sparse_kv_ratio = sparse_kv_ratio
        self.sparse_local_num = sparse_local_num
        self.stream_chunk_size = int(_FALLBACK_STREAM_CHUNK_SIZE if stream_chunk_size is None else stream_chunk_size)
        self.sparse_ref_spatial_tokens = sparse_ref_spatial_tokens
        self.sparse_train_local_num_random = sparse_train_local_num_random
        self.sparse_train_kv_ratio_random = sparse_train_kv_ratio_random
        # RoPE 动态缓存：与 FlashVSR 的 3D RoPE 切分一致，并允许按当前 f/h/w 和 temporal_offset 外推扩展。
        self.rope_theta = float(rope_theta)
        self.rope_cache_multiple = max(1, int(rope_cache_multiple))
        self.rope_max_seq_len = max(1, int(rope_max_seq_len))

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                window_size,
                qk_norm,
                cross_attn_norm,
                eps,
                use_sparse_attn=use_sparse_attn,
                sparse_block_size=self.sparse_block_size,
                sparse_top_k=sparse_top_k,
                sparse_causal=sparse_causal,  # 把全局因果稀疏开关下发到每个Block
                sparse_local_range=sparse_local_range,
                sparse_use_kernel=sparse_use_kernel,
                text_context_token_number=self.text_context_token_number,
                stream_chunk_size=self.stream_chunk_size,
            )
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.head_dim = dim // num_heads
        self.freqs = self._build_rope_freqs(self.rope_max_seq_len, device=torch.device("cpu"))

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(
                1280,
                dim,
                flf_pos_emb=model_type == 'flf2v',
                first_last_frame_context_token_number=self.first_last_frame_context_token_number,
            )

        # initialize weights
        self.init_weights()
        self._cross_kv_initialized = False

    def _round_rope_cache_len(self, required_len: int) -> int:
        """把 RoPE 缓存长度向上取整，避免长视频流式推理时频繁重建。"""
        required_len = max(1, int(required_len))
        m = self.rope_cache_multiple
        return ((required_len + m - 1) // m) * m

    def _build_rope_freqs(self, max_seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        """构建与 FlashVSR 对齐的 3D RoPE 频率缓存。

        返回形状为 `[max_seq_len, head_dim/2]` 的 complex tensor，内部按
        temporal / height / width 三段拼接。每个轴都缓存到同一长度，forward 时
        再按当前 `(f, h, w)` 动态切片。
        """
        t_dim, h_dim, w_dim = _rope_axis_dims(self.head_dim)
        return torch.cat([
            rope_params(max_seq_len, t_dim, theta=self.rope_theta, device=device),
            rope_params(max_seq_len, h_dim, theta=self.rope_theta, device=device),
            rope_params(max_seq_len, w_dim, theta=self.rope_theta, device=device),
        ], dim=1)

    def _required_rope_seq_len(self, grid_sizes: torch.Tensor, temporal_offset=None) -> int:
        """根据当前 batch 的 f/h/w 与 temporal_offset 动态计算 RoPE 需要的最大长度。"""
        if isinstance(grid_sizes, DTensor):
            grid_sizes = grid_sizes.to_local()
        offsets = _normalize_temporal_offsets(temporal_offset, grid_sizes.size(0))
        required = 1
        for (f, h, w), t_off in zip(grid_sizes.detach().cpu().tolist(), offsets):
            if t_off < 0:
                raise ValueError(f"temporal_offset must be non-negative, got {t_off}")
            # 时间轴使用 offset 后的绝对位置；空间轴直接按当前网格长度切片。
            required = max(required, int(t_off) + int(f), int(h), int(w))
        return required

    def _ensure_rope_freqs(
        self,
        grid_sizes: torch.Tensor,
        temporal_offset=None,
        device: torch.device | None = None,
    ) -> None:
        """动态扩展 RoPE cache，支持时间/空间外推。

        FlashVSR 在每次 forward 根据当前 f/h/w 拼出 freqs；这里保持同样语义，
        只是把三轴频率缓存下来，并在当前 resolution 或 streaming offset 超过
        缓存长度时自动重建更长的 cache。
        """
        device = device or self.patch_embedding.weight.device
        required_len = self._required_rope_seq_len(grid_sizes, temporal_offset)
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if self.freqs.size(0) >= required_len:
            return
        new_len = self._round_rope_cache_len(required_len)
        self.freqs = self._build_rope_freqs(new_len, device=device)

    def clear_cross_kv(self):
        for blk in self.blocks:
            if hasattr(blk.cross_attn, "clear_cache"):
                blk.cross_attn.clear_cache()
        self._cross_kv_initialized = False

    def reset_sparse_debug_state(self) -> None:
        """重置所有注意力块的稀疏调试打印状态。"""
        for blk in self.blocks:
            if hasattr(blk, "reset_sparse_debug_state"):
                blk.reset_sparse_debug_state()

    def reinit_cross_kv(self, context):
        if context is None:
            return
        if context.dim() == 3 and context.size(-1) != self.dim:
            context = self.text_embedding(context)
        for blk in self.blocks:
            if hasattr(blk.cross_attn, "init_cache"):
                blk.cross_attn.init_cache(context)
        self._cross_kv_initialized = True

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        top_k=None,
        local_num=None,
        local_range=None,
        temporal_offset=None,
        kv_len=None,
        train_img=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        dtype = self.patch_embedding.weight.dtype

        x = [u.to(device=device, dtype=dtype) for u in x]
        if y is not None:
            y = [v.to(device=device, dtype=dtype) for v in y]
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        self._ensure_rope_freqs(grid_sizes, temporal_offset=temporal_offset, device=device)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        if context is None:
            raise ValueError("context must be provided for WanModel forward.")
        if isinstance(context, (list, tuple)):
            context = torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]).to(device=device, dtype=dtype)
            if context.size(-1) != self.dim:
                context = self.text_embedding(context)
        elif torch.is_tensor(context):
            if context.dim() == 2:
                context = context.unsqueeze(0)
            if context.dim() == 3 and context.size(-1) != self.dim:
                context = self.text_embedding(context.to(device=device, dtype=dtype))
        else:
            raise TypeError("context must be a tensor or a list of tensors.")

        if clip_fea is not None:
            if not hasattr(self, "img_emb"):
                raise ValueError("clip_fea is provided but model has no image embedding module.")
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            if context is None:
                raise ValueError("clip_fea requires valid context embeddings.")
            context = torch.concat([context_clip, context], dim=1)

        top_k_eff = top_k
        local_num_eff = local_num
        local_range_eff = local_range if local_range is not None else self.sparse_local_range
        kv_len_eff = kv_len
        if self.use_sparse_attn and kv_len_eff is not None:
            kv_len_eff = int(kv_len_eff)
            if kv_len_eff <= 0:
                raise ValueError(f"kv_len must be positive or None, got {kv_len_eff}.")
        if self.use_sparse_attn and len(grid_sizes) > 0:
            f0, h0, w0 = grid_sizes[0].tolist()
            if (f0 % self.sparse_block_size[0] == 0 and
                    h0 % self.sparse_block_size[1] == 0 and
                    w0 % self.sparse_block_size[2] == 0):
                temporal_blocks = max(1, f0 // self.sparse_block_size[0])
                spatial_blocks = max(1, (h0 // self.sparse_block_size[1]) * (w0 // self.sparse_block_size[2]))

                # --- top_k: 支持分辨率自适应缩放 (FlashVSR 风格) ---
                if top_k_eff is None or top_k_eff <= 0:
                    if self.sparse_top_k is not None and self.sparse_top_k > 0:
                        top_k_eff = self.sparse_top_k
                    else:
                        square_num = spatial_blocks * spatial_blocks
                        max_k = max(int(square_num * temporal_blocks) - 1, 1)
                        ratio = self.sparse_top_k_ratio
                        if self.sparse_ref_spatial_tokens is not None and self.sparse_ref_spatial_tokens > 0:
                            cur_spatial_tokens = h0 * w0
                            ratio = ratio * self.sparse_ref_spatial_tokens / max(cur_spatial_tokens, 1)
                        if ratio <= 0:
                            top_k_eff = 0
                        else:
                            top_k_eff = min(max(int(square_num * ratio), 1), max_k)

                # --- local_num: 训练时随机截断因果窗口 (FlashVSR 风格) ---
                if local_num_eff is None:
                    if self.sparse_local_num is not None:
                        local_num_eff = self.sparse_local_num
                    elif self.sparse_causal:
                        local_num_eff = temporal_blocks
                    if local_num_eff is not None and self.training and self.sparse_train_local_num_random and temporal_blocks > 4:
                        r = random.random()
                        if r < 0.3:
                            local_num_eff = max(1, temporal_blocks - 3)
                        elif r < 0.4:
                            local_num_eff = max(1, temporal_blocks - 4)
                        elif r < 0.5:
                            local_num_eff = max(1, temporal_blocks - 2)
                        else:
                            local_num_eff = temporal_blocks
                    if local_num_eff is not None:
                        local_num_eff = max(1, min(int(local_num_eff), temporal_blocks))

                # --- kv_len: 训练时随机化 KV 缓存长度 (FlashVSR 风格) ---
                if kv_len_eff is None and self.sparse_kv_ratio is not None:
                    if self.training and self.sparse_train_kv_ratio_random and local_num_eff is not None and local_num_eff > 4:
                        rand_ratio = (random.uniform(0., 1.0) ** 2) * (local_num_eff - 4) + 2
                        max_kv = max(int(math.ceil(rand_ratio)), 1)
                    else:
                        max_kv = max(int(math.ceil(temporal_blocks * float(self.sparse_kv_ratio))), 1)
                    kv_len_eff = min(max_kv, temporal_blocks)
                if kv_len_eff is None:
                    local_num_for_kv = local_num_eff if local_num_eff is not None else temporal_blocks
                    kv_len_eff = max(1, min(int(local_num_for_kv), temporal_blocks))
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            local_num=local_num_eff,
            top_k=top_k_eff,
            local_range=local_range_eff,
            temporal_offset=temporal_offset,
            train_img=train_img,
            kv_len=kv_len_eff)

        for block_id, block in enumerate(self.blocks):
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        outputs = [u.to(dtype=self.patch_embedding.weight.dtype) for u in x]
        return outputs

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


# Keep backward compatibility with previous naming
Transformer3DModel = WanModel
