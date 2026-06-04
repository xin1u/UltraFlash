from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _conv3x3(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs)


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "silu":
        return nn.SiLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'. Choose from relu/silu/gelu.")


def _build_upsample_head(
    in_channels: int,
    out_channels: int,
    upsample_scale: int,
    activation: str | None = None,
) -> nn.Sequential:
    """
    构建多级 2x PixelShuffle 上采样头，支持任意整数倍率。

    对于整数倍率 s：
    - 先分解为 n 次 2x 上采样（s 中包含的 2 因子），剩余因子 r 用一次 PixelShuffle(r) 完成。
    - 例如 s=2 → 1 次 PS(2)；s=4 → 2 次 PS(2)；s=3 → 1 次 PS(3)；s=6 → PS(2)+PS(3)。
    """
    if upsample_scale <= 0:
        raise ValueError(f"upsample_scale must be positive, got {upsample_scale}.")
    if upsample_scale == 1:
        layers: list[nn.Module] = [_conv3x3(in_channels, out_channels)]
        if activation is not None:
            layers.append(_make_activation(activation))
        return nn.Sequential(*layers)

    factors: list[int] = []
    remaining = upsample_scale
    while remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    if remaining > 1:
        factors.append(remaining)

    layers = []
    current_ch = in_channels
    for i, factor in enumerate(factors):
        is_last = (i == len(factors) - 1)
        next_ch = out_channels if is_last else out_channels
        layers.append(_conv3x3(current_ch, next_ch * factor * factor))
        layers.append(nn.PixelShuffle(factor))
        current_ch = next_ch
    if activation is not None:
        layers.append(_make_activation(activation))
    return nn.Sequential(*layers)


class UltraLatentMemBlock(nn.Module):
    """
    TAEHV 风格的轻量 causal memory block。

    - 空间建模只使用 Conv2d，因此 3x3 卷积只作用在单帧 H/W 上；
    - 时间融合通过 `current feature + previous feature memory` 拼接后卷积实现；
    - 不使用 attention mask / 3D conv，避免未来帧泄露。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            _conv3x3(in_channels * 2, out_channels),
            _make_activation(activation),
            _conv3x3(out_channels, out_channels),
            _make_activation(activation),
            _conv3x3(out_channels, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], dim=1)) + self.skip(x))


class UltraLatentUpsampler(nn.Module):
    """
    带可控倍率空间上采样的 UltraLatent 变体。

    结构：
    - 多级 PixelShuffle 上采样头（支持 2/3/4/... 任意整数倍率）
    - TAEHV-style MemBlock × num_blocks
    - Conv2d 投影回输出通道

    输入:
        [B, in_channels, T, H, W]
    输出:
        [B, out_channels, T, H*scale, W*scale]
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int | None = None,
        mid_channels: int = 128,
        num_blocks: int = 9,
        upsample_scale: int = 2,
        activation: str = "relu",
        residual: bool = False,
        residual_scale: float = 1.0,
        memory_init: Literal["zero", "replicate"] = "zero",
        zero_init_final: bool = True,
        default_parallel: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if memory_init not in ("zero", "replicate"):
            raise ValueError("memory_init must be 'zero' or 'replicate'.")
        if not isinstance(upsample_scale, int) or upsample_scale < 1:
            raise ValueError(f"upsample_scale must be a positive integer, got {upsample_scale}.")

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.upsample_scale = upsample_scale
        self.activation_name = activation
        self.residual = residual
        self.residual_scale = float(residual_scale)
        self.memory_init = memory_init
        self.default_parallel = bool(default_parallel)

        self.pre_shuffle = _build_upsample_head(
            in_channels, mid_channels, upsample_scale, activation=activation,
        )
        self.blocks = nn.ModuleList(
            [UltraLatentMemBlock(mid_channels, mid_channels, activation=activation) for _ in range(num_blocks)]
        )
        self.final = _conv3x3(mid_channels, self.out_channels)
        if zero_init_final:
            nn.init.zeros_(self.final.weight)
            if self.final.bias is not None:
                nn.init.zeros_(self.final.bias)

        if self.residual:
            self.skip_up = _build_upsample_head(
                in_channels, self.out_channels, upsample_scale, activation=None,
            )
        else:
            self.skip_up = None

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        if self.memory_init == "replicate":
            return x
        return torch.zeros_like(x)

    def _forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _, _ = x.shape
        h = rearrange(x, "b c t h w -> (b t) c h w")
        h = self.pre_shuffle(h)

        for block in self.blocks:
            _, c, hh, ww = h.shape
            h_bt = h.reshape(b, t, c, hh, ww)
            if self.memory_init == "replicate":
                first = h_bt[:, :1]
            else:
                first = h_bt[:, :1] * 0
            past = torch.cat([first, h_bt[:, :-1]], dim=1).reshape(h.shape)
            h = block(h, past)

        h = self.final(h)
        return rearrange(h, "(b t) c h w -> b c t h w", b=b, t=t)

    def _forward_sequential(
        self,
        x: torch.Tensor,
        caches: list[torch.Tensor | None] | None = None,
        detach_caches: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, _, t, _, _ = x.shape
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} caches, got {len(caches)}.")

        outputs = []
        block_caches = list(caches)
        for frame_idx in range(t):
            h = self.pre_shuffle(x[:, :, frame_idx])
            next_block_caches: list[torch.Tensor] = []
            for block_idx, block in enumerate(self.blocks):
                past = block_caches[block_idx]
                if past is None:
                    past = self._initial_memory(h)
                else:
                    past = past.to(device=h.device, dtype=h.dtype)
                h_next = block(h, past)
                next_cache = h.detach() if detach_caches else h
                next_block_caches.append(next_cache)
                h = h_next
            block_caches = next_block_caches
            outputs.append(self.final(h))
        out = torch.stack(outputs, dim=2)
        return out, block_caches

    def forward(
        self,
        latent: torch.Tensor,
        parallel: bool | None = None,
        caches: list[torch.Tensor | None] | None = None,
        return_caches: bool = False,
        detach_caches: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if latent.ndim != 5:
            raise ValueError(f"Expected latent shape [B, C, T, H, W], got {latent.shape}.")
        if latent.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {latent.shape[1]}.")

        if parallel is None:
            parallel = self.default_parallel
        if caches is not None and parallel:
            raise ValueError("caches are only supported when parallel=False.")

        if parallel:
            correction = self._forward_parallel(latent)
            new_caches = [None] * len(self.blocks)
        else:
            correction, new_caches = self._forward_sequential(
                latent,
                caches=caches,
                detach_caches=detach_caches,
            )

        if self.residual:
            b, _, t, _, _ = latent.shape
            skip = rearrange(latent, "b c t h w -> (b t) c h w")
            skip = self.skip_up(skip)
            skip = rearrange(skip, "(b t) c h w -> b c t h w", b=b, t=t)
            out = skip + self.residual_scale * correction
        else:
            out = correction

        if return_caches:
            return out, new_caches
        return out
