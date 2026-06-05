from __future__ import annotations

from collections import namedtuple
from math import log2
from typing import Iterable, Literal, TypeVar

import torch
import torch.nn as nn
from einops import rearrange

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))
T = TypeVar("T")


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


def _validate_power_of_two_scale(name: str, value: int) -> int:
    value = int(value)
    if value not in (1, 2, 4, 8):
        raise ValueError(f"{name} must be one of 1/2/4/8, got {value}.")
    return value


def _num_2x_stages(scale: int) -> int:
    return int(log2(_validate_power_of_two_scale("scale", scale)))


def _expand_stage_value(value: T | Iterable[T], num_stages: int, name: str) -> list[T]:
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) != num_stages:
            raise ValueError(f"{name} must have length {num_stages}, got {len(values)}.")
        return values
    return [value for _ in range(num_stages)]


def _stage_factors_from_total_scale(
    total_scale: int,
    num_stages: int,
    name: str,
    align: Literal["left", "right"],
) -> list[int]:
    total_scale = _validate_power_of_two_scale(name, total_scale)
    num_2x = _num_2x_stages(total_scale)
    factors = [1] * num_stages
    if align == "left":
        indices = range(num_2x)
    elif align == "right":
        indices = range(num_stages - num_2x, num_stages)
    else:
        raise ValueError(f"Unsupported align '{align}'.")
    for idx in indices:
        factors[idx] = 2
    return factors


def _validate_stage_factors(
    value: Iterable[int] | None,
    total_scale: int,
    num_stages: int,
    name: str,
    align: Literal["left", "right"],
) -> list[int]:
    if value is None:
        return _stage_factors_from_total_scale(total_scale, num_stages, name, align=align)
    factors = [int(v) for v in value]
    if len(factors) != num_stages:
        raise ValueError(f"{name}_stage_factors must have length {num_stages}, got {len(factors)}.")
    if any(v not in (1, 2) for v in factors):
        raise ValueError(f"{name}_stage_factors only supports per-stage factors 1 or 2, got {factors}.")
    product = 1
    for v in factors:
        product *= v
    if product != int(total_scale):
        raise ValueError(f"Product of {name}_stage_factors must equal {name}={total_scale}, got {factors}.")
    return factors


class CausalMemBlock(nn.Module):
    """
    TAEHV-style causal memory block.

    当前帧特征只和上一帧 memory 融合，空间建模由 2D 3x3 conv 完成，不使用 3D conv，
    因此不会读未来帧。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        activation: str = "relu",
        memory_init: Literal["zero", "replicate"] = "replicate",
    ) -> None:
        super().__init__()
        if memory_init not in ("zero", "replicate"):
            raise ValueError("memory_init must be 'zero' or 'replicate'.")
        out_channels = out_channels or in_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.memory_init = memory_init
        self.conv = nn.Sequential(
            _conv3x3(self.in_channels * 2, self.out_channels),
            _make_activation(activation),
            _conv3x3(self.out_channels, self.out_channels),
            _make_activation(activation),
            _conv3x3(self.out_channels, self.out_channels),
        )
        self.skip = (
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], dim=1)) + self.skip(x))


class TGrow(nn.Module):
    """
    与 TAEHV 同构的 temporal grow：用 1x1 conv 扩通道后 reshape 成更多时间帧。

    输入 [N*T, C, H, W]，输出 [N*T*stride, C, H, W]。
    """

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")
        self.channels = int(channels)
        self.stride = int(stride)
        self.conv = nn.Conv2d(self.channels, self.channels * self.stride, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"TGrow expected {self.channels} channels, got {c}.")
        x = self.conv(x)
        return x.reshape(nt * self.stride, c, h, w)


class PixelShuffleUpsample(nn.Module):
    """学习型空间上采样：Conv2d 扩通道 + PixelShuffle。"""

    def __init__(self, channels: int, scale_factor: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.scale_factor = int(scale_factor)
        if self.scale_factor not in (1, 2):
            raise ValueError(f"PixelShuffleUpsample scale_factor must be 1 or 2, got {self.scale_factor}.")
        if self.scale_factor == 1:
            self.block = nn.Identity()
        else:
            self.block = nn.Sequential(
                _conv3x3(self.channels, self.channels * self.scale_factor * self.scale_factor),
                nn.PixelShuffle(self.scale_factor),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _apply_causal_model(
    model: nn.Sequential,
    x: torch.Tensor,
    parallel: bool,
    caches: list[object | None] | None = None,
    return_caches: bool = False,
    detach_caches: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, list[object | None]]:
    """
    执行由 Conv/Activation/PixelShuffleUpsample/TGrow/CausalMemBlock
    组成的 causal 模型。

    输入输出统一为 [B, C, T, H, W]。
    """
    if x.ndim != 5:
        raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}.")
    b, c, t, h, w = x.shape

    if parallel:
        if caches is not None:
            raise ValueError("caches are only supported when parallel=False.")
        h_flat = rearrange(x, "b c t h w -> (b t) c h w")
        for block in model:
            if isinstance(block, CausalMemBlock):
                nt, cc, hh, ww = h_flat.shape
                tt = nt // b
                h_bt = h_flat.reshape(b, tt, cc, hh, ww)
                if block.memory_init == "replicate":
                    first = h_bt[:, :1]
                else:
                    first = h_bt[:, :1] * 0
                past = torch.cat([first, h_bt[:, :-1]], dim=1).reshape(h_flat.shape)
                h_flat = block(h_flat, past)
            else:
                h_flat = block(h_flat)
        nt, cc, hh, ww = h_flat.shape
        if nt % b != 0:
            raise ValueError(f"Model produced invalid time dimension: first dim {nt}, batch {b}.")
        tt = nt // b
        out = rearrange(h_flat, "(b t) c h w -> b c t h w", b=b, t=tt)
        if return_caches:
            return out, [None] * len(model)
        return out

    if caches is None:
        caches = [None] * len(model)
    if len(caches) != len(model):
        raise ValueError(f"Expected {len(model)} caches, got {len(caches)}.")

    outputs: list[torch.Tensor] = []
    new_caches = list(caches)
    work_queue = [TWorkItem(x[:, :, i], 0) for i in range(t)]
    while work_queue:
        xt, block_idx = work_queue.pop(0)
        if block_idx == len(model):
            outputs.append(xt)
            continue

        block = model[block_idx]
        if isinstance(block, CausalMemBlock):
            past = new_caches[block_idx]
            if past is None:
                past = xt if block.memory_init == "replicate" else torch.zeros_like(xt)
            if not isinstance(past, torch.Tensor):
                raise TypeError(f"MemBlock cache must be Tensor or None, got {type(past)}.")
            past = past.to(device=xt.device, dtype=xt.dtype)
            xt_next = block(xt, past)
            new_caches[block_idx] = xt.detach() if detach_caches else xt
            work_queue.insert(0, TWorkItem(xt_next, block_idx + 1))
        elif isinstance(block, TGrow):
            xt_next = block(xt)
            _, cc, hh, ww = xt_next.shape
            for frame in reversed(xt_next.view(b, block.stride * cc, hh, ww).chunk(block.stride, dim=1)):
                work_queue.insert(0, TWorkItem(frame, block_idx + 1))
        else:
            work_queue.insert(0, TWorkItem(block(xt), block_idx + 1))

    out = torch.stack(outputs, dim=2)
    if return_caches:
        return out, new_caches
    return out


class UltraLatentUpV3(nn.Module):
    """
    TAEHV-style latent upsampler V3。

    与 V2 结构相同（三阶段 causal pipeline），但将空间上采样从
    nn.Upsample(nearest) 替换为 Conv3x3 + PixelShuffle（学习型上采样）。

    特点：
    - 输入/输出均为 [B, C, T, H, W]；
    - 固定三阶段，排列与 TAEHV decoder 一致；
    - 每个阶段使用 PixelShuffleUpsample（scale_factor 1 或 2）；
    - 每个阶段都有 TGrow，stride 可为 1/2；
    - 每个 stage 可配置 MemBlock 数量；
    - stage 排列顺序：MemBlock ×N -> PixelShuffleUpsample -> TGrow -> Conv transition。
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int | None = None,
        stage_channels: tuple[int, int, int] | list[int] = (256, 128, 64),
        blocks_per_stage: int | tuple[int, ...] | list[int] = 3,
        spatial_scale: int = 2,
        temporal_scale: int = 1,
        spatial_stage_factors: tuple[int, int, int] | list[int] | None = None,
        temporal_stage_factors: tuple[int, int, int] | list[int] | None = None,
        activation: str = "relu",
        memory_init: Literal["zero", "replicate"] = "replicate",
        zero_init_final: bool = False,
        default_parallel: bool = True,
        residual: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.spatial_scale = _validate_power_of_two_scale("spatial_scale", spatial_scale)
        self.temporal_scale = _validate_power_of_two_scale("temporal_scale", temporal_scale)
        self.num_stages = 3
        self.spatial_stage_factors = _validate_stage_factors(
            spatial_stage_factors,
            self.spatial_scale,
            self.num_stages,
            "spatial_scale",
            align="left",
        )
        self.temporal_stage_factors = _validate_stage_factors(
            temporal_stage_factors,
            self.temporal_scale,
            self.num_stages,
            "temporal_scale",
            align="right",
        )
        self.default_parallel = bool(default_parallel)
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)

        if memory_init not in ("zero", "replicate"):
            raise ValueError("memory_init must be 'zero' or 'replicate'.")
        self.memory_init = memory_init

        if len(stage_channels) != self.num_stages:
            raise ValueError(f"stage_channels must have length {self.num_stages}, got {len(stage_channels)}.")
        self.stage_channels = [int(ch) for ch in stage_channels]
        if any(ch <= 0 for ch in self.stage_channels):
            raise ValueError(f"stage_channels must be positive, got {self.stage_channels}.")

        self.blocks_per_stage = [int(v) for v in _expand_stage_value(blocks_per_stage, self.num_stages, "blocks_per_stage")]
        if any(v <= 0 for v in self.blocks_per_stage):
            raise ValueError(f"blocks_per_stage values must be positive, got {self.blocks_per_stage}.")

        layers: list[nn.Module] = [
            _conv3x3(self.in_channels, self.stage_channels[0]),
            _make_activation(activation),
        ]
        current_channels = self.stage_channels[0]
        for stage_idx in range(self.num_stages):
            for _ in range(self.blocks_per_stage[stage_idx]):
                layers.append(
                    CausalMemBlock(
                        current_channels,
                        current_channels,
                        activation=activation,
                        memory_init=memory_init,
                    )
                )

            layers.append(PixelShuffleUpsample(current_channels, scale_factor=self.spatial_stage_factors[stage_idx]))

            layers.append(TGrow(current_channels, stride=self.temporal_stage_factors[stage_idx]))

            if stage_idx + 1 < self.num_stages:
                next_channels = self.stage_channels[stage_idx + 1]
                layers.append(_conv3x3(current_channels, next_channels, bias=False))
                current_channels = next_channels

        layers.append(_make_activation(activation))
        layers.append(_conv3x3(current_channels, self.out_channels))
        self.net = nn.Sequential(*layers)

        if zero_init_final:
            final = self.net[-1]
            if not isinstance(final, nn.Conv2d):
                raise TypeError("Internal error: final layer must be Conv2d.")
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

        if self.residual:
            num_2x = _num_2x_stages(self.spatial_scale)
            if num_2x == 0:
                self.skip_up = _conv3x3(self.in_channels, self.out_channels)
            else:
                skip_layers: list[nn.Module] = []
                ch = self.in_channels
                for _ in range(num_2x):
                    skip_layers.append(_conv3x3(ch, self.out_channels * 4))
                    skip_layers.append(nn.PixelShuffle(2))
                    ch = self.out_channels
                self.skip_up = nn.Sequential(*skip_layers)
        else:
            self.skip_up = None

    def forward(
        self,
        latent: torch.Tensor,
        parallel: bool | None = None,
        caches: list[object | None] | None = None,
        return_caches: bool = False,
        detach_caches: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[object | None]]:
        if latent.ndim != 5:
            raise ValueError(f"Expected latent shape [B, C, T, H, W], got {tuple(latent.shape)}.")
        if latent.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {latent.shape[1]}.")
        if parallel is None:
            parallel = self.default_parallel

        correction, new_caches = _apply_causal_model(
            self.net,
            latent,
            parallel=bool(parallel),
            caches=caches,
            return_caches=True,
            detach_caches=detach_caches,
        )

        if self.residual:
            b, _, t_in, _, _ = latent.shape
            skip = rearrange(latent, "b c t h w -> (b t) c h w")
            skip = self.skip_up(skip)
            skip = rearrange(skip, "(b t) c h w -> b c t h w", b=b, t=t_in)
            if self.temporal_scale > 1:
                skip = skip.repeat_interleave(self.temporal_scale, dim=2)
            out = skip + self.residual_scale * correction
        else:
            out = correction

        if return_caches:
            return out, new_caches
        return out
