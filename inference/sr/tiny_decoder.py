from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(inplace=False),
            conv(n_out, n_out),
            nn.ReLU(inplace=False),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _nt, c, h, w = x.shape
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class TGrow(nn.Module):
    def __init__(self, n_f: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _nt, c, h, w = x.shape
        x = self.conv(x)
        return x.reshape(-1, c, h, w)


def apply_model_with_memblocks(
    model: nn.Sequential,
    x: torch.Tensor,
    parallel: bool,
    show_progress_bar: bool,
) -> torch.Tensor:
    assert x.ndim == 5, f"Expected NTCHW tensor, got {x.ndim} dims"
    n, t, c, h, w = x.shape
    if parallel:
        x = x.reshape(n * t, c, h, w)
        for block in tqdm(model, disable=not show_progress_bar):
            if isinstance(block, MemBlock):
                nt, c, h, w = x.shape
                t = nt // n
                x_view = x.reshape(n, t, c, h, w)
                mem = F.pad(x_view, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :t].reshape(x.shape)
                x = block(x, mem)
            else:
                x = block(x)
        nt, c, h, w = x.shape
        t = nt // n
        x = x.view(n, t, c, h, w)
        return x

    out = []
    work_queue = [TWorkItem(xt, 0) for xt in x.reshape(n, t * c, h, w).chunk(t, dim=1)]
    progress_bar = tqdm(range(t), disable=not show_progress_bar)
    mem: list[torch.Tensor | list[torch.Tensor] | None] = [None] * len(model)
    while work_queue:
        xt, i = work_queue.pop(0)
        if i == 0:
            progress_bar.update(1)
        if i == len(model):
            out.append(xt)
            continue
        block = model[i]
        if isinstance(block, MemBlock):
            if mem[i] is None:
                xt_new = block(xt, xt * 0)
                mem[i] = xt
            else:
                assert isinstance(mem[i], torch.Tensor)
                xt_new = block(xt, mem[i])
                mem[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(block, TPool):
            if mem[i] is None:
                mem[i] = []
            assert isinstance(mem[i], list)
            mem[i].append(xt)
            if len(mem[i]) == block.stride:
                n, c, h, w = xt.shape
                xt = block(torch.cat(mem[i], 1).view(n * block.stride, c, h, w))
                mem[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(block, TGrow):
            xt = block(xt)
            nt, c_cur, h_cur, w_cur = xt.shape
            for xt_next in reversed(xt.view(n, block.stride * c_cur, h_cur, w_cur).chunk(block.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = block(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    progress_bar.close()
    return torch.stack(out, 1)


class TAEHV(nn.Module):
    latent_channels = 16
    image_channels = 3

    def __init__(
        self,
        checkpoint_path: str | None = None,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            conv(self.image_channels, 64), nn.ReLU(inplace=False),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=False),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=False),
            conv(n_f[3], self.image_channels),
        )
        if checkpoint_path is not None:
            state_dict = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
            self.load_state_dict(self.patch_tgrow_layers(state_dict))

    def patch_tgrow_layers(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_state = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if state_dict[key].shape[0] > new_state[key].shape[0]:
                    state_dict[key] = state_dict[key][-new_state[key].shape[0]:]
        return state_dict

    def encode_video(self, x: torch.Tensor, parallel: bool = True, show_progress_bar: bool = False) -> torch.Tensor:
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x: torch.Tensor, parallel: bool = True, show_progress_bar: bool = False) -> torch.Tensor:
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        if self.frames_to_trim > 0 and x.shape[1] > self.frames_to_trim:
            x = x[:, self.frames_to_trim:]
        return x


class WanTinyDecoder(nn.Module):
    """
    项目内部 tiny decoder 包装。
    输入/输出统一使用:
    - decode_latents: BCHWT latent -> BCTHW pixel[-1, 1]
    """

    def __init__(self, checkpoint_path: str, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.model = TAEHV(checkpoint_path=checkpoint_path)
        self.runtime_dtype = dtype
        self.parallel_decode = False
        self.model = self.model.to(dtype=self.runtime_dtype)
        self.eval().requires_grad_(False)

    def decode_latents(
        self,
        latents_bcthw: torch.Tensor,
        parallel: bool | None = None,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        latents_btchw = latents_bcthw.permute(0, 2, 1, 3, 4).contiguous()
        latents_btchw = latents_btchw.to(device=device, dtype=self.runtime_dtype)
        if parallel is None:
            parallel = bool(self.parallel_decode)
        video = self.model.decode_video(
            latents_btchw,
            parallel=parallel,
            show_progress_bar=show_progress_bar,
        )
        video = video.to(dtype=torch.float32) * 2.0 - 1.0
        return video.permute(0, 2, 1, 3, 4).contiguous()
