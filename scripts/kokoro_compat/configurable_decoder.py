"""Configurable Kokoro decoder based on oddadmix's minimal upstream patch.

The pinned upstream patch makes the two decoder widths configurable and changes
``asr_res`` to consume ``dim_in``. All other layers use the installed Kokoro
implementation, whose defaults remain compatible with standard checkpoints.
"""

from __future__ import annotations

from typing import Any

import torch
from kokoro.istftnet import AdainResBlk1d, Generator
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class ConfigurableDecoder(nn.Module):
    """Kokoro decoder with configuration-driven hidden and output widths."""

    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        dim_out: int,
        resblock_kernel_sizes: Any,
        upsample_rates: Any,
        upsample_initial_channel: int,
        resblock_dilation_sizes: Any,
        upsample_kernel_sizes: Any,
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        disable_complex: bool = False,
        hidden_channels: int = 1024,
        out_channels: int = 512,
    ) -> None:
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, hidden_channels, style_dim)
        self.decode = nn.ModuleList(
            [
                AdainResBlk1d(hidden_channels + 2 + 64, hidden_channels, style_dim),
                AdainResBlk1d(hidden_channels + 2 + 64, hidden_channels, style_dim),
                AdainResBlk1d(hidden_channels + 2 + 64, hidden_channels, style_dim),
                AdainResBlk1d(
                    hidden_channels + 2 + 64,
                    out_channels,
                    style_dim,
                    upsample=True,
                ),
            ]
        )
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(dim_in, 64, kernel_size=1)))
        self.generator = Generator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
            disable_complex=disable_complex,
        )

    def forward(self, asr: Any, F0_curve: Any, N: Any, s: Any) -> Any:
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return self.generator(x, s, F0_curve)
