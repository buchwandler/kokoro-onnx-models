from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ExactOnnxISTFT(nn.Module):
    """torch.istft-compatible one-sided iSTFT using ONNX-exportable ops."""

    def __init__(
        self,
        *,
        filter_length: int,
        hop_length: int,
        win_length: int,
        window: str = "hann",
        center: bool = True,
    ) -> None:
        super().__init__()
        self._native_transform = None

        if window != "hann":
            raise ValueError(f"Unsupported window: {window!r}")
        if filter_length <= 0 or hop_length <= 0 or win_length <= 0:
            raise ValueError("STFT sizes must be positive")
        if win_length > filter_length:
            raise ValueError("win_length must not exceed filter_length")

        self.filter_length = int(filter_length)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.n_fft = int(filter_length)
        self.center = bool(center)

        analysis_window = torch.hann_window(
            self.win_length,
            periodic=True,
            dtype=torch.float32,
        )
        if self.win_length < self.n_fft:
            total_padding = self.n_fft - self.win_length
            left = total_padding // 2
            right = total_padding - left
            analysis_window = F.pad(analysis_window, (left, right))

        freq_bins = self.n_fft // 2 + 1
        n = torch.arange(self.n_fft, dtype=torch.float64)
        k = torch.arange(freq_bins, dtype=torch.float64)
        angle = 2.0 * math.pi * k[:, None] * n[None, :] / self.n_fft

        one_sided_scale = torch.ones(freq_bins, dtype=torch.float64)
        if self.n_fft % 2 == 0:
            one_sided_scale[1:-1] = 2.0
        else:
            one_sided_scale[1:] = 2.0

        window64 = analysis_window.to(torch.float64)
        common = one_sided_scale[:, None] * window64[None, :] / self.n_fft
        real_kernel = torch.cos(angle) * common
        imag_kernel = torch.sin(angle) * common
        forward_common = window64[None, :]
        forward_real = torch.cos(angle) * forward_common
        forward_imag = -torch.sin(angle) * forward_common
        self.register_buffer(
            "weight_forward_real",
            forward_real.float().unsqueeze(1),
        )
        self.register_buffer(
            "weight_forward_imag",
            forward_imag.float().unsqueeze(1),
        )

        self.register_buffer(
            "weight_backward_real",
            real_kernel.float().unsqueeze(1),
        )
        self.register_buffer(
            "weight_backward_imag",
            imag_kernel.float().unsqueeze(1),
        )
        self.register_buffer(
            "window_envelope_kernel",
            analysis_window.square().view(1, 1, -1),
        )

    def set_native_transform(self, transform: object) -> None:
        self._native_transform = transform

    def use_onnx_transform(self) -> None:
        self._native_transform = None

    @property
    def onnx_transform_enabled(self) -> bool:
        return self._native_transform is None

    def transform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._native_transform is not None:
            return self._native_transform(waveform)  # type: ignore[operator]
        if waveform.ndim != 2:
            raise ValueError("waveform must be [batch, samples]")
        if self.center:
            pad = self.n_fft // 2
            waveform = F.pad(waveform, (pad, pad), mode="reflect")
        samples = waveform.unsqueeze(1)
        real_part = F.conv1d(samples, self.weight_forward_real, stride=self.hop_length)
        imag_part = F.conv1d(samples, self.weight_forward_imag, stride=self.hop_length)
        if self.n_fft % 2 == 0:
            imag_part = torch.cat(
                [
                    torch.zeros_like(imag_part[:, :1, :]),
                    imag_part[:, 1:-1, :],
                    torch.zeros_like(imag_part[:, -1:, :]),
                ],
                dim=1,
            )
        else:
            imag_part = torch.cat(
                [torch.zeros_like(imag_part[:, :1, :]), imag_part[:, 1:, :]],
                dim=1,
            )
        magnitude = torch.sqrt(real_part.square() + imag_part.square() + 1.0e-14)
        phase = torch.atan2(imag_part, real_part)
        correction_mask = imag_part.abs() < 1.0e-7
        axis_phase = torch.where(
            imag_part < 0,
            torch.full_like(phase, -torch.pi),
            torch.full_like(phase, torch.pi),
        )
        phase = torch.where(correction_mask & (real_part < 0), axis_phase, phase)
        return magnitude, phase

    def inverse(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
        length: int | None = None,
    ) -> torch.Tensor:
        if magnitude.ndim != 3 or phase.ndim != 3:
            raise ValueError("magnitude and phase must be [batch, bins, frames]")

        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        waveform = F.conv_transpose1d(
            real_part,
            self.weight_backward_real,
            stride=self.hop_length,
        )
        waveform = waveform - F.conv_transpose1d(
            imag_part,
            self.weight_backward_imag,
            stride=self.hop_length,
        )

        envelope_source = torch.ones_like(magnitude[:, :1, :])
        envelope = F.conv_transpose1d(
            envelope_source,
            self.window_envelope_kernel,
            stride=self.hop_length,
        )

        if self.center:
            pad = self.n_fft // 2
            waveform = waveform[..., pad:-pad]
            envelope = envelope[..., pad:-pad]

        waveform = waveform / envelope.clamp_min(1.0e-11)

        if length is not None:
            waveform = waveform[..., :length]

        return waveform

    def forward(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        return self.inverse(magnitude, phase)
