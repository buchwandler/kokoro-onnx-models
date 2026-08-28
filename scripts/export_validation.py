"""Pure validation helpers for checkpoint-based ONNX exports."""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

RANDOM_SOURCE_OPS = frozenset(
    {
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)

ErrorT = TypeVar("ErrorT", bound=Exception)


def random_source_ops(model: Any) -> list[str]:
    """Return supported stochastic source operators present in an ONNX model."""
    return sorted(
        {node.op_type for node in model.graph.node if node.op_type in RANDOM_SOURCE_OPS}
    )


def validate_waveform_health(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_audio_rms: float,
    max_audio_abs: float,
    max_abs_dc: float,
    min_frame_rms_cv: float,
    max_stationary_tone_ratio: float,
    error_type: type[ErrorT] = ValueError,
) -> dict[str, float]:
    """Measure and validate basic speech waveform health indicators."""
    values = np.asarray(audio, dtype=np.float64).squeeze()

    def fail(message: str) -> None:
        raise error_type(message)

    if values.ndim != 1:
        fail(f"Audio must be one-dimensional after squeeze, got {values.shape}")
    if values.size == 0:
        fail("Audio is empty")
    if not np.isfinite(values).all():
        fail("Audio contains non-finite values")

    peak = float(np.max(np.abs(values), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(values))))
    dc = float(np.mean(values))

    if peak > max_audio_abs:
        fail(f"Audio peak {peak:.6f} exceeds maximum {max_audio_abs:.6f}")
    if rms < min_audio_rms:
        fail(f"Audio RMS {rms:.6f} is below minimum {min_audio_rms:.6f}")
    if abs(dc) > max_abs_dc:
        fail(f"Audio DC offset {dc:.6f} exceeds maximum {max_abs_dc:.6f}")

    frame_size = max(1, round(sample_rate * 0.050))
    hop_size = max(1, round(sample_rate * 0.025))
    frame_rms = []
    if values.size >= frame_size:
        for start in range(0, values.size - frame_size + 1, hop_size):
            frame = values[start : start + frame_size]
            frame_rms.append(float(np.sqrt(np.mean(np.square(frame)))))

    if len(frame_rms) < 2:
        fail("Audio is too short for frame-RMS health validation")

    frame_rms_np = np.asarray(frame_rms, dtype=np.float64)
    frame_rms_cv = float(
        np.std(frame_rms_np)
        / max(float(np.mean(frame_rms_np)), np.finfo(np.float64).eps)
    )
    if frame_rms_cv < min_frame_rms_cv:
        fail(
            "Audio has insufficient frame RMS variation: "
            f"{frame_rms_cv:.6f} < {min_frame_rms_cv:.6f}"
        )

    centered = values - dc
    windowed = centered * np.hanning(centered.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    if power.size:
        power[0] = 0.0

    total_power = float(np.sum(power))
    top_count = min(8, max(0, power.size - 1))
    if total_power <= 0.0 or top_count == 0:
        fail("Audio has no analyzable non-DC spectral power")

    strongest = np.partition(power, -top_count)[-top_count:]
    stationary_tone_ratio = float(np.sum(strongest) / total_power)
    if stationary_tone_ratio > max_stationary_tone_ratio:
        fail(
            "Audio stationary-tone ratio is too high: "
            f"{stationary_tone_ratio:.6f} > {max_stationary_tone_ratio:.6f}"
        )

    return {
        "samples": float(values.size),
        "seconds": float(values.size / sample_rate),
        "peak": peak,
        "rms": rms,
        "dc": dc,
        "frame_rms_cv": frame_rms_cv,
        "stationary_tone_ratio": stationary_tone_ratio,
    }
