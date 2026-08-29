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


def _audio_values(
    audio: np.ndarray, sample_rate: int, *, allow_empty: bool = True
) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    values = np.asarray(audio, dtype=np.float64).squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"Audio must be one-dimensional after squeeze, got {values.shape}"
        )
    if not allow_empty and values.size == 0:
        raise ValueError("Audio is empty")
    if not np.isfinite(values).all():
        raise ValueError("Audio contains non-finite values")
    return values


def random_source_ops(model: Any) -> list[str]:
    """Return supported stochastic source operators present in an ONNX model."""
    return sorted(
        {node.op_type for node in model.graph.node if node.op_type in RANDOM_SOURCE_OPS}
    )


def frame_rms(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 50.0,
    hop_ms: float = 25.0,
) -> np.ndarray:
    values = _audio_values(audio, sample_rate)
    frame_size = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, round(sample_rate * hop_ms / 1000.0))
    if values.size < frame_size:
        return np.asarray([], dtype=np.float64)
    return np.asarray(
        [
            np.sqrt(np.mean(np.square(values[start : start + frame_size])))
            for start in range(0, values.size - frame_size + 1, hop_size)
        ],
        dtype=np.float64,
    )


def _spectral_frames(
    values: np.ndarray, sample_rate: int, *, frame_ms: float, hop_ms: float
) -> tuple[np.ndarray, int]:
    frame_size = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, round(sample_rate * hop_ms / 1000.0))
    if values.size < frame_size:
        return np.empty((0, frame_size), dtype=np.float64), frame_size
    starts = range(0, values.size - frame_size + 1, hop_size)
    frames = np.asarray([values[start : start + frame_size] for start in starts])
    return frames, frame_size


def spectral_metrics(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 50.0,
    hop_ms: float = 25.0,
    high_band_hz: float = 4000.0,
) -> dict[str, float]:
    values = _audio_values(audio, sample_rate, allow_empty=False)
    frames, frame_size = _spectral_frames(
        values, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms
    )
    if not len(frames):
        return {
            "zcr_mean": 0.0,
            "spectral_centroid_mean_hz": 0.0,
            "spectral_centroid_cv": 0.0,
            "spectral_bandwidth_mean_hz": 0.0,
            "spectral_flatness_mean": 0.0,
            "normalized_spectral_flux_mean": 0.0,
            "high_band_energy_ratio_4k_nyquist": 0.0,
            "frame_rms_cv": 0.0,
        }
    window = np.hanning(frame_size)
    spectra = np.square(np.abs(np.fft.rfft(frames * window, axis=1)))
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    spectra[:, 0] = 0.0
    power_totals = np.maximum(np.sum(spectra, axis=1), np.finfo(np.float64).eps)
    normalized = spectra / power_totals[:, None]
    centroids = np.sum(normalized * frequencies[None, :], axis=1)
    bandwidths = np.sqrt(
        np.sum(normalized * (frequencies[None, :] - centroids[:, None]) ** 2, axis=1)
    )
    flatness = np.exp(
        np.mean(np.log(np.maximum(spectra, np.finfo(np.float64).tiny)), axis=1)
    ) / np.maximum(np.mean(spectra, axis=1), np.finfo(np.float64).eps)
    flux = (
        0.5 * np.linalg.norm(np.diff(normalized, axis=0), axis=1)
        if len(normalized) > 1
        else np.array([])
    )
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    high_band = frequencies >= high_band_hz
    return {
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean_hz": float(np.mean(centroids)),
        "spectral_centroid_cv": float(
            np.std(centroids) / max(float(np.mean(centroids)), np.finfo(np.float64).eps)
        ),
        "spectral_bandwidth_mean_hz": float(np.mean(bandwidths)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "normalized_spectral_flux_mean": float(np.mean(flux)) if flux.size else 0.0,
        "high_band_energy_ratio_4k_nyquist": float(
            np.sum(spectra[:, high_band]) / np.sum(spectra)
        ),
        "frame_rms_cv": float(
            np.std(np.sqrt(np.mean(np.square(frames), axis=1)))
            / max(
                float(np.mean(np.sqrt(np.mean(np.square(frames), axis=1)))),
                np.finfo(np.float64).eps,
            )
        ),
    }


def stationary_broadband_noise(
    metrics: dict[str, float],
    *,
    seconds: float,
    sample_rate: int,
    min_seconds: float = 1.0,
    min_zcr: float = 0.39,
    min_centroid_fraction: float = 0.22,
    max_centroid_cv: float = 0.05,
    min_high_band_ratio: float = 0.60,
    min_flatness: float = 0.05,
    max_frame_rms_cv: float = 0.08,
    max_spectral_flux: float = 0.05,
    min_broadband_votes: int = 3,
) -> bool:
    stationary = (
        metrics["spectral_centroid_cv"] < max_centroid_cv
        and metrics["frame_rms_cv"] < max_frame_rms_cv
        and metrics["normalized_spectral_flux_mean"] < max_spectral_flux
    )
    broadband_votes = (
        metrics["zcr_mean"] >= min_zcr,
        metrics["spectral_centroid_mean_hz"] >= min_centroid_fraction * sample_rate,
        metrics["high_band_energy_ratio_4k_nyquist"] >= min_high_band_ratio,
        metrics["spectral_flatness_mean"] >= min_flatness,
    )
    return (
        seconds >= min_seconds
        and stationary
        and sum(broadband_votes) >= min_broadband_votes
    )


def waveform_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    values = _audio_values(audio, sample_rate, allow_empty=False)
    rms = float(np.sqrt(np.mean(np.square(values))))
    dc = float(np.mean(values))
    frames = frame_rms(values, sample_rate)
    frame_rms_cv = (
        float(np.std(frames) / max(float(np.mean(frames)), np.finfo(np.float64).eps))
        if frames.size
        else 0.0
    )
    metrics: dict[str, Any] = {
        "samples": int(values.size),
        "seconds": float(values.size / sample_rate),
        "peak": float(np.max(np.abs(values))),
        "rms": rms,
        "dc": dc,
        "dc_to_rms_ratio": float(abs(dc) / max(rms, np.finfo(np.float64).eps)),
        "frame_rms_cv": frame_rms_cv,
        "frame_rms": frames.tolist(),
    }
    metrics.update(spectral_metrics(values, sample_rate))
    return metrics


def compare_waveform_structure(
    reference: np.ndarray,
    actual: np.ndarray,
    sample_rate: int,
    *,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    reference_values = _audio_values(reference, sample_rate, allow_empty=False)
    actual_values = _audio_values(actual, sample_rate, allow_empty=False)
    if reference_values.shape != actual_values.shape:
        raise ValueError(
            "Waveforms must have equal shape for structural comparison: "
            f"{reference_values.shape} != {actual_values.shape}"
        )
    reference_frames = frame_rms(reference_values, sample_rate)
    actual_frames = frame_rms(actual_values, sample_rate)
    if reference_frames.size != actual_frames.size or reference_frames.size < 2:
        raise ValueError("Waveforms are too short for frame-RMS comparison")
    reference_log = np.log(np.maximum(reference_frames, eps))
    actual_log = np.log(np.maximum(actual_frames, eps))
    reference_std = float(np.std(reference_log))
    actual_std = float(np.std(actual_log))
    if reference_std == 0.0 or actual_std == 0.0:
        envelope_correlation = 0.0
    else:
        envelope_correlation = float(np.corrcoef(reference_log, actual_log)[0, 1])
    reference_cv = float(
        np.std(reference_frames) / max(float(np.mean(reference_frames)), eps)
    )
    actual_cv = float(np.std(actual_frames) / max(float(np.mean(actual_frames)), eps))
    return {
        "envelope_correlation": envelope_correlation,
        "frame_rms_cv_ratio": float(actual_cv / max(reference_cv, eps)),
        "absolute_dc_delta": float(
            abs(np.mean(actual_values) - np.mean(reference_values))
        ),
        "rms_ratio": float(
            np.sqrt(np.mean(np.square(actual_values)))
            / max(np.sqrt(np.mean(np.square(reference_values))), eps)
        ),
    }


def validate_waveform_health(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_audio_rms: float,
    max_audio_abs: float,
    max_abs_dc: float,
    min_frame_rms_cv: float,
    max_stationary_tone_ratio: float,
    max_dc_to_rms_ratio: float | None = None,
    reject_stationary_broadband_noise: bool = True,
    noise_min_seconds: float = 1.0,
    noise_min_zcr: float = 0.39,
    noise_min_centroid_fraction: float = 0.22,
    noise_max_centroid_cv: float = 0.05,
    noise_min_high_band_ratio: float = 0.60,
    noise_min_flatness: float = 0.05,
    noise_max_frame_rms_cv: float = 0.08,
    noise_max_spectral_flux: float = 0.05,
    error_type: type[ErrorT] = ValueError,
) -> dict[str, float]:
    """Measure and validate basic speech waveform health indicators."""
    if sample_rate <= 0:
        raise error_type(f"Sample rate must be positive, got {sample_rate}")
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
    dc_to_rms_ratio = float(abs(dc) / max(rms, np.finfo(np.float64).eps))

    if peak > max_audio_abs:
        fail(f"Audio peak {peak:.6f} exceeds maximum {max_audio_abs:.6f}")
    if rms < min_audio_rms:
        fail(f"Audio RMS {rms:.6f} is below minimum {min_audio_rms:.6f}")
    if abs(dc) > max_abs_dc:
        fail(f"Audio DC offset {dc:.6f} exceeds maximum {max_abs_dc:.6f}")
    if max_dc_to_rms_ratio is not None and dc_to_rms_ratio > max_dc_to_rms_ratio:
        fail(
            "Audio DC-to-RMS ratio is too high: "
            f"{dc_to_rms_ratio:.6f} > {max_dc_to_rms_ratio:.6f}"
        )

    frames = frame_rms(values, sample_rate)
    if frames.size < 2:
        fail("Audio is too short for frame-RMS health validation")
    frame_rms_cv = float(
        np.std(frames) / max(float(np.mean(frames)), np.finfo(np.float64).eps)
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

    spectral = spectral_metrics(values, sample_rate)
    if reject_stationary_broadband_noise and stationary_broadband_noise(
        spectral,
        seconds=float(values.size / sample_rate),
        sample_rate=sample_rate,
        min_seconds=noise_min_seconds,
        min_zcr=noise_min_zcr,
        min_centroid_fraction=noise_min_centroid_fraction,
        max_centroid_cv=noise_max_centroid_cv,
        min_high_band_ratio=noise_min_high_band_ratio,
        min_flatness=noise_min_flatness,
        max_frame_rms_cv=noise_max_frame_rms_cv,
        max_spectral_flux=noise_max_spectral_flux,
    ):
        fail(
            "Audio resembles stationary broadband noise: "
            f"zcr={spectral['zcr_mean']:.6f}, "
            f"centroid={spectral['spectral_centroid_mean_hz']:.2f}Hz, "
            f"high_band={spectral['high_band_energy_ratio_4k_nyquist']:.6f}"
        )
    return {
        "samples": int(values.size),
        "seconds": float(values.size / sample_rate),
        "peak": peak,
        "rms": rms,
        "dc": dc,
        "dc_to_rms_ratio": dc_to_rms_ratio,
        "frame_rms_cv": frame_rms_cv,
        "stationary_tone_ratio": stationary_tone_ratio,
        **spectral,
    }
