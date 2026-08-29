from __future__ import annotations

import numpy as np
import pytest

from scripts.export_validation import (
    spectral_metrics,
    stationary_broadband_noise,
    validate_waveform_health,
)

SAMPLE_RATE = 24_000


def _incident_fixture() -> np.ndarray:
    rng = np.random.default_rng(20260828)
    noise = rng.standard_normal(SAMPLE_RATE * 2)
    noise = noise - np.roll(noise, 1)
    noise = 0.055 * noise / np.std(noise) - 0.015
    return np.clip(noise, -0.32, 0.32).astype(np.float32)


def _speech_fixture() -> np.ndarray:
    t = np.arange(SAMPLE_RATE * 2, dtype=np.float64) / SAMPLE_RATE
    envelope = 0.01 + 0.09 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t))
    phase = 2 * np.pi * (180 * t + 25 * t**2)
    rng = np.random.default_rng(1234)
    audio = envelope * (
        np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    )
    return (audio + 0.003 * rng.normal(size=t.size)).astype(np.float32)


def test_incident_passes_legacy_metrics_but_fails_broadband_detector() -> None:
    audio = _incident_fixture()
    legacy = validate_waveform_health(
        audio,
        SAMPLE_RATE,
        min_audio_rms=0.0005,
        max_audio_abs=1.0,
        max_abs_dc=0.03,
        max_dc_to_rms_ratio=0.7,
        min_frame_rms_cv=0.02,
        max_stationary_tone_ratio=0.7,
        reject_stationary_broadband_noise=False,
    )
    metrics = spectral_metrics(audio, SAMPLE_RATE)
    assert legacy["stationary_tone_ratio"] <= 0.7
    assert stationary_broadband_noise(metrics, seconds=2.0, sample_rate=SAMPLE_RATE)
    with pytest.raises(ValueError, match="stationary broadband noise"):
        validate_waveform_health(
            audio,
            SAMPLE_RATE,
            min_audio_rms=0.0005,
            max_audio_abs=1.0,
            max_abs_dc=0.03,
            max_dc_to_rms_ratio=0.7,
            min_frame_rms_cv=0.02,
            max_stationary_tone_ratio=0.7,
        )


def test_native_speech_fixture_is_not_stationary_broadband_noise() -> None:
    metrics = spectral_metrics(_speech_fixture(), SAMPLE_RATE)
    assert not stationary_broadband_noise(metrics, seconds=2.0, sample_rate=SAMPLE_RATE)


def test_spectral_metrics_expose_required_fields() -> None:
    metrics = spectral_metrics(_incident_fixture(), SAMPLE_RATE)
    assert set(metrics) == {
        "zcr_mean",
        "spectral_centroid_mean_hz",
        "spectral_centroid_cv",
        "spectral_bandwidth_mean_hz",
        "spectral_flatness_mean",
        "normalized_spectral_flux_mean",
        "high_band_energy_ratio_4k_nyquist",
        "frame_rms_cv",
    }
