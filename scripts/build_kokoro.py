#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub",
#   "numpy",
#   "onnx",
#   "onnxruntime",
#   "torch",
#   "kokoro==0.9.4",
# ]
# ///
"""Build Kokoro-family ONNX + sherpa-onnx raw voice bundles from HF profiles.

This intentionally separates *acoustic-model compatibility* from *text frontend
compatibility*. A model can be converted successfully while still requiring its
training-time G2P (for example vig2p or phonikud) before token IDs are passed to
ONNX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.export_validation import (
        RANDOM_SOURCE_OPS,
        compare_waveform_structure,
        random_source_ops,
        waveform_metrics,
    )
    from scripts.export_validation import (
        validate_waveform_health as _validate_waveform_health,
    )
except ModuleNotFoundError:
    try:
        from export_validation import (
            RANDOM_SOURCE_OPS,
            compare_waveform_structure,
            random_source_ops,
            waveform_metrics,
        )
        from export_validation import (
            validate_waveform_health as _validate_waveform_health,
        )
    except ModuleNotFoundError:
        import importlib.util

        _validation_spec = importlib.util.spec_from_file_location(
            "_build_kokoro_export_validation",
            Path(__file__).with_name("export_validation.py"),
        )
        if _validation_spec is None or _validation_spec.loader is None:
            raise
        _validation_module = importlib.util.module_from_spec(_validation_spec)
        _validation_spec.loader.exec_module(_validation_module)
        RANDOM_SOURCE_OPS = _validation_module.RANDOM_SOURCE_OPS
        random_source_ops = _validation_module.random_source_ops
        compare_waveform_structure = _validation_module.compare_waveform_structure
        _validate_waveform_health = _validation_module.validate_waveform_health
        waveform_metrics = _validation_module.waveform_metrics
PROFILE_FILE = Path(__file__).with_name("kokoro_profiles.json")
STYLE_WIDTH = 256


class BuildError(RuntimeError):
    pass


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
) -> dict[str, float]:
    return _validate_waveform_health(
        audio,
        sample_rate,
        min_audio_rms=min_audio_rms,
        max_audio_abs=max_audio_abs,
        max_abs_dc=max_abs_dc,
        min_frame_rms_cv=min_frame_rms_cv,
        max_dc_to_rms_ratio=max_dc_to_rms_ratio,
        max_stationary_tone_ratio=max_stationary_tone_ratio,
        error_type=BuildError,
    )


def validate_random_source_graph(
    model: Any, *, requires_random_source_ops: bool
) -> list[str]:
    detected = random_source_ops(model)
    if requires_random_source_ops and not detected:
        raise BuildError(
            "Exported ONNX graph has no stochastic source operator; expected one of "
            + ", ".join(sorted(RANDOM_SOURCE_OPS))
        )
    return detected


def load_profiles(path: Path = PROFILE_FILE) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise BuildError(f"Expected an object in {path}")
    return data


def hf_download(repo_id: str, filename: str, revision: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(cache_dir),
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_hash(path: Path, expected: str | None, *, label: str) -> None:
    if expected is None:
        return
    actual = sha256(path)
    if actual != expected:
        raise BuildError(
            f"SHA-256 mismatch for {label}: expected {expected}, got {actual}"
        )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_voice(value: Any, *, name: str) -> np.ndarray:
    """Normalize a Kokoro style table to [rows, 1, 256] float32."""
    arr = _as_numpy(value)
    if arr.ndim == 2 and arr.shape[1] == STYLE_WIDTH:
        arr = arr[:, None, :]
    if arr.ndim != 3 or arr.shape[1:] != (1, STYLE_WIDTH):
        raise BuildError(
            f"Voice {name!r} has shape {tuple(arr.shape)}; expected "
            f"[rows, {STYLE_WIDTH}] or [rows, 1, {STYLE_WIDTH}]"
        )
    return np.ascontiguousarray(arr, dtype="<f4")


def load_pt_voice(path: Path, *, name: str) -> np.ndarray:
    import torch

    value = torch.load(path, weights_only=True, map_location="cpu")
    return normalize_voice(value, name=name)


def _validate_same_shape(voices: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    if not voices:
        raise BuildError("No voices were resolved")
    shapes = {name: tuple(arr.shape) for name, arr in voices.items()}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise BuildError(f"All speakers must share one style shape; got {shapes}")
    shape = next(iter(unique))
    return int(shape[0]), int(shape[1]), int(shape[2])


def resolve_json_pt_voices(
    repo_id: str, revision: str, cache_dir: Path, index_path: str
) -> dict[str, np.ndarray]:
    local_index = hf_download(repo_id, index_path, revision, cache_dir)
    with local_index.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        raise BuildError(f"Expected an object in {index_path}")

    voices: dict[str, np.ndarray] = {}
    # Preserve publisher order so sid mapping is deterministic and inspectable.
    for name, item in index.items():
        if not isinstance(item, dict) or not item.get("filename"):
            raise BuildError(f"Invalid voice entry {name!r} in {index_path}")
        local = hf_download(repo_id, str(item["filename"]), revision, cache_dir)
        voices[name] = load_pt_voice(local, name=name)
    return voices


def resolve_fixed_pt_voices(
    repo_id: str, revision: str, cache_dir: Path, items: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    voices: dict[str, np.ndarray] = {}
    for name, item in items.items():
        if isinstance(item, str):
            filename = item
            expected_hash = None
        else:
            filename = str(item["path"])
            expected_hash = item.get("sha256")
        local = hf_download(repo_id, filename, revision, cache_dir)
        verify_source_hash(local, expected_hash, label=f"voice {name}")
        voices[name] = load_pt_voice(local, name=name)
    return voices


def _read_raw_voice_file(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    if not names:
        raise BuildError("Raw voice source requires speaker names")
    raw = np.fromfile(path, dtype="<f4")
    per_row = len(names) * STYLE_WIDTH
    if raw.size == 0 or raw.size % per_row:
        raise BuildError(
            f"{path} contains {raw.size} float32 values, which cannot be split "
            f"into {len(names)} speaker(s) with {STYLE_WIDTH} columns"
        )
    rows = raw.size // per_row
    packed = raw.reshape(len(names), rows, 1, STYLE_WIDTH)
    return {name: np.ascontiguousarray(packed[i]) for i, name in enumerate(names)}


def _read_npz_voice_archive(
    path: Path, expected_names: list[str]
) -> dict[str, np.ndarray]:
    voices: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        for name in expected_names:
            if name in archive:
                voices[name] = normalize_voice(archive[name], name=name)
        if not voices and len(expected_names) == 1 and len(keys) == 1:
            name = expected_names[0]
            voices[name] = normalize_voice(archive[keys[0]], name=name)
    missing = [name for name in expected_names if name not in voices]
    if missing:
        raise BuildError(
            f"Voice archive {path} does not contain expected speaker(s): {missing}"
        )
    return voices


def resolve_prepacked_voices(
    repo_id: str,
    revision: str,
    cache_dir: Path,
    filename: str,
    names: list[str],
    *,
    archive_or_raw: bool,
) -> dict[str, np.ndarray]:
    local = hf_download(repo_id, filename, revision, cache_dir)
    if archive_or_raw and zipfile.is_zipfile(local):
        return _read_npz_voice_archive(local, names)
    return _read_raw_voice_file(local, names)


def resolve_voices(profile: dict[str, Any], cache_dir: Path) -> dict[str, np.ndarray]:
    repo_id = profile["repo_id"]
    revision = profile.get("revision", "main")
    spec = profile["voices"]
    kind = spec["kind"]
    if kind == "json_pt":
        return resolve_json_pt_voices(repo_id, revision, cache_dir, spec["index"])
    if kind == "fixed_pt":
        return resolve_fixed_pt_voices(repo_id, revision, cache_dir, spec["items"])
    if kind == "raw":
        return resolve_prepacked_voices(
            repo_id,
            revision,
            cache_dir,
            spec["path"],
            list(spec["names"]),
            archive_or_raw=False,
        )
    if kind == "archive_or_raw":
        return resolve_prepacked_voices(
            repo_id,
            revision,
            cache_dir,
            spec["path"],
            list(spec["names"]),
            archive_or_raw=True,
        )
    raise BuildError(f"Unsupported voices.kind={kind!r}")


def write_sherpa_voices_bin(
    voices: Mapping[str, np.ndarray], out_path: Path
) -> tuple[int, int, int]:
    shape = _validate_same_shape(voices)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for arr in voices.values():
            f.write(np.asarray(arr, dtype="<f4").tobytes(order="C"))

    expected = len(voices) * int(np.prod(shape)) * 4
    actual = out_path.stat().st_size
    if expected != actual:
        raise BuildError(f"Voice pack size mismatch: expected {expected}, got {actual}")
    return shape


def write_numpy_voice_archive(
    voices: Mapping[str, np.ndarray], out_path: Path
) -> tuple[int, int, int]:
    """Write a named NumPy archive for pykokoro consumers."""
    shape = _validate_same_shape(voices)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{name: arr for name, arr in voices.items()})
    if not zipfile.is_zipfile(out_path):
        raise BuildError(f"Voice archive {out_path} is not a NumPy zip archive")
    return shape


def _parity_inputs(
    case: Mapping[str, Any],
    voice: np.ndarray,
    *,
    max_phonemes: int,
) -> tuple[Any, Any, Any]:
    import torch

    phonemes = str(case["phonemes"])[:max_phonemes]
    if not phonemes:
        raise BuildError("Export validation requires non-empty phonemes")
    tokens = torch.tensor(case["tokens"], dtype=torch.long).unsqueeze(0)
    expected_tokens = [0, *[int(token) for token in case["tokens"][1:-1]], 0]
    if tokens.shape[1] < 2 or tokens.squeeze(0).tolist() != expected_tokens:
        raise BuildError(
            f"Invalid frozen token fixture for {case.get('name', 'case')!r}"
        )
    style_index = min(len(phonemes), max_phonemes) - 1
    if style_index >= voice.shape[0]:
        raise BuildError(
            f"Voice pack has {voice.shape[0]} rows; cannot select style row {style_index}"
        )
    style = torch.from_numpy(voice[style_index]).to(dtype=torch.float32)
    speed = torch.tensor([float(case.get("speed", 1.0))], dtype=torch.float32)
    return tokens, style, speed


def _validate_duration_audio_consistency(
    audio: np.ndarray,
    duration: np.ndarray,
    *,
    token_count: int,
    samples_per_frame: int = 600,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    values = np.asarray(duration)
    if values.size != token_count:
        raise BuildError(
            f"Duration count {values.size} does not match token count {token_count}"
        )
    if not np.isfinite(values).all():
        raise BuildError("Duration contains non-finite values")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-4):
        raise BuildError("Duration contains non-integer values")
    integer_values = rounded.astype(np.int64)
    if np.any(integer_values < 1):
        raise BuildError("Duration contains values below one frame")
    audio_values = np.asarray(audio)
    if not np.isfinite(audio_values).all():
        raise BuildError("Audio contains non-finite values")
    total_frames = int(integer_values.sum())
    if total_frames <= 0:
        raise BuildError("Duration has no synthesis frames")
    actual_samples_per_frame = audio_values.size / total_frames
    if not np.isclose(
        actual_samples_per_frame, samples_per_frame, rtol=tolerance, atol=tolerance
    ):
        raise BuildError(
            f"Audio/duration frame ratio {actual_samples_per_frame:.6f} "
            f"does not match {samples_per_frame}"
        )
    return {
        "samples_per_frame": samples_per_frame,
        "duration_frames": total_frames,
        "audio_samples": int(audio_values.size),
        "audio_samples_per_frame": float(actual_samples_per_frame),
    }


def install_exact_onnx_istft(model: Any) -> dict[str, Any]:
    try:
        from scripts.onnx_istft import ExactOnnxISTFT
    except ModuleNotFoundError:
        try:
            from onnx_istft import ExactOnnxISTFT
        except ModuleNotFoundError:
            import importlib.util

            istft_spec = importlib.util.spec_from_file_location(
                "_build_kokoro_onnx_istft",
                Path(__file__).with_name("onnx_istft.py"),
            )
            if istft_spec is None or istft_spec.loader is None:
                raise
            istft_module = importlib.util.module_from_spec(istft_spec)
            istft_spec.loader.exec_module(istft_module)
            ExactOnnxISTFT = istft_module.ExactOnnxISTFT

    generator = model.decoder.generator
    current = generator.stft
    filter_length = int(current.filter_length)
    hop_length = int(current.hop_length)
    win_length = int(current.win_length)

    exact_stft = ExactOnnxISTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
    )
    exact_stft.set_native_transform(current.transform)
    generator.stft = exact_stft
    return {
        "backend": "exact-convtranspose-istft-v1",
        "filter_length": filter_length,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": "hann-periodic",
        "center": True,
        "one_sided_bin_scaling": True,
        "window_envelope_normalization": True,
    }


def _capture_reference_case(
    model: Any,
    case: Mapping[str, Any],
    voice: np.ndarray,
    *,
    max_phonemes: int,
    seed: int,
    name: str,
) -> dict[str, Any]:
    import torch

    tokens, style, speed = _parity_inputs(case, voice, max_phonemes=max_phonemes)
    torch.manual_seed(seed)
    with torch.no_grad():
        audio, duration = model(tokens, style, speed)
    return {
        "name": name,
        "tokens": tokens,
        "style": style,
        "speed": speed,
        "audio": np.asarray(_as_numpy(audio)),
        "duration": np.asarray(_as_numpy(duration)),
    }


def _capture_native_reference_cases(
    model: Any,
    cases: list[Mapping[str, Any]],
    voice: np.ndarray,
    *,
    max_phonemes: int,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        _capture_reference_case(
            model,
            case,
            voice,
            max_phonemes=max_phonemes,
            seed=seed,
            name=str(case.get("name", "case")),
        )
        for case in cases
    ]


def _validate_patched_pytorch_case(
    model: Any,
    native_case: Mapping[str, Any],
    *,
    seed: int,
    sample_rate: int,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    with torch.no_grad():
        patched_audio, patched_duration = model(
            native_case["tokens"], native_case["style"], native_case["speed"]
        )
    patched_audio_np = np.asarray(_as_numpy(patched_audio))
    patched_duration_np = np.asarray(_as_numpy(patched_duration))
    native_audio_np = np.asarray(native_case["audio"])
    native_duration_np = np.asarray(native_case["duration"])
    try:
        np.testing.assert_allclose(
            patched_audio_np, native_audio_np, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_array_equal(patched_duration_np, native_duration_np)
    except AssertionError as exc:
        raise BuildError(
            f"Native/patched PyTorch parity failed for {native_case['name']!r}: {exc}"
        ) from exc

    health_kwargs = _health_kwargs(validation)
    return {
        "name": str(native_case["name"]),
        "native": validate_waveform_health(
            native_audio_np, sample_rate, **health_kwargs
        ),
        "patched": validate_waveform_health(
            patched_audio_np, sample_rate, **health_kwargs
        ),
        "waveform_structure": compare_waveform_structure(
            native_audio_np, patched_audio_np, sample_rate
        ),
        "duration": _validate_duration_audio_consistency(
            patched_audio_np,
            patched_duration_np,
            token_count=int(native_case["tokens"].shape[1]),
        ),
        "max_abs_error": float(
            np.max(np.abs(patched_audio_np - native_audio_np), initial=0.0)
        ),
    }


def _health_kwargs(validation: Mapping[str, Any]) -> dict[str, float]:
    return {
        "min_audio_rms": float(validation.get("min_audio_rms", 0.0005)),
        "max_audio_abs": float(validation.get("max_audio_abs", 1.0)),
        "max_abs_dc": float(validation.get("max_abs_dc", 0.05)),
        "max_dc_to_rms_ratio": float(
            validation.get("max_dc_to_rms_ratio", float("inf"))
        ),
        "min_frame_rms_cv": float(validation.get("min_frame_rms_cv", 0.03)),
        "max_stationary_tone_ratio": float(
            validation.get("max_stationary_tone_ratio", 0.35)
        ),
    }


def _validate_patched_pytorch_against_native(
    model: Any,
    native_cases: list[Mapping[str, Any]],
    *,
    sample_rate: int,
    validation: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    cases = [
        _validate_patched_pytorch_case(
            model,
            native_case,
            seed=seed,
            sample_rate=sample_rate,
            validation=validation,
        )
        for native_case in native_cases
    ]
    return {
        "cases": cases,
        "max_abs_error": max((case["max_abs_error"] for case in cases), default=0.0),
    }


def _validate_onnx_case(
    session: Any,
    native_case: Mapping[str, Any],
    patched_case: Mapping[str, Any],
    *,
    sample_rate: int,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    actual_audio, actual_duration = session.run(
        ["audio", "duration"],
        {
            "tokens": native_case["tokens"].numpy(),
            "style": native_case["style"].numpy(),
            "speed": native_case["speed"].numpy(),
        },
    )
    actual_audio_np = np.asarray(actual_audio)
    actual_duration_np = np.asarray(actual_duration)
    native_duration_np = np.asarray(native_case["duration"])
    try:
        np.testing.assert_array_equal(actual_duration_np, native_duration_np)
    except AssertionError as exc:
        raise BuildError(
            f"Duration parity failed for {native_case['name']!r}: {exc}"
        ) from exc

    health_kwargs = _health_kwargs(validation)
    native_audio_np = np.asarray(native_case["audio"])
    structure = compare_waveform_structure(
        native_audio_np, actual_audio_np, sample_rate
    )
    min_correlation = float(validation.get("min_envelope_correlation", 0.0))
    if structure["envelope_correlation"] < min_correlation:
        raise BuildError(
            f"ONNX envelope correlation {structure['envelope_correlation']:.6f} "
            f"is below {min_correlation:.6f}"
        )
    cv_ratio = structure["frame_rms_cv_ratio"]
    min_cv_ratio = float(validation.get("min_frame_rms_cv_ratio", 0.0))
    max_cv_ratio = float(validation.get("max_frame_rms_cv_ratio", float("inf")))
    if not min_cv_ratio <= cv_ratio <= max_cv_ratio:
        raise BuildError(
            f"ONNX frame RMS CV ratio {cv_ratio:.6f} is outside "
            f"[{min_cv_ratio:.6f}, {max_cv_ratio:.6f}]"
        )
    return {
        "name": str(native_case["name"]),
        "native": waveform_metrics(native_audio_np, sample_rate),
        "patched": patched_case["patched"],
        "onnx": validate_waveform_health(actual_audio_np, sample_rate, **health_kwargs),
        "waveform_structure": structure,
        "timing": _validate_duration_audio_consistency(
            actual_audio_np,
            actual_duration_np,
            token_count=int(native_case["tokens"].shape[1]),
        ),
    }


def _validate_parity_case(
    session: Any,
    model: Any,
    case: Mapping[str, Any],
    voice: np.ndarray,
    *,
    max_phonemes: int,
    sample_rate: int,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    tokens, style, speed = _parity_inputs(case, voice, max_phonemes=max_phonemes)
    with torch.no_grad():
        expected_audio, expected_duration = model(tokens, style, speed)
    actual_audio, actual_duration = session.run(
        ["audio", "duration"],
        {
            "tokens": tokens.numpy(),
            "style": style.numpy(),
            "speed": speed.numpy(),
        },
    )
    expected_audio_np = np.asarray(_as_numpy(expected_audio))
    expected_duration_np = np.asarray(_as_numpy(expected_duration))
    actual_audio_np = np.asarray(actual_audio)
    actual_duration_np = np.asarray(actual_duration)

    for label, values in (
        ("PyTorch audio", expected_audio_np),
        ("ONNX audio", actual_audio_np),
    ):
        if not np.isfinite(values).all():
            raise BuildError(f"{label} contains non-finite values")
    if expected_audio_np.shape != actual_audio_np.shape:
        raise BuildError(
            f"Audio shape mismatch for {case.get('name', 'case')!r}: "
            f"PyTorch {expected_audio_np.shape}, ONNX {actual_audio_np.shape}"
        )
    if expected_duration_np.shape != actual_duration_np.shape:
        raise BuildError(
            f"Duration shape mismatch for {case.get('name', 'case')!r}: "
            f"PyTorch {expected_duration_np.shape}, ONNX {actual_duration_np.shape}"
        )
    try:
        np.testing.assert_array_equal(actual_duration_np, expected_duration_np)
    except AssertionError as exc:
        raise BuildError(
            f"Duration parity failed for {case.get('name', 'case')!r}: {exc}"
        ) from exc

    health_kwargs = {
        "min_audio_rms": float(validation.get("min_audio_rms", 0.0005)),
        "max_audio_abs": float(validation.get("max_audio_abs", 1.0)),
        "max_abs_dc": float(validation.get("max_abs_dc", 0.05)),
        "min_frame_rms_cv": float(validation.get("min_frame_rms_cv", 0.03)),
        "max_stationary_tone_ratio": float(
            validation.get("max_stationary_tone_ratio", 0.35)
        ),
    }
    pytorch_health = validate_waveform_health(
        expected_audio_np, sample_rate, **health_kwargs
    )
    onnx_health = validate_waveform_health(
        actual_audio_np, sample_rate, **health_kwargs
    )

    timing = _validate_duration_audio_consistency(
        actual_audio_np,
        actual_duration_np,
        token_count=int(tokens.shape[1]),
    )
    min_rms_ratio = float(validation.get("min_rms_ratio", 0.10))
    max_rms_ratio = float(validation.get("max_rms_ratio", 10.0))
    rms_ratio = onnx_health["rms"] / max(
        pytorch_health["rms"], np.finfo(np.float64).eps
    )
    if not min_rms_ratio <= rms_ratio <= max_rms_ratio:
        raise BuildError(
            f"ONNX/PyTorch RMS ratio {rms_ratio:.6f} is outside "
            f"[{min_rms_ratio:.6f}, {max_rms_ratio:.6f}]"
        )

    return {
        "name": str(case.get("name", "case")),
        "pytorch": pytorch_health,
        "onnx": onnx_health,
        "rms_ratio": float(rms_ratio),
        "timing": timing,
    }


def export_checkpoint_to_onnx(
    checkpoint: Path,
    config_path: Path,
    out_path: Path,
    *,
    opset: int,
    seq_len: int,
    voice: np.ndarray | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import platform

    import kokoro
    import onnx
    import onnxruntime as ort
    import torch
    from kokoro import KModel
    from kokoro.model import KModelForONNX

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    n_token = int(config.get("n_token", len(config.get("vocab", {})) or 178))
    native_model = (
        KModel(
            repo_id="not-used-local-checkpoint",
            model=str(checkpoint),
            config=config,
            disable_complex=False,
        )
        .to("cpu")
        .eval()
    )
    native_onnx_wrapper = KModelForONNX(native_model).eval()
    validation_config = validation or {}
    cases = list(validation_config.get("cases") or [])
    if cases:
        if voice is None:
            raise BuildError("Export validation requires a voice pack")
        tokens, style, speed = _parity_inputs(cases[0], voice, max_phonemes=510)
    else:
        inner_len = max(8, int(seq_len))
        middle = torch.randint(1, max(2, n_token), (inner_len,), dtype=torch.long)
        tokens = torch.cat(
            [torch.zeros(1, dtype=torch.long), middle, torch.zeros(1, dtype=torch.long)]
        ).unsqueeze(0)
        style = torch.rand(1, STYLE_WIDTH, dtype=torch.float32)
        speed = torch.tensor([1.0], dtype=torch.float32)

    parity_cases = cases
    if not parity_cases and voice is not None:
        probe_length = max(8, int(seq_len))
        parity_cases = [
            {
                "name": "synthetic-probe",
                "phonemes": "a" * probe_length,
                "tokens": [0, *([1] * probe_length), 0],
                "speed": 1.0,
            }
        ]

    export_seed = int(validation_config.get("export_seed", 0))
    native_cases: list[dict[str, Any]] = []
    if parity_cases:
        if voice is None:
            raise BuildError("Export validation requires a voice pack")
        native_cases = _capture_native_reference_cases(
            native_onnx_wrapper,
            parity_cases,
            voice,
            max_phonemes=510,
            seed=export_seed,
        )

    istft_metadata = install_exact_onnx_istft(native_model)
    export_model = KModelForONNX(native_model).eval()
    patched_validation = {"cases": [], "max_abs_error": 0.0}
    if native_cases:
        patched_validation = _validate_patched_pytorch_against_native(
            export_model,
            native_cases,
            sample_rate=int(validation_config.get("sample_rate", 24000)),
            validation=validation_config,
            seed=export_seed,
        )
    native_model.decoder.generator.stft.use_onnx_transform()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (tokens, style, speed),
            str(out_path),
            input_names=["tokens", "style", "speed"],
            output_names=["audio", "duration"],
            dynamic_axes={
                "tokens": {1: "sequence_length"},
                "audio": {0: "audio_length"},
                "duration": {0: "duration_length"},
            },
            opset_version=opset,
            dynamo=False,
        )
    exported_model = onnx.load(str(out_path), load_external_data=True)
    detected_random_ops = validate_random_source_graph(
        exported_model,
        requires_random_source_ops=bool(
            validation_config.get("requires_random_source_ops")
        ),
    )

    result: dict[str, Any] = {
        "kokoro_version": str(getattr(kokoro, "__version__", "unknown")),
        "torch_version": str(torch.__version__),
        "onnx_version": str(onnx.__version__),
        "onnxruntime_version": str(ort.__version__),
        "python_version": platform.python_version(),
        "opset": opset,
        "inputs": ["tokens", "style", "speed"],
        "outputs": ["audio", "duration"],
        "random_source_ops": detected_random_ops,
        "timing": {
            "output": "duration",
            "samples_per_frame": 600,
            "validated": bool(parity_cases),
        },
        "decoder_reconstruction": {
            "reference_backend": "torch.istft",
            **istft_metadata,
            "native_patched_validation": {
                "max_abs_error": patched_validation["max_abs_error"],
                "cases": patched_validation["cases"],
            },
        },
    }
    if native_cases:
        session = ort.InferenceSession(
            str(out_path), providers=["CPUExecutionProvider"]
        )
        validation_cases = [
            _validate_onnx_case(
                session,
                native_case,
                patched_case,
                sample_rate=int(validation_config.get("sample_rate", 24000)),
                validation=validation_config,
            )
            for native_case, patched_case in zip(
                native_cases, patched_validation["cases"]
            )
        ]
        result["validation"] = {"cases": validation_cases}
        result["waveform_validation"] = {"cases": validation_cases}
    return result


def resolve_model(
    profile: dict[str, Any],
    cache_dir: Path,
    out_path: Path,
    *,
    opset: int,
    seq_len: int,
    voice: np.ndarray | None = None,
    export_provenance: dict[str, Any] | None = None,
) -> Path | None:
    repo_id = profile["repo_id"]
    revision = profile.get("revision", "main")
    spec = profile["model"]
    config_local: Path | None = None
    if spec.get("config"):
        config_name = str(spec["config"])
        config_local = hf_download(repo_id, config_name, revision, cache_dir)
        verify_source_hash(
            config_local,
            spec.get("config_sha256"),
            label=f"model config {config_name}",
        )

    if spec["kind"] == "onnx":
        source = hf_download(repo_id, spec["path"], revision, cache_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, out_path)
        return config_local

    if spec["kind"] == "checkpoint":
        if config_local is None:
            raise BuildError("Checkpoint export requires model.config")
        checkpoint = hf_download(repo_id, str(spec["path"]), revision, cache_dir)
        verify_source_hash(
            checkpoint,
            spec.get("sha256"),
            label=f"model checkpoint {spec['path']}",
        )
        metadata = export_checkpoint_to_onnx(
            checkpoint,
            config_local,
            out_path,
            opset=opset,
            seq_len=seq_len,
            voice=voice,
            validation=profile.get("export_validation"),
        )
        if export_provenance is not None and metadata:
            export_provenance.update(metadata)
        return config_local

    raise BuildError(f"Unsupported model.kind={spec['kind']!r}")


def resolve_auxiliary_files(
    profile: dict[str, Any], cache_dir: Path, out_dir: Path
) -> list[dict[str, str]]:
    repo_id = profile["repo_id"]
    revision = profile.get("revision", "main")
    resolved: list[dict[str, str]] = []
    for item in profile.get("auxiliary_files", []):
        source = str(item["path"])
        output = str(item.get("output") or Path(source).name)
        local = hf_download(repo_id, source, revision, cache_dir)
        target = out_dir / output
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        resolved.append(
            {
                "path": output,
                "role": str(item.get("role", "auxiliary")),
                "format": str(item.get("format", "unknown")),
            }
        )
    return resolved


def _onnx_type_name(elem_type: int) -> str:
    import onnx

    names = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.DOUBLE: "float64",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT32: "int32",
    }
    return names.get(elem_type, onnx.TensorProto.DataType.Name(elem_type).lower())


def validate_onnx_contract(
    path: Path, expected: dict[str, Any] | None = None
) -> list[str]:
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    issues: list[str] = []
    actual_inputs = {
        value.name: _onnx_type_name(value.type.tensor_type.elem_type)
        for value in model.graph.input
    }
    expected_inputs = (expected or {}).get("inputs")
    if expected_inputs:
        for name, expected_type in expected_inputs.items():
            actual_type = actual_inputs.get(name)
            if actual_type is None:
                issues.append(f"missing input {name!r}")
            elif actual_type != expected_type:
                issues.append(
                    f"input {name!r} is {actual_type}, expected {expected_type}"
                )
    else:
        if len(model.graph.input) < 3:
            raise BuildError(
                f"{path} exposes only {len(model.graph.input)} inputs; expected 3"
            )
        positional = list(model.graph.input[:3])
        expected_types = ["int64", "float32", "float32"]
        labels = ["token", "style", "speed"]
        for value, label, expected_type in zip(positional, labels, expected_types):
            actual_type = _onnx_type_name(value.type.tensor_type.elem_type)
            if actual_type != expected_type:
                issues.append(
                    f"{label} input {value.name!r} is {actual_type}, "
                    f"expected {expected_type}"
                )

    actual_outputs = {
        value.name: _onnx_type_name(value.type.tensor_type.elem_type)
        for value in model.graph.output
    }
    expected_outputs = (expected or {}).get("outputs")
    if expected_outputs:
        for name, expected_type in expected_outputs.items():
            actual_type = actual_outputs.get(name)
            if actual_type is None:
                issues.append(f"missing output {name!r}")
            elif actual_type != expected_type:
                issues.append(
                    f"output {name!r} is {actual_type}, expected {expected_type}"
                )
    elif not actual_outputs:
        issues.append("model has no outputs")

    timing = (expected or {}).get("timing")
    if timing is not None:
        if not isinstance(timing, dict):
            issues.append("timing metadata must be an object")
        else:
            required_timing = {
                "kind",
                "output",
                "unit",
                "samples_per_frame",
                "includes_boundary_tokens",
            }
            missing_timing = sorted(required_timing - timing.keys())
            if missing_timing:
                issues.append(
                    "timing metadata is missing: " + ", ".join(missing_timing)
                )
            if timing.get("kind") != "token-duration-v1":
                issues.append("timing kind must be token-duration-v1")
            timing_output = timing.get("output")
            if timing_output not in (expected_outputs or {}):
                issues.append(
                    f"timing output {timing_output!r} is not declared in outputs"
                )
            samples_per_frame = timing.get("samples_per_frame")
            if not isinstance(samples_per_frame, int) or samples_per_frame <= 0:
                issues.append("timing samples_per_frame must be positive")
    if "sample_rate" in (expected or {}) and expected["sample_rate"] != 24000:
        issues.append("sample_rate metadata must be 24000 Hz")
    max_tokens = (expected or {}).get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        issues.append("max_tokens metadata must be a positive integer")
    return issues


def add_sherpa_metadata(
    onnx_path: Path,
    profile: dict[str, Any],
    voices: Mapping[str, np.ndarray],
    style_shape: tuple[int, int, int],
) -> None:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    names = list(voices.keys())
    speaker2id = {name: i for i, name in enumerate(names)}
    frontend = profile["frontend"]

    # These profiles are single-language custom fine-tunes. sherpa's version>=2
    # path means "multi-lingual Kokoro frontend", so version 1 is intentional.
    meta = {
        "model_type": "kokoro",
        "language": profile["language"],
        "has_espeak": int(frontend["kind"] in {"espeak", "espeak_plus_preprocess"}),
        "sample_rate": int(profile.get("sample_rate", 24000)),
        "version": 1,
        "voice": profile["language"],
        "style_dim": ",".join(map(str, style_shape)),
        "n_speakers": len(names),
        "id2speaker": ",".join(f"{i}->{name}" for i, name in enumerate(names)),
        "speaker2id": ",".join(f"{name}->{i}" for name, i in speaker2id.items()),
        "speaker_names": ",".join(names),
        "model_url": f"https://huggingface.co/{profile['repo_id']}",
        "source_repo": profile["repo_id"],
        "license": profile["license"],
        "frontend": frontend["name"],
        "sherpa_text_compatible": str(bool(frontend["sherpa_text_compatible"])).lower(),
        "comment": frontend["note"],
    }

    # Preserve source-specific metadata (for example an embedded vocab) and
    # overwrite only the keys this builder owns.
    merged = {prop.key: prop.value for prop in model.metadata_props}
    merged.update({key: str(value) for key, value in meta.items()})
    while model.metadata_props:
        model.metadata_props.pop()
    for key, value in merged.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, str(onnx_path))


def source_artifacts(profile: Mapping[str, Any]) -> dict[str, Any]:
    model = profile["model"]
    model_artifact: dict[str, Any] = {"path": str(model["path"])}
    if model.get("sha256") is not None:
        model_artifact["sha256"] = str(model["sha256"])
    config_name = model.get("config")
    if config_name is not None:
        model_artifact["config"] = str(config_name)
        if model.get("config_sha256") is not None:
            model_artifact["config_sha256"] = str(model["config_sha256"])
    voices: dict[str, Any] = {}
    for name, item in (profile.get("voices") or {}).get("items", {}).items():
        if isinstance(item, str):
            voices[name] = {"path": item}
        else:
            voices[name] = {
                key: str(value)
                for key, value in item.items()
                if key in {"path", "sha256"}
            }
    return {"model": model_artifact, "voices": voices}


def write_bundle_manifest(
    out_dir: Path,
    profile_key: str,
    profile: dict[str, Any],
    voices: Mapping[str, np.ndarray],
    style_shape: tuple[int, int, int],
    contract_issues: list[str],
    config_local: Path | None,
    auxiliary_files: list[dict[str, str]],
    export_provenance: Mapping[str, Any] | None = None,
) -> None:
    manifest = {
        "profile": profile_key,
        "source_repo": profile["repo_id"],
        "revision": profile.get("revision", "main"),
        "source_artifacts": source_artifacts(profile),
        "license": profile["license"],
        "language": profile["language"],
        "sample_rate": profile.get("sample_rate", 24000),
        "speakers": [{"sid": i, "name": name} for i, name in enumerate(voices.keys())],
        "voice_artifacts": {
            "pykokoro": {"path": "voices.npz", "format": "numpy-npz"},
            "sherpa": {"path": "voices.raw.bin", "format": "raw-float32-le"},
        },
        "auxiliary_artifacts": auxiliary_files,
        "style_dim": list(style_shape),
        "frontend": profile["frontend"],
        "onnx_contract": profile.get("onnx_contract", {}),
        "onnx_contract_issues": contract_issues,
        "publication": profile.get("release", {}),
        "runtime_hints": profile.get("runtime_hints", {}),
        "postprocess": profile.get("postprocess", {}),
    }
    if export_provenance:
        manifest["exporter"] = dict(export_provenance)
    with (out_dir / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if config_local is not None:
        shutil.copyfile(config_local, out_dir / "config.json")


def build_profile(
    profile_key: str,
    profile: dict[str, Any],
    out_root: Path,
    *,
    opset: int,
    seq_len: int,
    run_checker: bool,
) -> Path:
    out_dir = out_root / profile_key
    cache_dir = out_dir / ".hf"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{profile_key}] resolving voices", file=sys.stderr)
    voices = resolve_voices(profile, cache_dir)
    style_shape = write_sherpa_voices_bin(voices, out_dir / "voices.raw.bin")
    write_numpy_voice_archive(voices, out_dir / "voices.npz")

    export_provenance: dict[str, Any] = {}

    print(f"[{profile_key}] resolving model", file=sys.stderr)
    config_local = resolve_model(
        profile,
        cache_dir,
        out_dir / "model.onnx",
        opset=opset,
        seq_len=seq_len,
        voice=next(iter(voices.values()), None),
        export_provenance=export_provenance,
    )

    auxiliary_files = resolve_auxiliary_files(profile, cache_dir, out_dir)
    contract = dict(profile.get("onnx_contract", {}))
    contract["sample_rate"] = profile.get("sample_rate", 24000)
    contract_issues = validate_onnx_contract(out_dir / "model.onnx", contract)
    if contract_issues:
        for issue in contract_issues:
            print(f"warning: {issue}", file=sys.stderr)

    add_sherpa_metadata(out_dir / "model.onnx", profile, voices, style_shape)

    if run_checker:
        import onnx

        onnx.checker.check_model(str(out_dir / "model.onnx"))

    write_bundle_manifest(
        out_dir,
        profile_key,
        profile,
        voices,
        style_shape,
        contract_issues,
        config_local,
        auxiliary_files,
        export_provenance,
    )
    shutil.rmtree(cache_dir, ignore_errors=True)
    return out_dir


def list_profiles(profiles: Mapping[str, dict[str, Any]]) -> None:
    print(f"{'PROFILE':16} {'LANG':5} {'MODEL':10} {'TEXT FRONTEND':16} SOURCE")
    for key, profile in profiles.items():
        front = profile["frontend"]
        status = "stock-sherpa" if front["sherpa_text_compatible"] else "external"
        print(
            f"{key:16} {profile['language']:5} {profile['model']['kind']:10} "
            f"{status:16} {profile['repo_id']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Kokoro-family ONNX/sherpa voice bundles from Hugging Face profiles"
    )
    parser.add_argument("--profiles", type=Path, default=PROFILE_FILE)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available model profiles")

    build = sub.add_parser("build", help="Build one profile or all profiles")
    build.add_argument("profile", help="Profile key or 'all'")
    build.add_argument("--out", type=Path, default=Path("build"))
    build.add_argument("--opset", type=int, default=17)
    build.add_argument("--seq-len", type=int, default=64)
    build.add_argument("--skip-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles)
    if args.command == "list":
        list_profiles(profiles)
        return 0

    keys = list(profiles) if args.profile == "all" else [args.profile]
    unknown = [key for key in keys if key not in profiles]
    if unknown:
        raise BuildError(f"Unknown profile(s): {', '.join(unknown)}")

    for key in keys:
        out = build_profile(
            key,
            profiles[key],
            args.out,
            opset=args.opset,
            seq_len=args.seq_len,
            run_checker=not args.skip_check,
        )
        print(out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
