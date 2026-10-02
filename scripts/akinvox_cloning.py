#!/usr/bin/env python3
"""Dedicated build-time exporter for AkinVox Kokoro Cloning v1 ONNX bundles.

AkinVox cloning is a substantially different architecture from the ordinary
single-checkpoint Kokoro exporter, so it lives here rather than in
``build_kokoro.py``. PyTorch, Transformers and the pinned upstream source tree
are build-time-only inputs; the emitted bundle is ONNX graphs, JSON metadata, a
pickle-free NumPy source-parameter archive, provenance and license notices. No
PyTorch checkpoint or LoRA tensor is shipped as a runtime asset.

Toolchain imports stay inside functions so pin, contract, provenance and
metadata helpers remain importable without a build environment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

try:
    from scripts.runtime_contracts import (
        CLONING_LAYOUT,
        REFERENCE_FORMAT,
        validate_reference_constraints,
    )
except ModuleNotFoundError:
    from runtime_contracts import (  # type: ignore[no-redef]
        CLONING_LAYOUT,
        REFERENCE_FORMAT,
        validate_reference_constraints,
    )

MODEL_ID = "en-akinvox-cloning-v1"
MODEL_VERSION = "1.0.1"
RELEASE_VERSION = 1

LAYOUT = CLONING_LAYOUT
VOICE_MODE = "reference"
SAMPLE_RATE = 24000
IDENTITY_SAMPLE_RATE = 16000
MAX_TOKENS = 510
STYLE_DIMENSIONS = {"acoustic": 128, "duration": 128}
REFERENCE_CONSTRAINTS = {
    "format": REFERENCE_FORMAT,
    "sample_rate": SAMPLE_RATE,
    "identity_sample_rate": IDENTITY_SAMPLE_RATE,
    "min_seconds": 3.0,
    "max_seconds": 30.0,
    "memory_width": 192,
    "style_width": 256,
}

# The six exported graphs plus the non-neural source parameter archive.
COMPONENTS = (
    "reference_wavlm",
    "reference_encoders",
    "reference_mapper",
    "prosody",
    "curves",
    "decoder",
)
SOURCE_PARAMETERS = ("weight", "bias", "window")
SOURCE_PARAMS_FILENAME = "source-params.npz"
PARITY_REPORT_FILENAME = "parity-report.json"

EXPECTED_ADAPTER_COUNT = 199
HARMONIC_COUNT = 9
ISTFT_FFT = 20
ISTFT_HOP = 5
UPSAMPLE_SCALE = 300
SINE_AMP = 0.1
NOISE_STD = 0.003
VOICED_THRESHOLD = 10.0

BASE_ROOTS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")
FROZEN_ENCODERS = ("style_encoder", "predictor_encoder")

COMPONENT_CONTRACTS: dict[str, dict[str, Any]] = {
    "reference_wavlm": {
        "inputs": {"input_values": "float32"},
        "outputs": {"wavlm": "float32"},
        "dynamic_axes": {"input_values": {1: "audio_samples"}},
    },
    "reference_encoders": {
        "inputs": {"mel": "float32"},
        "outputs": {"raw_sdec": "float32", "raw_spred": "float32"},
        "dynamic_axes": {"mel": {3: "mel_frames"}},
    },
    "reference_mapper": {
        "inputs": {
            "mel": "float32",
            "mel_lengths": "int64",
            "reference_ids": "int64",
            "reference_lengths": "int64",
            "wavlm": "float32",
            "raw_sdec": "float32",
            "raw_spred": "float32",
        },
        "outputs": {
            "style": "float32",
            "reference_memory": "float32",
            "reference_mask": "bool",
        },
        "dynamic_axes": {
            "mel": {2: "mel_frames"},
            "reference_ids": {1: "reference_tokens"},
            "reference_memory": {1: "memory_rows"},
            "reference_mask": {1: "memory_rows"},
        },
    },
    "prosody": {
        "inputs": {
            "input_ids": "int64",
            "style_dur": "float32",
            "reference_memory": "float32",
            "reference_mask": "bool",
        },
        "outputs": {"pred_dur": "int64", "d": "float32", "t_en": "float32"},
        "dynamic_axes": {
            "input_ids": {1: "text_tokens"},
            "reference_memory": {1: "memory_rows"},
            "reference_mask": {1: "memory_rows"},
            "pred_dur": {0: "text_tokens"},
            "d": {1: "text_tokens"},
            "t_en": {2: "text_tokens"},
        },
    },
    "curves": {
        "inputs": {"en": "float32", "style_dur": "float32"},
        "outputs": {"f0_curve": "float32", "n_curve": "float32"},
        "dynamic_axes": {
            "en": {2: "synthesis_frames"},
            "f0_curve": {1: "curve_frames"},
            "n_curve": {1: "curve_frames"},
        },
    },
    "decoder": {
        "inputs": {
            "asr": "float32",
            "f0_curve": "float32",
            "n_curve": "float32",
            "style_acou": "float32",
            "har": "float32",
        },
        "outputs": {"audio": "float32"},
        "dynamic_axes": {
            "asr": {2: "synthesis_frames"},
            "f0_curve": {1: "curve_frames"},
            "n_curve": {1: "curve_frames"},
            "har": {2: "source_frames"},
            "audio": {1: "audio_samples"},
        },
    },
}

_MEL_FILTERBANK: np.ndarray | None = None
_WRAPPER_TYPES: Any = None


class AkinvoxBuildError(RuntimeError):
    """Raised when the AkinVox cloning build cannot be completed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AkinvoxBuildError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Pins, contracts and metadata (no build toolchain required)
# --------------------------------------------------------------------------


def pinned_sources(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return every upstream identity the build consumes, including hashes."""
    pins = (profile.get("model") or {}).get("pins") or {}
    _require(
        isinstance(pins, Mapping) and pins, "AkinVox profile is missing model.pins"
    )
    for name in ("akinvox", "kokoro_base", "wavlm", "source_code"):
        _require(name in pins, f"AkinVox profile is missing the {name} pin")
    return dict(pins)


def _require_commit(value: Any, label: str) -> str:
    text = str(value)
    _require(
        len(text) == 40
        and text == text.lower()
        and all(c in "0123456789abcdef" for c in text),
        f"{label} must be a lowercase 40-character commit SHA",
    )
    return text


def _require_digest(value: Any, label: str) -> str:
    text = str(value)
    _require(
        len(text) == 64
        and text == text.lower()
        and all(c in "0123456789abcdef" for c in text),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return text


def validate_source_pins(profile: Mapping[str, Any]) -> None:
    """Fail fast unless every upstream input is immutable and checksummed."""
    pins = pinned_sources(profile)
    _require(
        profile.get("source_type") == "huggingface",
        "AkinVox source_type must be huggingface",
    )
    _require_commit(profile.get("revision"), "AkinVox profile revision")

    for name in ("akinvox", "kokoro_base", "wavlm"):
        pin = pins[name]
        _require(isinstance(pin, Mapping), f"{name} pin must be an object")
        _require(bool(pin.get("repo_id")), f"{name} pin is missing repo_id")
        _require_commit(pin.get("revision"), f"{name} revision")

    files = pins["akinvox"].get("files") or {}
    _require(isinstance(files, Mapping) and files, "AkinVox pin lists no files")
    for name in (
        "adapter.pt",
        "reference_mapper.pt",
        "reference_encoders.pt",
        "config.json",
    ):
        _require(name in files, f"AkinVox pin is missing {name}")
    for name, record in files.items():
        _require(
            isinstance(record, Mapping), f"AkinVox file pin {name} must be an object"
        )
        _require_digest(record.get("sha256"), f"AkinVox file pin {name}")

    _require_digest(pins["kokoro_base"].get("sha256"), "Kokoro base checkpoint pin")
    wavlm_files = pins["wavlm"].get("files") or {}
    _require(
        isinstance(wavlm_files, Mapping) and wavlm_files, "WavLM pin lists no files"
    )
    for name in ("pytorch_model.bin", "config.json", "preprocessor_config.json"):
        _require(name in wavlm_files, f"WavLM pin is missing {name}")
    for name, digest in wavlm_files.items():
        _require_digest(digest, f"WavLM file pin {name}")

    source_code = pins["source_code"]
    _require(isinstance(source_code, Mapping), "source_code pin must be an object")
    _require(
        bool(source_code.get("repository")), "source_code pin is missing repository"
    )
    _require_commit(source_code.get("commit"), "source_code commit")
    _require(
        bool(source_code.get("tag")), "source_code pin must record its release tag"
    )


def component_contract() -> dict[str, Any]:
    """Return the machine-readable ONNX contract for the cloning layout."""
    return {
        "inputs": {"reference": "reference-conditioned"},
        "outputs": {"audio": "float32"},
        "max_tokens": MAX_TOKENS,
        "components": {
            name: {
                "inputs": dict(COMPONENT_CONTRACTS[name]["inputs"]),
                "outputs": dict(COMPONENT_CONTRACTS[name]["outputs"]),
            }
            for name in COMPONENTS
        },
    }


def runtime_metadata() -> dict[str, Any]:
    """Return the reference-mode runtime description published to consumers."""
    runtime = {
        "layout": LAYOUT,
        "voice_mode": VOICE_MODE,
        "max_tokens": MAX_TOKENS,
        "speed_supported": False,
        "style_dimensions": dict(STYLE_DIMENSIONS),
        "reference": dict(REFERENCE_CONSTRAINTS),
    }
    validate_reference_constraints(runtime)
    return runtime


def build_filenames() -> dict[str, str]:
    """Return the intermediate filenames written by the exporter."""
    names = {component: f"{component}.onnx" for component in COMPONENTS}
    names["source_params"] = SOURCE_PARAMS_FILENAME
    names["config"] = "config.json"
    return names


def component_filenames(version: str = MODEL_VERSION) -> dict[str, str]:
    """Return the published filename of every released AkinVox asset."""
    names = {
        "reference_wavlm": f"kokoro-akinvox-cloning-reference-wavlm-v{version}.onnx",
        "reference_encoders": f"kokoro-akinvox-cloning-reference-encoders-v{version}.onnx",
        "reference_mapper": f"kokoro-akinvox-cloning-reference-mapper-v{version}.onnx",
        "prosody": f"kokoro-akinvox-cloning-prosody-v{version}.onnx",
        "curves": f"kokoro-akinvox-cloning-curves-v{version}.onnx",
        "decoder": f"kokoro-akinvox-cloning-decoder-v{version}.onnx",
        "source_params": f"kokoro-akinvox-cloning-source-params-v{version}.npz",
        "config": f"kokoro-akinvox-cloning-config-v{version}.json",
    }
    _require(
        set(names) == set(COMPONENTS) | {"source_params", "config"}, "Asset map drifted"
    )
    return names


def source_params_metadata(
    arrays: Mapping[str, np.ndarray], path: Path
) -> dict[str, Any]:
    """Describe an extracted source-parameter archive with shapes and hashes."""
    _require(
        set(arrays) == set(SOURCE_PARAMETERS),
        f"Source parameters must be {list(SOURCE_PARAMETERS)}, got {sorted(arrays)}",
    )
    records: dict[str, Any] = {}
    for name in SOURCE_PARAMETERS:
        values = np.asarray(arrays[name])
        _require(
            bool(np.isfinite(values).all()), f"Source parameter {name} is not finite"
        )
        _require(
            values.dtype == np.float32,
            f"Source parameter {name} must be float32, got {values.dtype}",
        )
        records[name] = {"shape": list(values.shape), "dtype": "float32"}
    weight = np.asarray(arrays["weight"])
    bias = np.asarray(arrays["bias"])
    window = np.asarray(arrays["window"])
    _require(
        weight.ndim == 2 and bias.ndim == 1, "Source weight/bias shapes are invalid"
    )
    _require(
        weight.shape[0] == bias.shape[0],
        "Source weight and bias output counts do not match",
    )
    _require(
        weight.shape[1] == HARMONIC_COUNT, "Source merge layer must consume 9 harmonics"
    )
    _require(
        window.ndim == 1 and window.size == ISTFT_FFT,
        "Source window must hold 20 values",
    )
    return {
        "path": path.name,
        "format": "numpy-npz",
        "sha256": sha256(path),
        "allow_pickle": False,
        "arrays": records,
    }


def array_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """Compare two arrays for the build-time PyTorch/ONNX parity report."""
    reference = np.asarray(expected, dtype=np.float64).reshape(-1)
    candidate = np.asarray(actual, dtype=np.float64).reshape(-1)
    _require(
        reference.shape == candidate.shape,
        f"Parity shapes differ: {reference.shape} != {candidate.shape}",
    )
    _require(bool(reference.size), "Parity comparison requires non-empty arrays")
    _require(
        bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
        "Parity comparison requires finite values",
    )
    delta = np.abs(candidate - reference)
    norm = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    return {
        "shape_exact": list(np.shape(expected)) == list(np.shape(actual)),
        "all_finite": True,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "cosine_similarity": (
            float(np.dot(reference, candidate) / norm) if norm else 1.0
        ),
    }


def write_parity_report(out_dir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    path = out_dir / PARITY_REPORT_FILENAME
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"path": path.name, "sha256": sha256(path)}


# --------------------------------------------------------------------------
# Consumer-side signal processing used by the parity harness
# --------------------------------------------------------------------------


def mel_filterbank() -> np.ndarray:
    """Torchaudio-compatible 80-bin mel projection for 16 kHz reference audio."""
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is not None:
        return _MEL_FILTERBANK
    n_fft = 2048
    frequencies = np.linspace(
        0.0, IDENTITY_SAMPLE_RATE / 2, n_fft // 2 + 1, dtype=np.float32
    )
    mel_max = 2595.0 * np.log10(1.0 + (IDENTITY_SAMPLE_RATE / 2) / 700.0)
    points = np.linspace(0.0, mel_max, 82, dtype=np.float32)
    hz = (700.0 * (10.0 ** (points / 2595.0) - 1.0)).astype(np.float32)
    span = hz[1:] - hz[:-1]
    slopes = hz[:, None] - frequencies[None, :]
    lower = -slopes[:-2] / span[:-1, None]
    upper = slopes[2:] / span[1:, None]
    _MEL_FILTERBANK = np.maximum(0.0, np.minimum(lower, upper)).T.astype(np.float32)
    return _MEL_FILTERBANK


def reference_mels(audio_24k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mapper mel and the padded/even encoder mel used by AkinVox."""
    n_fft, win_length, hop = 2048, 1200, 300
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)
    window = np.pad(window.astype(np.float32), (n_fft // 2 - win_length // 2,) * 2)

    def compute(wave: np.ndarray) -> np.ndarray:
        values = np.asarray(wave, dtype=np.float32)
        centered = np.pad(values, (n_fft // 2, n_fft // 2), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(centered, n_fft)[::hop]
        power = np.square(np.abs(np.fft.rfft(frames * window[None, :], axis=1)))
        mel = power.astype(np.float32) @ mel_filterbank()
        return np.ascontiguousarray(
            (np.log(1.0e-5 + mel.T[None]) + 4.0) / 4.0, dtype=np.float32
        )

    wave = np.asarray(audio_24k, dtype=np.float32)
    mapper_mel = compute(wave)
    encoder_mel = compute(np.pad(wave, (5000, 5000)))
    even = encoder_mel.shape[-1] - encoder_mel.shape[-1] % 2
    return mapper_mel, np.ascontiguousarray(encoder_mel[:, :, :even][:, None])


def harmonic_source(f0_curve: np.ndarray, seed: int, source_params: Path) -> np.ndarray:
    """Reproduce the AkinVox/Kokoro seeded harmonic source without PyTorch."""
    with np.load(source_params, allow_pickle=False) as archive:
        weight = np.asarray(archive["weight"], dtype=np.float64)
        bias = np.asarray(archive["bias"], dtype=np.float64)
        window = np.asarray(archive["window"], dtype=np.float32)
    values = np.asarray(f0_curve, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    rng = np.random.default_rng(seed)
    f0 = np.repeat(values, UPSAMPLE_SCALE, axis=1)[..., None]
    rad = f0 * np.arange(1, HARMONIC_COUNT + 1) / SAMPLE_RATE
    rad -= np.floor(rad)
    initial = rng.random((f0.shape[0], HARMONIC_COUNT))
    initial[:, 0] = 0.0
    rad[:, 0, :] += initial
    phase = np.cumsum(_resample(rad, 1 / UPSAMPLE_SCALE), axis=1) * 2 * np.pi
    phase = _resample(phase * UPSAMPLE_SCALE, UPSAMPLE_SCALE)
    voiced = (f0 > VOICED_THRESHOLD).astype(np.float64)
    amplitude = voiced * NOISE_STD + (1 - voiced) * SINE_AMP / 3
    noise = rng.standard_normal((f0.shape[0], f0.shape[1], HARMONIC_COUNT))
    waves = np.sin(phase) * SINE_AMP * voiced + amplitude * noise
    merged = np.tanh(waves @ weight.T + bias)
    return _stft(merged[0, :, 0].astype(np.float32), window)


def _resample(values: np.ndarray, scale: float) -> np.ndarray:
    length = values.shape[1]
    output_length = int(length * scale)
    source = (np.arange(output_length, dtype=np.float64) + 0.5) / scale - 0.5
    np.clip(source, 0, length - 1, out=source)
    lower = np.floor(source).astype(np.int64)
    upper = np.minimum(lower + 1, length - 1)
    weight = (source - lower)[None, :, None]
    return values[:, lower] * (1 - weight) + values[:, upper] * weight


def _stft(audio: np.ndarray, window: np.ndarray) -> np.ndarray:
    centered = np.pad(audio, (ISTFT_FFT // 2,) * 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(centered, ISTFT_FFT)[::ISTFT_HOP]
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    magnitude = np.abs(spectrum).T.astype(np.float32)
    phase = np.angle(spectrum).T.astype(np.float32)
    return np.ascontiguousarray(np.concatenate([magnitude, phase], axis=0)[None])


def expand_frames(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Repeat per-token channels over their predicted duration frames."""
    index = np.repeat(np.arange(durations.size, dtype=np.int64), durations)
    return np.ascontiguousarray(values[:, :, index])


# --------------------------------------------------------------------------
# Build-time model reconstruction
# --------------------------------------------------------------------------


def _load_upstream(source_dir: Path) -> SimpleNamespace:
    """Import the pinned AkinVox source tree without installing its package."""
    package_root = source_dir / "kokoro_cloning"
    _require(
        package_root.is_dir(), f"Pinned source tree has no kokoro_cloning: {source_dir}"
    )
    if "kokoro_cloning" not in sys.modules:
        package = ModuleType("kokoro_cloning")
        package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        package.__package__ = "kokoro_cloning"
        sys.modules["kokoro_cloning"] = package

    def module(name: str) -> Any:
        return importlib.import_module(f"kokoro_cloning.{name}")

    return SimpleNamespace(
        en_models=module("en_models"),
        adapters=module("adapters"),
        reference_pitch_text=module("reference_pitch_text"),
        reference_sequence=module("reference_sequence"),
        reference_mel=module("reference_mel"),
        istftnet=module("Modules.istftnet"),
    )


def resolve_source_code(profile: Mapping[str, Any], cache_dir: Path) -> Path:
    """Fetch the pinned AkinVox source commit and verify the checkout identity."""
    import tarfile
    import urllib.request

    pin = pinned_sources(profile)["source_code"]
    commit = _require_commit(pin["commit"], "source_code commit")
    checkout = cache_dir / "source" / commit
    if checkout.is_dir():
        return checkout
    checkout.parent.mkdir(parents=True, exist_ok=True)
    archive = checkout.with_suffix(".tar.gz")
    if not archive.is_file():
        url = f"https://codeload.github.com/{pin['repository']}/tar.gz/{commit}"
        with (
            urllib.request.urlopen(url, timeout=180) as response,
            archive.open("wb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    if pin.get("archive_sha256"):
        _require(
            sha256(archive) == pin["archive_sha256"],
            "AkinVox source archive SHA-256 does not match its pin",
        )
    staging = checkout.parent / f"{commit}.unpack"
    with tarfile.open(archive) as bundle:
        bundle.extractall(staging, filter="data")
    roots = [entry for entry in staging.iterdir() if entry.is_dir()]
    _require(len(roots) == 1, "AkinVox source archive must contain exactly one root")
    roots[0].replace(checkout)
    staging.rmdir()
    return checkout


def _download(
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: Path,
    namespace: str,
    expected: str,
    label: str,
) -> Path:
    from huggingface_hub import hf_hub_download

    local = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(cache_dir / "downloads" / namespace),
        )
    )
    actual = sha256(local)
    _require(
        actual == expected,
        f"{label} SHA-256 mismatch: expected {expected}, got {actual}",
    )
    return local


def resolve_sources(profile: Mapping[str, Any], cache_dir: Path) -> dict[str, Path]:
    """Download and checksum every model input named by the profile pins."""
    pins = pinned_sources(profile)
    resolved: dict[str, Path] = {"source_code": resolve_source_code(profile, cache_dir)}

    for namespace in ("akinvox", "kokoro_base", "wavlm"):
        pin = pins[namespace]
        records: Mapping[str, Any]
        if namespace == "akinvox":
            records = pin["files"]
        elif namespace == "kokoro_base":
            records = {
                str(pin.get("path", "kokoro-v1_0.pth")): {"sha256": pin["sha256"]}
            }
        else:
            records = pin["files"]
        for name, record in records.items():
            digest = record["sha256"] if isinstance(record, Mapping) else record
            path = (
                str(record.get("path", name)) if isinstance(record, Mapping) else name
            )
            resolved[f"{namespace}:{name}"] = _download(
                str(pin["repo_id"]),
                path,
                str(pin["revision"]),
                cache_dir,
                namespace,
                str(digest),
                f"{namespace} {name}",
            )
    return resolved


def _base_state_candidates(raw_state: Mapping[str, Any]) -> dict[str, Any]:
    """Translate legacy weight_norm keys to parametrized-weight keys."""
    translated = dict(raw_state)
    for key, value in raw_state.items():
        if key.endswith(".weight_g"):
            translated[key[:-9] + ".parametrizations.weight.original0"] = value
            translated.pop(key, None)
        elif key.endswith(".weight_v"):
            translated[key[:-9] + ".parametrizations.weight.original1"] = value
            translated.pop(key, None)
    return translated


def _load_exactly(module: Any, state: Mapping[str, Any], label: str) -> int:
    missing, unexpected = module.load_state_dict(state, strict=True)
    _require(
        not missing and not unexpected,
        f"{label} did not load exactly: missing={missing} unexpected={unexpected}",
    )
    return len(state)


def build_native_model(
    upstream: SimpleNamespace, config: Mapping[str, Any], sources: Mapping[str, Path]
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Reconstruct the exact AkinVox inference model and audit every tensor."""
    import torch
    from munch import munchify
    from transformers import AlbertConfig, AlbertModel

    class CustomAlbert(AlbertModel):
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return super().forward(*args, **kwargs).last_hidden_state

    architecture = config["architecture"]
    bert = CustomAlbert(
        AlbertConfig(vocab_size=architecture["n_token"], **architecture["plbert"])
    )
    core = dict(upstream.en_models.build_model(munchify(architecture), bert))
    _require(
        set(core) == set(BASE_ROOTS) | set(FROZEN_ENCODERS),
        "AkinVox architecture roots changed",
    )
    audit: dict[str, Any] = {
        "roots": {},
        "frozen_encoders": {},
        "mapper": {},
        "adapters": {},
        "strict": True,
    }

    state = torch.load(
        sources["kokoro_base:kokoro-v1_0.pth"], map_location="cpu", weights_only=True
    )
    _require(set(state) == set(BASE_ROOTS), "Kokoro base checkpoint roots changed")
    for root in BASE_ROOTS:
        candidate = _base_state_candidates(state[root])
        missing, unexpected = core[root].load_state_dict(candidate, strict=False)
        _require(
            not missing and not unexpected,
            f"Kokoro base root {root!r} did not load exactly: "
            f"missing={missing} unexpected={unexpected}",
        )
        audit["roots"][root] = {
            "tensors": len(candidate),
            "missing_keys": [],
            "unexpected_keys": [],
        }
    del state

    observations = torch.load(
        sources["akinvox:reference_encoders.pt"], map_location="cpu", weights_only=True
    )
    _require(
        set(observations) == set(FROZEN_ENCODERS),
        "Reference encoder checkpoint roots changed",
    )
    for name in FROZEN_ENCODERS:
        audit["frozen_encoders"][name] = {
            "tensors": _load_exactly(
                core[name], observations[name], f"frozen encoder {name}"
            ),
            "missing_keys": [],
            "unexpected_keys": [],
        }

    delta = torch.load(
        sources["akinvox:adapter.pt"], map_location="cpu", weights_only=True
    )
    adapters = upstream.adapters.attach(core, config["targets"], delta)
    _require(
        len(adapters) == EXPECTED_ADAPTER_COUNT, "AkinVox adapter inventory changed"
    )
    audit["adapters"] = {
        "count": len(adapters),
        "targets": len(config["targets"]),
        "consumed_keys": len(delta),
    }

    mapper_state = torch.load(
        sources["akinvox:reference_mapper.pt"], map_location="cpu", weights_only=True
    )
    mapper = upstream.reference_sequence.SequenceReferenceMapper(
        mapper_state["pooled.base_table"].clone()
    )
    upstream.reference_pitch_text.attach_reference_text(mapper)
    audit["mapper"] = {
        "tensors": _load_exactly(mapper, mapper_state, "reference mapper"),
        "missing_keys": [],
        "unexpected_keys": [],
    }

    for module in [*core.values(), mapper]:
        module.eval().requires_grad_(False)
    return core, mapper, audit


def materialize_lora(
    core: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Permanently fold every LoRA update into its target parameter."""
    import torch
    from torch.nn.utils import parametrize

    def resolve(target: Mapping[str, Any]) -> tuple[Any, str]:
        module = core[target["root"]]
        if target["module"]:
            module = module.get_submodule(target["module"])
        return module, str(target["parameter"])

    adapted = {
        target["prefix"]: getattr(*resolve(target)).detach().clone()
        for target in config["targets"]
    }

    for target in config["targets"]:
        module, name = resolve(target)
        parametrize.remove_parametrizations(module, name, leave_parametrized=True)

    for target in config["targets"]:
        module, name = resolve(target)
        registry = getattr(module, "parametrizations", None)
        _require(
            registry is None or name not in registry,
            f"Target {target['prefix']} is still parametrized",
        )
        torch.testing.assert_close(
            getattr(module, name).detach(),
            adapted[target["prefix"]],
            rtol=0.0,
            atol=0.0,
            msg=f"Materialized weights differ for {target['prefix']}",
        )
        _require(
            not any(type(value).__name__ == "Adapter" for value in module.modules()),
            f"Adapter module survived materialization for {target['prefix']}",
        )
    return {
        "formula": "W_final = W_base + reshape(B @ A) * alpha / rank",
        "leave_parametrized": True,
        "materialized_targets": len(config["targets"]),
        "residual_parametrizations": 0,
        "adapter_modules_on_export_path": 0,
        "weight_parity": "exact",
    }


def extract_source_params(decoder: Any, out_path: Path) -> dict[str, Any]:
    """Export the non-neural harmonic source parameters for the ONNX runtime."""
    source = decoder.generator.m_source.l_linear
    arrays = {
        "weight": source.weight.detach().cpu().float().numpy(),
        "bias": source.bias.detach().cpu().float().numpy(),
        "window": decoder.generator.stft.window.detach().cpu().float().numpy(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    with np.load(out_path, allow_pickle=False) as archive:
        _require(
            sorted(archive.files) == sorted(SOURCE_PARAMETERS),
            "Source parameter archive is not reloadable without pickle",
        )
        for name in SOURCE_PARAMETERS:
            np.testing.assert_array_equal(archive[name], arrays[name])
    return source_params_metadata(arrays, out_path)


# --------------------------------------------------------------------------
# ONNX export wrappers
# --------------------------------------------------------------------------


def _positions(count: int, width: int, device: Any, dtype: Any) -> Any:
    import math

    import torch

    position = torch.arange(count, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / width)
    )
    return torch.stack(
        (torch.sin(position * frequency), torch.cos(position * frequency)), -1
    ).flatten(-2)


def _safe_wave(wave: Any) -> Any:
    import torch

    magnitude = wave.abs()
    compressed = 0.95 + 0.03 * torch.tanh((magnitude - 0.95) / 0.03)
    return torch.where(magnitude <= 0.95, wave, wave.sign() * compressed).clamp(
        -0.98, 0.98
    )


def _onnx_text_encoder(text_encoder: Any, input_ids: Any) -> Any:
    """ONNX-safe ``TextEncoder`` path for the un-padded batch-1 export case."""
    value = text_encoder.embedding(input_ids).transpose(1, 2)
    for block in text_encoder.cnn:
        value = block(value)
    value, _ = text_encoder.lstm(value.transpose(1, 2))
    return value.transpose(-1, -2)


def _onnx_duration_encoder(duration_encoder: Any, value: Any, style: Any) -> Any:
    """ONNX-safe ``DurationEncoder`` path for the un-padded batch-1 export case."""
    import torch

    rows = value.shape[1]
    style_rows = style[:, None, :].expand(-1, rows, -1)
    hidden = torch.cat([value.transpose(1, 2), style_rows], dim=-1).transpose(1, 2)
    for block in duration_encoder.lstms:
        if type(block).__name__ == "AdaLayerNorm":
            hidden = block(hidden.transpose(1, 2), style).transpose(1, 2)
            hidden = torch.cat([hidden, style_rows.transpose(1, 2)], dim=1)
        else:
            output, _ = block(hidden.transpose(1, 2))
            hidden = output.transpose(1, 2)
    return hidden.transpose(1, 2)


def wrapper_types() -> Any:
    """Define the torch export wrappers once the build toolchain is present."""
    global _WRAPPER_TYPES
    if _WRAPPER_TYPES is not None:
        return _WRAPPER_TYPES
    import torch
    from torch import nn

    class ReferenceWavLM(nn.Module):
        """Normalize 16 kHz audio and return the normalized x-vector embedding."""

        def __init__(self, identity: Any) -> None:
            super().__init__()
            self.identity = identity

        def forward(self, input_values: Any) -> Any:
            mean = input_values.mean()
            variance = ((input_values - mean) ** 2).mean()
            normalized = (input_values - mean) / torch.sqrt(variance + 1.0e-7)
            embedding = self.identity(input_values=normalized).embeddings.float()
            return torch.nn.functional.normalize(embedding, dim=-1, eps=1.0e-8)

    class ReferenceEncoders(nn.Module):
        """Export the frozen AkinVox observation encoders."""

        def __init__(self, core: Mapping[str, Any]) -> None:
            super().__init__()
            self.style_encoder = core["style_encoder"]
            self.predictor_encoder = core["predictor_encoder"]

        def forward(self, mel: Any) -> tuple[Any, Any]:
            return self.style_encoder(mel).float(), self.predictor_encoder(mel).float()

    class ReferenceMapper(nn.Module):
        """Reproduce AkinVox enrollment and expose reusable reference memory."""

        def __init__(self, mapper: Any) -> None:
            super().__init__()
            self.shared = mapper

        def forward(
            self,
            mel: Any,
            mel_lengths: Any,
            reference_ids: Any,
            reference_lengths: Any,
            wavlm: Any,
            raw_sdec: Any,
            raw_spred: Any,
        ) -> tuple[Any, Any, Any]:
            shared = self.shared
            columns = reference_ids.shape[1]
            phone_mask = torch.arange(columns, device=reference_ids.device)[None, :]
            phone_mask = phone_mask >= reference_lengths[:, None]

            frames = mel.shape[-1]
            mask = shared.mask(mel_lengths, frames)
            value = torch.nn.functional.gelu(
                shared.conv1(mel.masked_fill(mask[:, None], 0))
            )
            lengths = (mel_lengths + 1) // 2
            value = value.masked_fill(shared.mask(lengths, value.shape[-1])[:, None], 0)
            value = torch.nn.functional.gelu(shared.conv2(value))
            lengths = (lengths + 1) // 2
            mask = shared.mask(lengths, value.shape[-1])
            value = shared.reference_transcript(
                shared.norm(value.transpose(1, 2)), mask, reference_ids, phone_mask
            )
            value = shared.sequence(
                value
                + _positions(value.shape[1], 192, value.device, value.dtype)[None],
                src_key_padding_mask=mask,
            )
            value = value.masked_fill(mask[..., None], 0)
            denominator = lengths[:, None].to(value.dtype)
            mean = value.sum(1) / denominator
            variance = (value - mean[:, None]).square().masked_fill(mask[..., None], 0)
            stats = torch.cat(
                (mean, (variance / denominator).clamp_min(1.0e-8).sqrt()), -1
            )

            row_indices = (reference_lengths - 1).to(torch.int64)
            coarse = shared.pooled(wavlm, raw_sdec, raw_spred, row_indices)
            return coarse + shared.global_style(stats), value, mask

    class Prosody(nn.Module):
        """Adapted BERT, reference attention and duration prediction."""

        def __init__(self, core: Mapping[str, Any], mapper: Any) -> None:
            super().__init__()
            self.bert = core["bert"]
            self.bert_encoder = core["bert_encoder"]
            self.predictor = core["predictor"]
            self.text_encoder = core["text_encoder"]
            self.prosody_attention = mapper.prosody
            self.content_attention = mapper.content

        def forward(
            self,
            input_ids: Any,
            style_dur: Any,
            reference_memory: Any,
            reference_mask: Any,
        ) -> tuple[Any, Any, Any]:
            text_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            prompt = (reference_memory, reference_mask, None)
            hidden = self.bert(input_ids, attention_mask=(~text_mask).int())
            duration_embedding = self.bert_encoder(hidden).transpose(-1, -2)
            duration_embedding = self.prosody_attention(
                duration_embedding, text_mask, prompt
            )
            duration_context = _onnx_duration_encoder(
                self.predictor.text_encoder, duration_embedding, style_dur
            )
            duration_hidden, _ = self.predictor.lstm(duration_context)
            duration_logits = self.predictor.duration_proj(duration_hidden)
            pred_dur = torch.sigmoid(duration_logits).sum(dim=-1).squeeze(0)
            pred_dur = torch.round(pred_dur).clamp(min=1).to(torch.int64)
            content = self.content_attention(
                _onnx_text_encoder(self.text_encoder, input_ids), text_mask, prompt
            )
            return pred_dur, duration_context, content

    class Curves(nn.Module):
        """Duration-conditioned F0 and noise curves."""

        def __init__(self, core: Mapping[str, Any]) -> None:
            super().__init__()
            self.predictor = core["predictor"]

        def forward(self, en: Any, style_dur: Any) -> tuple[Any, Any]:
            hidden, _ = self.predictor.shared(en.transpose(-1, -2))
            f0 = hidden.transpose(-1, -2)
            for block in self.predictor.F0:
                f0 = block(f0, style_dur)
            f0 = self.predictor.F0_proj(f0).squeeze(1)
            noise = hidden.transpose(-1, -2)
            for block in self.predictor.N:
                noise = block(noise, style_dur)
            return f0, self.predictor.N_proj(noise).squeeze(1)

    class Decoder(nn.Module):
        """ONNX-safe decoder consuming a precomputed harmonic source spectrum."""

        def __init__(self, core: Mapping[str, Any], slope: float) -> None:
            super().__init__()
            self.decoder = core["decoder"]
            self.generator = core["decoder"].generator
            self.slope = slope

        def forward(
            self,
            asr: Any,
            f0_curve: Any,
            n_curve: Any,
            style_acou: Any,
            har: Any,
        ) -> Any:
            decoder, generator = self.decoder, self.generator
            f0 = decoder.F0_conv(f0_curve[:, None, :])
            noise = decoder.N_conv(n_curve[:, None, :])
            value = torch.cat([asr, f0, noise], dim=1)
            value = decoder.encode(value, style_acou)
            residual = decoder.asr_res(asr)
            reuse = True
            for block in decoder.decode:
                if reuse:
                    value = torch.cat([value, residual, f0, noise], dim=1)
                value = block(value, style_acou)
                if block.upsample_type != "none":
                    reuse = False

            for index in range(generator.num_upsamples):
                value = torch.nn.functional.leaky_relu(value, self.slope)
                source = generator.noise_res[index](
                    generator.noise_convs[index](har), style_acou
                )
                value = generator.ups[index](value)
                if index == generator.num_upsamples - 1:
                    value = generator.reflection_pad(value)
                value = value + source
                combined = None
                for offset in range(generator.num_kernels):
                    block = generator.resblocks[index * generator.num_kernels + offset]
                    contribution = block(value, style_acou)
                    combined = (
                        contribution if combined is None else combined + contribution
                    )
                value = combined / generator.num_kernels
            value = generator.conv_post(torch.nn.functional.leaky_relu(value))
            bins = generator.post_n_fft // 2 + 1
            magnitude = torch.exp(value[:, :bins, :])
            phase = torch.sin(value[:, bins:, :])
            return _safe_wave(generator.stft.inverse(magnitude, phase))

    _WRAPPER_TYPES = SimpleNamespace(
        ReferenceWavLM=ReferenceWavLM,
        ReferenceEncoders=ReferenceEncoders,
        ReferenceMapper=ReferenceMapper,
        Prosody=Prosody,
        Curves=Curves,
        Decoder=Decoder,
    )
    return _WRAPPER_TYPES


def install_onnx_istft(generator: Any) -> dict[str, Any]:
    """Replace the decoder iSTFT with the repository's ONNX-safe equivalent."""
    try:
        from scripts.onnx_istft import ExactOnnxISTFT
    except ModuleNotFoundError:
        from onnx_istft import ExactOnnxISTFT  # type: ignore[no-redef]

    native = generator.stft
    exact = ExactOnnxISTFT(
        filter_length=int(native.filter_length),
        hop_length=int(native.hop_length),
        win_length=int(native.win_length),
        center=True,
    )
    exact.set_native_transform(native.transform)
    generator.stft = exact
    exact.use_onnx_transform()
    _require(exact.onnx_transform_enabled, "Export iSTFT did not enter ONNX-safe mode")
    return {
        "backend": "exact-convtranspose-istft-v1",
        "reference_backend": "torch.istft",
        "filter_length": int(native.filter_length),
        "hop_length": int(native.hop_length),
        "win_length": int(native.win_length),
        "window": "hann-periodic",
        "center": True,
        "one_sided_bin_scaling": True,
        "window_envelope_normalization": True,
    }


def export_component(
    wrapper: Any,
    args: Sequence[Any],
    out_path: Path,
    *,
    component: str,
    opset: int,
) -> dict[str, Any]:
    """Export one component graph and verify its declared tensor contract."""
    import onnx
    import torch

    contract = COMPONENT_CONTRACTS[component]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_names = list(contract["inputs"])
    output_names = list(contract["outputs"])
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(args),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dict(contract["dynamic_axes"]),
            opset_version=opset,
            dynamo=False,
        )
    model = onnx.load(str(out_path), load_external_data=True)
    onnx.checker.check_model(model)
    _require(
        {value.name for value in model.graph.input} == set(input_names),
        f"{component} inputs differ from contract",
    )
    _require(
        {value.name for value in model.graph.output} == set(output_names),
        f"{component} outputs differ from contract",
    )
    return {
        "component": component,
        "path": out_path.name,
        "opset": opset,
        "sha256": sha256(out_path),
        "size": out_path.stat().st_size,
        "inputs": dict(contract["inputs"]),
        "outputs": dict(contract["outputs"]),
        "dynamic_axes": sorted(
            {
                axis
                for axes in contract["dynamic_axes"].values()
                for axis in axes.values()
            }
        ),
    }


# --------------------------------------------------------------------------
# PyTorch versus ONNX parity
# --------------------------------------------------------------------------


def synthetic_reference(seconds: float, seed: int) -> np.ndarray:
    """Generate a repository-owned synthetic reference; no private recordings."""
    rng = np.random.default_rng(seed)
    count = round(seconds * SAMPLE_RATE)
    time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    envelope = 0.25 + 0.2 * np.sin(2.0 * np.pi * 1.7 * time)
    wave = envelope * np.sin(2.0 * np.pi * 155.0 * time)
    wave += 0.02 * rng.standard_normal(count)
    return np.asarray(wave, dtype=np.float32)


def synthetic_tokens(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 178, size=count).astype(np.int64)


def parity_cases(seed: int = 20260926) -> list[dict[str, Any]]:
    """Build the documented multi-duration, multi-sentence parity matrix."""
    return [
        {
            "name": f"reference-{int(seconds)}s",
            "reference_seconds": seconds,
            "reference_audio": synthetic_reference(seconds, seed + int(seconds)),
            "reference_tokens": synthetic_tokens(24, seed),
            "seed": seed,
            "sentences": [
                {"name": "short", "tokens": synthetic_tokens(32, seed + 1)},
                {
                    "name": "long",
                    "tokens": synthetic_tokens(120, seed + 2),
                },
            ],
        }
        for seconds in (5.0, 10.0, 20.0)
    ]


def compare_arrays(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    names = sorted(set(expected) | set(actual))
    _require(set(expected) == set(actual), f"Parity outputs differ: {names}")
    return {name: array_metrics(expected[name], actual[name]) for name in names}


def assert_component_parity(report: Mapping[str, Any]) -> None:
    """Require every deterministic component graph to match its PyTorch source."""
    for component, metrics in report.items():
        for name, record in metrics.items():
            if not record["all_finite"]:
                raise AkinvoxBuildError(
                    f"{component}/{name} produced non-finite values"
                )
            if record["cosine_similarity"] < 0.9999:
                raise AkinvoxBuildError(
                    f"{component}/{name} cosine {record['cosine_similarity']:.8f} "
                    f"is below 0.9999 (max error {record['max_abs_error']:.3g})"
                )


def assert_waveform_parity(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """Compare stochastic waveforms by deterministic audio metrics."""
    reference = np.asarray(expected, dtype=np.float64).reshape(-1)
    candidate = np.asarray(actual, dtype=np.float64).reshape(-1)
    metrics = array_metrics(expected, actual)
    rms = float(np.sqrt(np.mean(np.square(reference))))
    actual_rms = float(np.sqrt(np.mean(np.square(candidate))))
    rms_ratio = actual_rms / max(rms, np.finfo(np.float64).eps)
    peak = float(np.max(np.abs(reference)))
    actual_peak = float(np.max(np.abs(candidate)))
    peak_ratio = actual_peak / max(peak, np.finfo(np.float64).eps)
    if not 0.5 <= rms_ratio <= 2.0:
        raise AkinvoxBuildError(
            f"Waveform RMS ratio {rms_ratio:.4f} is outside [0.5, 2.0]"
        )
    if not 0.5 <= peak_ratio <= 2.0:
        raise AkinvoxBuildError(
            f"Waveform peak ratio {peak_ratio:.4f} is outside [0.5, 2.0]"
        )
    return {
        **metrics,
        "rms": rms,
        "onnx_rms": actual_rms,
        "rms_ratio": rms_ratio,
        "peak": peak,
        "onnx_peak": actual_peak,
        "peak_ratio": peak_ratio,
    }


def assert_end_to_end_parity(cases_report: Sequence[Mapping[str, Any]]) -> None:
    """Require exact durations, output length and reference masks per case."""
    for case in cases_report:
        if not case["reference_mask_exact"]:
            raise AkinvoxBuildError(f"Reference mask parity failed for {case['name']}")
        for synthesis in case["synthesis"]:
            label = f"{case['name']}/{synthesis['name']}"
            if not synthesis["durations_exact"]:
                raise AkinvoxBuildError(f"Duration parity failed for {label}")
            if not synthesis["output_length_exact"]:
                raise AkinvoxBuildError(f"Output length parity failed for {label}")


def probe_inputs() -> dict[str, tuple[Any, ...]]:
    """Return one representative input tuple per component graph."""
    import torch

    mel_frames = 64
    reference_tokens = 24
    memory_rows = mel_frames // 4
    text_tokens = 32
    synthesis_frames = 8
    source_frames = (UPSAMPLE_SCALE // ISTFT_HOP) * (2 * synthesis_frames) + 1

    return {
        "reference_wavlm": (torch.zeros(1, IDENTITY_SAMPLE_RATE),),
        "reference_encoders": (torch.zeros(1, 1, 80, mel_frames),),
        "reference_mapper": (
            torch.zeros(1, 80, mel_frames),
            torch.tensor([mel_frames], dtype=torch.int64),
            torch.zeros(1, reference_tokens, dtype=torch.int64),
            torch.tensor([reference_tokens], dtype=torch.int64),
            torch.zeros(1, 512),
            torch.zeros(1, 128),
            torch.zeros(1, 128),
        ),
        "prosody": (
            torch.zeros(1, text_tokens, dtype=torch.int64),
            torch.zeros(1, 128),
            torch.zeros(1, memory_rows, 192),
            torch.zeros(1, memory_rows, dtype=torch.bool),
        ),
        "curves": (torch.zeros(1, 640, synthesis_frames), torch.zeros(1, 128)),
        "decoder": (
            torch.zeros(1, 512, synthesis_frames),
            torch.zeros(1, 2 * synthesis_frames),
            torch.zeros(1, 2 * synthesis_frames),
            torch.zeros(1, 128),
            torch.zeros(1, 2 * (ISTFT_FFT // 2 + 1), source_frames),
        ),
    }


def _native_mels(wave: np.ndarray) -> tuple[Any, Any]:
    """Compute AkinVox reference mels with the pinned upstream PyTorch path."""
    import torch

    mel = _upstream_preprocess()(torch.from_numpy(np.asarray(wave, dtype=np.float32)))
    padded = np.pad(np.asarray(wave, dtype=np.float32), (5000, 5000))
    encoder_mel = _upstream_preprocess()(torch.from_numpy(padded)).squeeze()
    even = encoder_mel.shape[-1] - encoder_mel.shape[-1] % 2
    return mel, encoder_mel[:, :even][None, None]


def _upstream_preprocess() -> Any:
    module = sys.modules.get("kokoro_cloning.reference_mel")
    if module is None:
        raise AkinvoxBuildError("Pinned reference_mel module is not loaded")
    return module.preprocess


def _native_encode_reference(
    mapper: Any, mel: Any, lengths: Any, reference_ids: Any
) -> Any:
    import torch

    mask = mapper.mask(lengths, mel.shape[-1])
    value = torch.nn.functional.gelu(mapper.conv1(mel.masked_fill(mask[:, None], 0)))
    halved = (lengths + 1) // 2
    value = value.masked_fill(mapper.mask(halved, value.shape[-1])[:, None], 0)
    value = torch.nn.functional.gelu(mapper.conv2(value))
    halved = (halved + 1) // 2
    mask = mapper.mask(halved, value.shape[-1])
    columns = int(reference_ids.shape[1])
    phone_mask = torch.zeros(1, columns, dtype=torch.bool)
    value = mapper.reference_transcript(
        mapper.norm(value.transpose(1, 2)), mask, reference_ids, phone_mask
    )
    value = mapper.sequence(
        value + _positions(value.shape[1], 192, value.device, value.dtype)[None],
        src_key_padding_mask=mask,
    )
    value = value.masked_fill(mask[..., None], 0)
    denominator = halved[:, None].to(value.dtype)
    mean = value.sum(1) / denominator
    variance = (value - mean[:, None]).square().masked_fill(mask[..., None], 0)
    stats = torch.cat((mean, (variance / denominator).clamp_min(1.0e-8).sqrt()), -1)
    return (value, mask, stats)


def _native_wavlm(wave: np.ndarray) -> Any:
    import torch

    identity = sys.modules.get("_akinvox_wavlm_identity")
    if identity is None:
        raise AkinvoxBuildError("WavLM identity model is not loaded")
    wave16 = _resample_wave(wave, IDENTITY_SAMPLE_RATE)
    inputs = torch.from_numpy(np.ascontiguousarray(wave16[None, :]))
    embedding = identity(input_values=inputs).embeddings.float()
    return torch.nn.functional.normalize(embedding, dim=-1, eps=1.0e-8)


def _native_speech(
    core: Mapping[str, Any],
    mapper: Any,
    input_ids: Any,
    mask: Any,
    style: Any,
    prompt: Any,
) -> dict[str, Any]:
    import torch

    hidden = core["bert"](input_ids, attention_mask=(~mask).int())
    duration_embedding = core["bert_encoder"](hidden).transpose(-1, -2)
    duration_embedding = mapper.prosody(duration_embedding, mask, prompt)
    lengths = torch.full((1,), mask.shape[-1], dtype=torch.long)
    duration_context = core["predictor"].text_encoder(
        duration_embedding, style[:, 128:], lengths, mask
    )
    duration_hidden, _ = core["predictor"].lstm(duration_context)
    duration_logits = core["predictor"].duration_proj(duration_hidden)
    duration_float = torch.sigmoid(duration_logits).sum(dim=-1).squeeze(0)
    durations = torch.round(duration_float).clamp(min=1).long()
    index = torch.repeat_interleave(torch.arange(durations.numel()), durations)
    en = duration_context.transpose(0, 2, 1)[:, :, index]
    f0_curve, n_curve = core["predictor"].predict_f0_noise(en, style[:, 128:])
    content = mapper.content(
        core["text_encoder"](input_ids, lengths, mask), mask, prompt
    )
    asr = content[:, :, index]
    waveform = core["decoder"](asr, f0_curve, n_curve, style[:, :128])
    return {
        "pred_dur": durations.numpy(),
        "f0_curve": f0_curve.detach().cpu().numpy(),
        "n_curve": n_curve.detach().cpu().numpy(),
        "audio": waveform.detach().cpu().numpy().reshape(-1),
    }


def _resample_wave(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    source_rate = SAMPLE_RATE
    count = round(len(wave) * sample_rate / source_rate)
    return np.interp(
        np.linspace(0, len(wave) - 1, count), np.arange(len(wave)), wave
    ).astype(np.float32)


def run_parity(
    *,
    core: Mapping[str, Any],
    mapper: Any,
    wrappers: Mapping[str, Any],
    out_dir: Path,
    names: Mapping[str, str],
    source_params: Path,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare every component and full enrollment/synthesis against PyTorch."""
    import onnxruntime as ort

    sessions = {
        component: ort.InferenceSession(
            str(out_dir / names[component]), providers=["CPUExecutionProvider"]
        )
        for component in COMPONENTS
    }
    probes = probe_inputs()
    component_report = {
        component: _component_case(
            wrappers[component], sessions[component], probes[component]
        )
        for component in COMPONENTS
    }
    assert_component_parity(component_report)

    native = _NativePipeline(core, mapper)
    consumer = _ConsumerPipeline(sessions, source_params)
    cases_report = []
    for case in cases:
        wave = np.asarray(case["reference_audio"], dtype=np.float32)
        tokens = np.asarray(case["reference_tokens"], dtype=np.int64)
        expected_reference = native.enroll(wave, tokens)
        actual_reference = consumer.enroll(wave, tokens)
        entry: dict[str, Any] = {
            "name": str(case["name"]),
            "reference_seconds": float(case["reference_seconds"]),
            "enrollment": compare_arrays(
                {
                    "style": expected_reference["style"],
                    "reference_memory": expected_reference["memory"],
                },
                {
                    "style": actual_reference["style"],
                    "reference_memory": actual_reference["memory"],
                },
            ),
            "reference_mask_exact": bool(
                np.array_equal(expected_reference["mask"], actual_reference["mask"])
            ),
            "synthesis": [],
        }
        for sentence in case["sentences"]:
            expected = native.synthesize(
                expected_reference, sentence["tokens"], seed=case["seed"]
            )
            actual = consumer.synthesize(
                actual_reference, sentence["tokens"], seed=case["seed"]
            )
            entry["synthesis"].append(
                {
                    "name": str(sentence["name"]),
                    "durations_exact": bool(
                        np.array_equal(expected["pred_dur"], actual["pred_dur"])
                    ),
                    "output_length_exact": expected["audio"].shape
                    == actual["audio"].shape,
                    "curves": compare_arrays(
                        {
                            "f0_curve": expected["f0_curve"],
                            "n_curve": expected["n_curve"],
                        },
                        {"f0_curve": actual["f0_curve"], "n_curve": actual["n_curve"]},
                    ),
                    "waveform": assert_waveform_parity(
                        expected["audio"], actual["audio"]
                    ),
                }
            )
        cases_report.append(entry)
    assert_end_to_end_parity(cases_report)

    return {
        "status": "pass",
        "seeded_cases": len(cases),
        "components": component_report,
        "end_to_end": cases_report,
        "summary": {
            "status": "pass",
            "component_count": len(component_report),
            "cases": len(cases_report),
        },
    }


def _component_case(
    wrapper: Any, session: Any, args: tuple[Any, ...]
) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        expected = wrapper(*args)
    if not isinstance(expected, tuple):
        expected = (expected,)
    feeds = {
        node.name: value.detach().cpu().numpy()
        for node, value in zip(session.get_inputs(), args, strict=True)
    }
    actual = session.run(None, feeds)
    names = [node.name for node in session.get_outputs()]
    return compare_arrays(
        {
            name: value.detach().cpu().numpy()
            for name, value in zip(names, expected, strict=True)
        },
        {name: np.asarray(value) for name, value in zip(names, actual, strict=True)},
    )


class _NativePipeline:
    """AkinVox PyTorch reference path used as the parity ground truth."""

    def __init__(self, core: Mapping[str, Any], mapper: Any) -> None:
        self.core = core
        self.mapper = mapper

    def enroll(self, wave: np.ndarray, tokens: np.ndarray) -> dict[str, Any]:
        import torch

        mel, encoder_mel = _native_mels(wave)
        reference_ids = torch.tensor([[0, *tokens.tolist(), 0]], dtype=torch.long)
        lengths = torch.tensor([reference_ids.shape[1]], dtype=torch.long)
        prompt = _native_encode_reference(self.mapper, mel, lengths, reference_ids)
        wavlm, raw_sdec, raw_spred = self._observations(encoder_mel, wave)
        style = self.mapper(wavlm, raw_sdec, raw_spred, lengths - 1, prompt)
        return {
            "style": style.detach().cpu().numpy(),
            "memory": prompt[0].detach().cpu().numpy(),
            "mask": prompt[1].detach().cpu().numpy(),
        }

    def synthesize(
        self, reference: Mapping[str, Any], tokens: np.ndarray, *, seed: int
    ) -> dict[str, Any]:
        import torch

        torch.manual_seed(seed)
        input_ids = torch.tensor([[0, *tokens.tolist(), 0]], dtype=torch.long)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        prompt = (
            torch.from_numpy(np.asarray(reference["memory"], dtype=np.float32)),
            torch.from_numpy(np.asarray(reference["mask"], dtype=np.bool_)),
            None,
        )
        style = torch.from_numpy(np.asarray(reference["style"], dtype=np.float32))
        return _native_speech(self.core, self.mapper, input_ids, mask, style, prompt)

    def _observations(self, encoder_mel: Any, wave: np.ndarray) -> tuple[Any, ...]:
        import torch

        mel = torch.from_numpy(np.asarray(encoder_mel, dtype=np.float32))
        raw_sdec = self.core["style_encoder"](mel).float()
        raw_spred = self.core["predictor_encoder"](mel).float()
        return (_native_wavlm(wave), raw_sdec, raw_spred)


class _ConsumerPipeline:
    """ONNX component pipeline with consumer-side signal processing."""

    def __init__(self, sessions: Mapping[str, Any], source_params: Path) -> None:
        self.sessions = sessions
        self.source_params = source_params

    def enroll(self, wave: np.ndarray, tokens: np.ndarray) -> dict[str, Any]:
        mapper_mel, encoder_mel = reference_mels(wave)
        raw_sdec, raw_spred = self.sessions["reference_encoders"].run(
            None, {"mel": encoder_mel}
        )
        (wavlm,) = self.sessions["reference_wavlm"].run(
            None,
            {
                "input_values": np.ascontiguousarray(
                    _resample_wave(wave, IDENTITY_SAMPLE_RATE)[None, :]
                )
            },
        )
        reference_ids = np.asarray([[0, *tokens.tolist(), 0]], dtype=np.int64)
        style, memory, mask = self.sessions["reference_mapper"].run(
            None,
            {
                "mel": mapper_mel,
                "mel_lengths": np.asarray([mapper_mel.shape[-1]], dtype=np.int64),
                "reference_ids": reference_ids,
                "reference_lengths": np.asarray(
                    [reference_ids.shape[1]], dtype=np.int64
                ),
                "wavlm": np.asarray(wavlm, dtype=np.float32),
                "raw_sdec": np.asarray(raw_sdec, dtype=np.float32),
                "raw_spred": np.asarray(raw_spred, dtype=np.float32),
            },
        )
        return {"style": style, "memory": memory, "mask": mask}

    def synthesize(
        self, reference: Mapping[str, Any], tokens: np.ndarray, *, seed: int
    ) -> dict[str, Any]:
        input_ids = np.asarray([[0, *tokens.tolist(), 0]], dtype=np.int64)
        style = np.asarray(reference["style"], dtype=np.float32)
        pred_dur, duration_context, t_en = self.sessions["prosody"].run(
            None,
            {
                "input_ids": input_ids,
                "style_dur": np.ascontiguousarray(style[:, 128:]),
                "reference_memory": np.asarray(reference["memory"], dtype=np.float32),
                "reference_mask": np.asarray(reference["mask"], dtype=np.bool_),
            },
        )
        durations = np.asarray(pred_dur).reshape(-1)
        f0_curve, n_curve = self.sessions["curves"].run(
            None,
            {
                "en": expand_frames(duration_context.transpose(0, 2, 1), durations),
                "style_dur": np.ascontiguousarray(style[:, 128:]),
            },
        )
        (audio,) = self.sessions["decoder"].run(
            None,
            {
                "asr": expand_frames(t_en, durations),
                "f0_curve": f0_curve,
                "n_curve": n_curve,
                "style_acou": np.ascontiguousarray(style[:, :128]),
                "har": harmonic_source(f0_curve, seed, self.source_params),
            },
        )
        return {
            "pred_dur": durations,
            "f0_curve": np.asarray(f0_curve),
            "n_curve": np.asarray(n_curve),
            "audio": np.asarray(audio).reshape(-1),
        }


# --------------------------------------------------------------------------
# Bundle assembly
# --------------------------------------------------------------------------


def verify_no_runtime_checkpoints(out_dir: Path) -> None:
    """Guarantee that no build-time checkpoint is shipped as a runtime asset."""
    forbidden = {
        "adapter.pt",
        "reference_mapper.pt",
        "reference_encoders.pt",
        "kokoro-v1_0.pth",
        "pytorch_model.bin",
    }
    released = {path.name for path in out_dir.iterdir() if path.is_file()}
    _require(
        not (released & forbidden),
        f"Build-time checkpoints leaked into the bundle: {sorted(released & forbidden)}",
    )
    for suffix in (".pt", ".pth", ".bin", ".safetensors"):
        leaked = sorted(path.name for path in out_dir.rglob(f"*{suffix}"))
        _require(not leaked, f"PyTorch artifacts leaked into the bundle: {leaked}")


def export_bundle(
    profile: Mapping[str, Any],
    out_dir: Path,
    *,
    cache_dir: Path,
    opset: int,
    run_checker: bool,
) -> dict[str, Any]:
    """Build the model, materialize LoRA, export six graphs and prove parity."""
    import onnxruntime as ort
    import torch

    sources = resolve_sources(profile, cache_dir)
    config = json.loads(sources["akinvox:config.json"].read_text(encoding="utf-8"))
    _require(
        config.get("format") == "akinvox-kokoro-cloning-adapter-v1",
        "AkinVox adapter config format changed",
    )
    _require(
        len(config["targets"]) == EXPECTED_ADAPTER_COUNT,
        "AkinVox target inventory changed",
    )

    upstream = _load_upstream(sources["source_code"])
    core, mapper, audit = build_native_model(upstream, config, sources)
    lora = materialize_lora(core, config)

    names = build_filenames()
    source_params = extract_source_params(
        core["decoder"], out_dir / names["source_params"]
    )
    shutil.copyfile(sources["akinvox:config.json"], out_dir / names["config"])

    istft = install_onnx_istft(core["decoder"].generator)
    types = wrapper_types()
    slope = float(upstream.istftnet.LRELU_SLOPE)
    wrappers = {
        "reference_wavlm": types.ReferenceWavLM(_load_wavlm(sources, cache_dir)),
        "reference_encoders": types.ReferenceEncoders(core),
        "reference_mapper": types.ReferenceMapper(mapper),
        "prosody": types.Prosody(core, mapper),
        "curves": types.Curves(core),
        "decoder": types.Decoder(core, slope),
    }
    for wrapper in wrappers.values():
        wrapper.eval().requires_grad_(False)

    probes = probe_inputs()
    exports = [
        export_component(
            wrappers[name],
            probes[name],
            out_dir / names[name],
            component=name,
            opset=opset,
        )
        for name in COMPONENTS
    ]

    report = run_parity(
        core=core,
        mapper=mapper,
        wrappers=wrappers,
        out_dir=out_dir,
        names=names,
        source_params=out_dir / names["source_params"],
        cases=parity_cases(
            int(profile.get("export_validation", {}).get("export_seed", 20260926))
        ),
    )
    parity = write_parity_report(out_dir, report)
    _require(
        report["status"] == "pass",
        "PyTorch/ONNX parity failed; see parity-report.json",
    )

    if run_checker:
        import onnx

        for record in exports:
            onnx.checker.check_model(str(out_dir / record["path"]))

    verify_no_runtime_checkpoints(out_dir)
    session = ort.InferenceSession(
        str(out_dir / names["decoder"]), providers=["CPUExecutionProvider"]
    )
    del session
    return {
        "source_params": source_params,
        "exporter": {
            "module": "scripts/akinvox_cloning.py",
            "opset": opset,
            "torch_version": str(torch.__version__),
            "onnxruntime_version": str(ort.__version__),
            "components": exports,
            "checkpoint_load": audit,
            "lora_materialization": lora,
            "decoder_reconstruction": {
                "native_delegate_validation": {"cases": []},
                **istft,
            },
            "parity_report": parity,
            "parity": report["summary"],
            "inputs": {"voice_mode": "reference"},
            "outputs": ["audio"],
            "random_source_ops": [],
        },
    }


def _load_wavlm(sources: Mapping[str, Path], cache_dir: Path) -> Any:
    from transformers import WavLMForXVector

    folder = sources["wavlm:config.json"].parent
    with (folder / "preprocessor_config.json").open("r", encoding="utf-8") as handle:
        preprocessing = json.load(handle)
    _require(
        preprocessing.get("do_normalize") is True,
        "WavLM preprocessor must declare do_normalize",
    )
    identity = WavLMForXVector.from_pretrained(
        str(folder), local_files_only=True, use_safetensors=False
    )
    return identity.cpu().eval().requires_grad_(False)


def build_akinvox_profile(
    profile_key: str,
    profile: Mapping[str, Any],
    out_root: Path,
    *,
    opset: int,
    cache_dir: Path,
    run_checker: bool,
) -> Path:
    """Build the ONNX-only AkinVox cloning bundle for one profile."""
    validate_source_pins(profile)
    runtime_metadata()

    out_dir = out_root / profile_key
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = export_bundle(
        profile, out_dir, cache_dir=cache_dir, opset=opset, run_checker=run_checker
    )

    names = build_filenames()
    verify_no_runtime_checkpoints(out_dir)
    published = component_filenames()
    artifacts = [
        {
            "component": component,
            "filename": published[component],
            "source": names[component],
            "role": "model",
            "quality": "fp32",
            "format": "onnx",
            "sha256": summary["exporter"]["components"][index]["sha256"],
            "size": summary["exporter"]["components"][index]["size"],
        }
        for index, component in enumerate(COMPONENTS)
    ]
    manifest = {
        "profile": profile_key,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "layout": LAYOUT,
        "voice_mode": VOICE_MODE,
        "source_repo": str(profile["repo_id"]),
        "revision": str(profile["revision"]),
        "source_artifacts": pinned_sources(profile),
        "license": str(profile["license"]),
        "language": str(profile["language"]),
        "sample_rate": SAMPLE_RATE,
        "speakers": [],
        "components": artifacts,
        "source_params": summary["source_params"],
        "onnx_contract": component_contract(),
        "runtime": runtime_metadata(),
        "publication": dict(profile.get("release") or {}),
        "exporter": summary["exporter"],
    }
    (out_dir / "bundle.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export the AkinVox cloning ONNX bundle"
    )
    parser.add_argument("profile", nargs="?", default=MODEL_ID)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).with_name("kokoro_profiles.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("build"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args(argv)

    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    profile = profiles[args.profile]
    out_dir = build_akinvox_profile(
        args.profile,
        profile,
        args.out,
        opset=args.opset,
        cache_dir=args.out / args.profile / ".cache",
        run_checker=not args.skip_check,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AkinvoxBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
