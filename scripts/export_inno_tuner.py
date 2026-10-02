#!/usr/bin/env python3
"""Build-time exporter for the Inno v0.2 Kokoro voicepack tuner.

The Inno tuner is an optional voice-enrollment capability appended to the
Kokoro v1.0 release. It is a different artifact from the mirrored v1.0
synthesis model, so its export logic lives here instead of the generic v1.0
mirror code. PyTorch, safetensors and the pinned upstream v0.2 source are
build-time-only inputs; the emitted artifacts are one ONNX graph, a pickle-free
NumPy tuner metadata archive with a JSON sidecar, parity fixtures, provenance
and checksums. No PyTorch tensor is shipped as a runtime asset.

The fbank frontend is deliberately kept out of the ONNX graph: deterministic
host NumPy code below computes features and enrollment statistics, and the
graph starts at the ResNet speaker encoder. Toolchain imports stay inside
functions so pin, metadata and host-algorithm helpers remain importable
without a build environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "catalog" / "releases.json"

ENROLLER_ID = "inno-v0.2"
ENROLLER_KIND = "kokoro-voicepack-tuner"
ENROLLER_VERSION = "0.2.0"
SOURCE_FILE_NAMES = ("model.safetensors", "config.json")

GRAPH_FILENAME = "inno-voicepack-v0.2.onnx"
TUNER_NPZ_FILENAME = "inno-tuner-v0.2.npz"
TUNER_JSON_FILENAME = "inno-tuner-v0.2.json"
BUNDLE_FILENAME = "bundle.json"
PARITY_FIXTURES_FILENAME = "parity-fixtures.npz"
PARITY_REPORT_FILENAME = "parity-report.json"

ROWS, STYLE, HALF = 510, 256, 128
VOICEPACK_FORMAT = "kokoro-voicepack-v1"
GRAPH_INPUTS = ("fbank", "tilt", "head_stats", "blend_weights")
GRAPH_OUTPUT = "voicepack"

TUNER_NPZ_FIELDS = (
    "blend_stats",
    "blend_grades",
    "blend_scale",
    "blend_gate",
    "grade_pen",
)

SR_ENC = 16000
SR_TILT = 24000
SR_KOKORO = 24000
FBANK_FRAME, FBANK_HOP, FBANK_NFFT, NMEL = 400, 160, 512, 80
REF_MAX_S = 30
MIN_PACKS = 3

FRAME_S = 0.02
HOP = int(SR_ENC * FRAME_S)
F0_MAX = 400.0
F0_MIN = 60.0
MEDIAN_FRAMES = 30
ST_REF_HZ = 100.0
TILT_WIN, TILT_HOP = 1024, 256
TILT_BAND = (300.0, 4000.0)
CEPS_N = 4096
CEPS_MIN_PEAK = 0.01
CEPS_TOL = 0.25
LAG_TOL = 0.9
CEPS_CEIL = 1.6
HEAD_CHUNK_S = 6.0

LICENSE_COMPONENTS = {
    "inno_code": {"component": "Inno code/adapter", "license": "Apache-2.0"},
    "wespeaker_code": {"component": "WeSpeaker ResNet34 code", "license": "Apache-2.0"},
    "voxceleb_init": {
        "component": "VoxCeleb-trained ResNet34-LM initialization",
        "license": "CC-BY-4.0",
    },
    "speaker_encoder": {
        "component": "UniSpeech-SAT teacher/derived speaker encoder weights",
        "license": "CC-BY-SA-3.0",
    },
}
ARTIFACT_LICENSE = (
    "Apache-2.0 code and adapter with CC-BY-SA-3.0 speaker encoder weights; "
    "see MODEL_LICENSES.md"
)


class InnoTunerBuildError(RuntimeError):
    """Raised when the Inno tuner build inputs or outputs are invalid."""


@dataclass(frozen=True)
class BlendTables:
    names: tuple[str, ...]
    stats: np.ndarray  # [V, 3] F0 mean st, F0 sd st, syllables/s
    grades: np.ndarray  # [V] pack quality, A = 4.0, C = 2.0
    scale: np.ndarray  # [3] per-stat cost scale
    gate: float  # semitones
    grade_pen: float


@dataclass(frozen=True)
class TunerState:
    enc: dict[str, np.ndarray]
    style: dict[str, np.ndarray]
    tilt_dir: np.ndarray  # [STYLE]
    blend_rows: np.ndarray  # [V, ROWS, HALF]
    blend: BlendTables
    head_weight: np.ndarray  # [3, HALF]
    head_mu: np.ndarray  # [2]
    head_sd: np.ndarray  # [2]
    tilt_mean: float
    tilt_sd: float
    version: str
    name: str
    meta: dict[str, str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InnoTunerBuildError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Source pins
# --------------------------------------------------------------------------


def load_augmentation(
    releases_path: Path = RELEASES, augmentation_id: str = ENROLLER_ID
) -> dict[str, Any]:
    releases = json.loads(releases_path.read_text(encoding="utf-8"))
    for spec in releases.get("releases", {}).values():
        for augmentation in spec.get("augmentations", []):
            if augmentation.get("id") == augmentation_id:
                return augmentation
    raise InnoTunerBuildError(f"Unknown augmentation: {augmentation_id}")


def pinned_sources(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tuner": {
            "repository": str(spec["source_repository"]),
            "revision": str(spec["source_revision"]),
            "release_tag": str(spec["source_tag"]),
            "files": {
                name: dict(details)
                for name, details in spec["source_files"].items()
            },
        },
        "source_code": {
            "repository": str(spec["code_repository"]),
            "revision": str(spec["code_revision"]),
            "tag": str(spec["code_tag"]),
        },
        "declared_version": str(spec["declared_version"]),
    }


def validate_source_pins(spec: Mapping[str, Any]) -> None:
    _require(
        spec.get("id") == ENROLLER_ID and spec.get("kind") == ENROLLER_KIND,
        "Augmentation identity is not the Inno voicepack tuner",
    )
    for field in ("source_revision", "code_revision"):
        value = str(spec.get(field, ""))
        _require(
            len(value) == 40 and value.isalnum() and value.islower(),
            f"{field} must be a lowercase commit SHA, not a moving ref",
        )
    for repository_field in ("source_repository", "code_repository"):
        _require(
            bool(spec.get(repository_field)),
            f"{repository_field} is missing",
        )
    _require(
        str(spec.get("source_type", "")) == "huggingface",
        "Tuner weights must come from a pinned Hugging Face revision",
    )
    _require(
        spec.get("declared_version") == ENROLLER_VERSION,
        f"Declared tuner version must be {ENROLLER_VERSION}",
    )
    files = spec.get("source_files")
    _require(
        isinstance(files, dict) and set(files) == set(SOURCE_FILE_NAMES),
        "Tuner pin must declare model.safetensors and config.json",
    )
    for name, details in files.items():
        _require(
            isinstance(details.get("size"), int) and details["size"] > 0,
            f"{name} pin is missing a positive size",
        )
        digest = str(details.get("sha256", ""))
        _require(
            len(digest) == 64 and digest.islower() and digest.isalnum(),
            f"{name} pin is missing a lowercase SHA-256",
        )


def source_checksums(sources: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"size": path.stat().st_size, "sha256": sha256(path)}
        for name, path in sorted(sources.items())
    }


def verify_sources(spec: Mapping[str, Any], sources: Mapping[str, Path]) -> None:
    for name, details in spec["source_files"].items():
        path = sources.get(name)
        _require(path is not None and path.is_file(), f"Missing build input: {name}")
        _require(
            path.stat().st_size == int(details["size"]),
            f"{name} size does not match the pinned size",
        )
        _require(
            sha256(path) == details["sha256"],
            f"{name} SHA-256 does not match the pinned digest",
        )


def resolve_sources(spec: Mapping[str, Any], cache_dir: Path) -> dict[str, Path]:
    """Download pinned build inputs into the cache and verify them."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for name in SOURCE_FILE_NAMES:
        target = cache_dir / name
        if not (
            target.is_file()
            and target.stat().st_size == int(spec["source_files"][name]["size"])
            and sha256(target) == spec["source_files"][name]["sha256"]
        ):
            url = (
                f"https://huggingface.co/{spec['source_repository']}/resolve/"
                f"{urllib.parse.quote(str(spec['source_revision']), safe='')}/"
                f"{urllib.parse.quote(name, safe='/')}?download=true"
            )
            _download(url, target)
        sources[name] = target
    verify_sources(spec, sources)
    return sources


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "kokoro-onnx-models-inno-export"}
    )
    partial = target.with_name(target.name + ".part")
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial.open("wb") as file,
        ):
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def resolve_source_code(spec: Mapping[str, Any], cache_dir: Path) -> Path:
    """Download the pinned upstream v0.2 source tree (parity oracle, build-time only)."""
    import tarfile

    revision = str(spec["code_revision"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkout = cache_dir / f"inno-kokoro-{revision}"
    if not (checkout / "inno_ref").is_dir():
        archive = cache_dir / f"inno-kokoro-{revision}.tar.gz"
        if not archive.is_file():
            _download(
                f"https://github.com/{spec['code_repository']}/archive/{revision}.tar.gz",
                archive,
            )
        with tarfile.open(archive) as tar:
            # filter= is Python 3.12+; this package still supports 3.10.
            try:
                tar.extractall(cache_dir, filter="data")
            except TypeError:
                tar.extractall(cache_dir)
    _require(
        (checkout / "inno_kokoro" / "enroll.py").is_file(),
        "Upstream source checkout is missing inno_kokoro/enroll.py",
    )
    return checkout

# --------------------------------------------------------------------------
# Weight loading
# --------------------------------------------------------------------------

_SAFETENSORS_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read a safetensors file without the safetensors package (pickle-free)."""
    with path.open("rb") as file:
        header_length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_length))
        data = file.read()
    meta = header.pop("__metadata__", {})
    tensors: dict[str, np.ndarray] = {}
    for name, entry in header.items():
        dtype = _SAFETENSORS_DTYPES.get(str(entry["dtype"]))
        _require(dtype is not None, f"Unsupported safetensors dtype for {name}")
        begin, end = (int(offset) for offset in entry["data_offsets"])
        values = np.frombuffer(data, dtype=dtype, count=(end - begin) // np.dtype(dtype).itemsize, offset=begin)
        tensors[name] = values.reshape([int(dim) for dim in entry["shape"]]).copy()
    return tensors, dict(meta)


def load_tuner_state(path: Path) -> TunerState:
    tensors, meta = read_safetensors(path)

    def _float(name: str) -> np.ndarray:
        _require(name in tensors, f"{path.name} is missing tensor {name!r}")
        return np.ascontiguousarray(tensors[name], dtype=np.float32)

    enc = {
        name[len("enc.") :]: value
        for name, value in tensors.items()
        if name.startswith("enc.") and not name.endswith("num_batches_tracked")
    }
    _require(bool(enc), f"{path.name} has no enc.* speaker-encoder keys")
    enc = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in enc.items()}
    style = {
        name[len("heads.style.") :]: np.ascontiguousarray(value, dtype=np.float32)
        for name, value in tensors.items()
        if name.startswith("heads.style.")
    }
    _require(set(style) == {"0.weight", "0.bias", "2.weight", "2.bias"}, "Style head keys are incomplete")

    tilt_dir = _float("heads.tilt.weight").reshape(-1)
    blend_rows = _float("blend.rows")
    blend_stats = _float("blend.stats")
    blend_grades = _float("blend.grades").reshape(-1)
    head_weight = _float("head.W")
    head_mu = _float("head.mu").reshape(-1)
    head_sd = _float("head.sd").reshape(-1)

    packs = blend_rows.shape[0]
    _require(blend_rows.shape == (packs, ROWS, HALF), "blend.rows must be [V, 510, 128]")
    _require(blend_stats.shape == (packs, 3), "blend.stats must be [V, 3]")
    _require(blend_grades.shape == (packs,), "blend.grades must be [V]")
    _require(tilt_dir.shape == (STYLE,), "heads.tilt.weight must be [256, 1]")
    _require(head_weight.shape == (3, HALF), "head.W must be [3, 128]")
    _require(head_mu.shape == (2,) and head_sd.shape == (2,), "head.mu/head.sd must be [2]")

    names = tuple(str(meta["blend_names"]).split(","))
    _require(len(names) == packs, "blend_names does not match blend.rows")
    scale = np.array([float(value) for value in str(meta["blend_scale"]).split(",")], dtype=np.float64)
    _require(scale.shape == (3,), "blend_scale must hold three values")
    for name in ("version", "blend_gate", "grade_pen", "tilt_mean", "tilt_sd"):
        _require(name in meta, f"{path.name} metadata is missing {name!r}")

    return TunerState(
        enc=enc,
        style=style,
        tilt_dir=tilt_dir,
        blend_rows=blend_rows,
        blend=BlendTables(
            names=names,
            stats=blend_stats.astype(np.float64),
            grades=blend_grades.astype(np.float64),
            scale=scale,
            gate=float(meta["blend_gate"]),
            grade_pen=float(meta["grade_pen"]),
        ),
        head_weight=head_weight,
        head_mu=head_mu,
        head_sd=head_sd,
        tilt_mean=float(meta["tilt_mean"]),
        tilt_sd=float(meta["tilt_sd"]),
        version=str(meta["version"]),
        name=str(meta.get("name", "inno_ref")),
        meta=dict(meta),
    )


# --------------------------------------------------------------------------
# Host feature algorithms (deterministic NumPy/SciPy; the fbank frontend stays
# out of the ONNX graph)
# --------------------------------------------------------------------------


def _st(hz: np.ndarray | float) -> np.ndarray | float:
    return 12.0 * np.log2(np.asarray(hz) / ST_REF_HZ)


def _hz(semitones: np.ndarray | float) -> np.ndarray | float:
    return ST_REF_HZ * 2.0 ** (np.asarray(semitones) / 12.0)


def host_resample(w: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return np.asarray(w, dtype=np.float32)
    from scipy.signal import resample_poly

    divisor = math.gcd(sr, target)
    return np.asarray(
        resample_poly(np.asarray(w, dtype=np.float64), target // divisor, sr // divisor),
        dtype=np.float32,
    )


def mel_filterbank() -> np.ndarray:
    """[80, 257] Kaldi triangular mel filters over a 512-point FFT at 16 kHz."""
    def mel(f: Any) -> Any:
        return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)

    spacing = (mel(8000.0) - mel(20.0)) / (NMEL + 1)
    left = mel(20.0) + np.arange(NMEL)[:, None] * spacing
    centers = mel(np.arange(FBANK_NFFT // 2) * SR_ENC / FBANK_NFFT)
    banks = np.minimum(
        (centers - left) / spacing, (left + 2 * spacing - centers) / spacing
    ).clip(min=0)
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)


def host_fbank(wav: np.ndarray) -> np.ndarray:
    """Kaldi fbank on [-1, 1] 16 kHz audio -> [F, 80] with per-utterance CMN."""
    frames = np.lib.stride_tricks.sliding_window_view(
        np.asarray(wav, dtype=np.float32) * 32768.0, FBANK_FRAME
    )[:: FBANK_HOP].copy()
    frames -= frames.mean(-1, keepdims=True)
    previous = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames -= 0.97 * previous
    window = np.hamming(FBANK_FRAME).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, n=FBANK_NFFT, axis=-1)) ** 2
    features = np.log(
        np.maximum(spectrum @ mel_filterbank().T, np.finfo(np.float32).eps)
    )
    features -= features.mean(0, keepdims=True)
    return features.astype(np.float32)


def _frame_rms(w: np.ndarray, hop: int) -> np.ndarray:
    count = len(w) // hop
    return np.sqrt((w[: count * hop].reshape(-1, hop) ** 2).mean(1))


def _fundamental(rows: np.ndarray, off: int, tol: float) -> np.ndarray:
    """Peak index per row resolved to the period rather than a multiple of it."""
    best = rows.argmax(1)
    pick = np.arange(len(rows))
    peak = rows[pick, best]
    idx = best.copy()
    for factor in (2, 3, 4):
        candidate = np.round((best + off) / factor).astype(np.int64) - off
        near = (candidate[:, None] + np.array([-1, 0, 1])).clip(0, rows.shape[1] - 1)
        values = rows[pick[:, None], near]
        winner = values.argmax(1)
        value = values.max(1)
        ok = (candidate >= 0) & (peak > 0) & (value >= tol * peak)
        idx = np.where(ok, near[pick, winner], idx)
    return idx


def _track_f0(w: np.ndarray, lo: float, hi: float, win: int = 2 * HOP):
    """Normalized cross-correlation pitch tracker at SR_ENC, one estimate per HOP."""
    lags = list(range(int(SR_ENC / hi), int(SR_ENC / lo) + 1))
    length = win + lags[-1]
    frames = np.lib.stride_tricks.sliding_window_view(
        np.pad(w, (0, length)), length
    )[::HOP]
    x = frames[:, :win]
    xx = (x**2).sum(1)
    correlations = []
    for lag in lags:
        y = frames[:, lag : lag + win]
        correlations.append(
            (x * y).sum(1) / np.maximum(np.sqrt(xx * (y**2).sum(1)), 1e-8)
        )
    correlations = np.stack(correlations, 1)
    best_lag = (
        lags[0] + _fundamental(correlations, lags[0], LAG_TOL)
    )[: len(w) // HOP].astype(np.float64)
    pad = MEDIAN_FRAMES // 2
    best_lag = np.pad(best_lag, (pad, MEDIAN_FRAMES - 1 - pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(best_lag, MEDIAN_FRAMES)
    # torch.median returns the lower middle value on even windows (MEDIAN_FRAMES = 30).
    best_lag = np.partition(windows, MEDIAN_FRAMES // 2 - 1, axis=1)[
        :, MEDIAN_FRAMES // 2 - 1
    ]
    return SR_ENC / best_lag, _frame_rms(w, HOP)


def host_rate(w: np.ndarray, sr: int, limit: int = 60) -> float:
    """Smoothed-energy peaks (roughly syllables) per second of speech."""
    hop = sr // 100
    energy = _frame_rms(np.asarray(w)[: limit * sr], hop)
    padded = np.pad(energy, (1, 1))
    energy = (padded[:-2] + padded[1:-1] + padded[2:]) / 3
    speech = energy > 0.05 * energy.max()
    is_peak = (energy[1:-1] > energy[:-2]) & (energy[1:-1] >= energy[2:]) & speech[1:-1]
    peaks = np.nonzero(is_peak)[0]
    if len(peaks) == 0:
        return 0.0
    syllables = 1 + int((np.diff(peaks) >= 8).sum())
    speech_seconds = float(speech.mean()) * len(energy) / 100
    return syllables / (speech_seconds + 1e-9)


def host_tilt(w: np.ndarray, sr: int) -> float:
    """Spectral tilt in dB per octave over TILT_BAND."""
    w = host_resample(w, sr, SR_TILT).astype(np.float64)
    padded = np.pad(w, (TILT_WIN // 2, TILT_WIN // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, TILT_WIN)[::TILT_HOP]
    window = np.hanning(TILT_WIN + 1)[:-1]
    spectrum = np.abs(np.fft.rfft(frames * window, n=TILT_WIN, axis=1)) ** 2
    power = spectrum.mean(0)
    frequencies = np.linspace(0, SR_TILT / 2, len(power))
    band = (frequencies > TILT_BAND[0]) & (frequencies < TILT_BAND[1])
    x = np.log2(frequencies[band])
    y = 10.0 * np.log10(power[band] + 1e-12)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def _cepstral_f0(w: np.ndarray) -> tuple[float, float]:
    frames = w[: len(w) // CEPS_N * CEPS_N].reshape(-1, CEPS_N)
    if len(frames):
        energy = (frames**2).mean(1)
        frames = frames[energy > np.quantile(energy, 0.7)]
    if not len(frames):
        return 0.0, 0.0
    window = np.hanning(CEPS_N + 1)[:-1].astype(np.float32)
    spectrum = np.log(np.abs(np.fft.rfft(frames * window, n=CEPS_N, axis=1)) + 1e-8)
    ceps = np.fft.irfft(spectrum - spectrum.mean(1, keepdims=True), n=CEPS_N).mean(0)
    quefrency = np.arange(len(ceps)) / SR_ENC
    mask = (quefrency > 1 / 300) & (quefrency < 1 / F0_MIN)
    band = ceps[mask]
    start = int(np.nonzero(mask)[0][0])
    index = int(_fundamental(band[None], start, CEPS_TOL)[0])
    return SR_ENC / (start + index), float(band[index])


def host_ceiling(w: np.ndarray, sr: int, fmax: float | None = None) -> float:
    if fmax is not None:
        return fmax
    candidate, peak = _cepstral_f0(host_resample(w, sr, SR_ENC))
    return min(F0_MAX, CEPS_CEIL * candidate) if peak > CEPS_MIN_PEAK else F0_MAX


def host_stats(
    w: np.ndarray, sr: int, fmax: float | None = None
) -> tuple[float, float, float]:
    """Log-F0 statistics of voiced frames: (mean st, std st, voiced fraction)."""
    fmax = host_ceiling(w, sr, fmax)
    w = host_resample(w, sr, SR_ENC)
    lo = F0_MIN
    voiced_st = np.zeros(0)
    voiced = np.zeros(0, dtype=bool)
    for _ in range(2):
        f0, energy = _track_f0(w, lo, fmax)
        voiced = (energy > 0.1 * energy.max()) & (f0 > lo) & (f0 < fmax)
        voiced_st = np.asarray(_st(f0[voiced]))
        if not len(voiced_st):
            return 0.0, 0.0, 0.0
        lo = min(0.65 * float(_hz(np.quantile(voiced_st, 0.75))), 180)
    return (
        float(voiced_st.mean()),
        float(voiced_st.std(ddof=1)),
        float(voiced.mean()),
    )


def host_head_stats(
    w: np.ndarray, sr: int, fmin: float = F0_MIN, fmax: float = F0_MAX
) -> tuple[float, float]:
    """(F0 mean st, F0 sd st) from Praat pitch at 10 ms over energy-gated frames."""
    import parselmouth

    w = np.asarray(w, dtype=np.float64)
    chunk = int(HEAD_CHUNK_S * sr)
    samples: list[tuple[float, float, int]] = []
    for start in range(0, len(w), chunk):
        x = w[start : start + chunk]
        if len(x) < 1.5 * sr:
            continue
        pitch = parselmouth.Sound(x, sampling_frequency=sr).to_pitch_ac(
            time_step=0.01, pitch_floor=fmin, pitch_ceiling=fmax
        )
        f0 = pitch.selected_array["frequency"]
        window = int(sr * 0.01)
        count = len(x) // window
        energy = np.sqrt((x[: count * window].reshape(-1, window) ** 2).mean(1))
        energy = energy[
            np.clip((np.asarray(pitch.xs()) / 0.01).astype(int), 0, len(energy) - 1)
        ]
        voiced = f0[(f0 > 0) & (energy > 0.1 * energy.max())]
        if len(voiced):
            semitones = np.asarray(_st(voiced))
            samples.append((float(semitones.mean()), float(semitones.std(ddof=0)), len(x)))
    if not samples:
        return 0.0, 0.0
    total = sum(length for _, _, length in samples)
    return (
        float(sum(value * length for value, _, length in samples) / total),
        float(sum(value * length for _, value, length in samples) / total),
    )


def host_blend_weights(tables: BlendTables, z: np.ndarray) -> dict[str, float]:
    """Reference stats -> {stock pack: weight}, nonnegative and summing to one."""
    from scipy.optimize import nnls

    distances = np.abs(tables.stats[:, 0] - z[0])
    allowed = distances <= tables.gate
    if allowed.sum() < MIN_PACKS:
        allowed = distances <= np.sort(distances)[MIN_PACKS - 1]
    penalty = tables.grade_pen * np.maximum(0.0, 2.0 - tables.grades[allowed])
    matrix = np.vstack(
        [
            tables.stats[allowed].T / tables.scale[:, None],
            10 * np.ones((1, int(allowed.sum()))),
            np.diag(penalty),
        ]
    )
    target = np.concatenate([z / tables.scale, [10.0], np.zeros(int(allowed.sum()))])
    weights, _ = nnls(matrix, target)
    weights = weights / weights.sum()
    return {
        name: float(weight)
        for name, weight in zip(np.array(tables.names)[allowed], weights)
        if weight > 1e-3
    }


def blend_weight_vector(tables: BlendTables, z: np.ndarray) -> np.ndarray:
    """Dense [V] blend weights for the ONNX `blend_weights` input."""
    named = host_blend_weights(tables, z)
    vector = np.zeros(len(tables.names), dtype=np.float32)
    for name, weight in named.items():
        vector[tables.names.index(name)] = weight
    return vector


def reference_features(
    wav: np.ndarray, sr: int, state: TunerState, fmax: float | None = None
) -> dict[str, np.ndarray]:
    """Reference audio -> the four ONNX graph inputs (host side only)."""
    dur = len(wav) / sr
    _require(dur >= 3, f"reference is {dur:.1f} s; need at least 3 s to measure prosody")
    wav = wav[: REF_MAX_S * sr]
    encoder_wav = host_resample(wav, sr, SR_ENC)
    fmax = host_ceiling(wav, sr, fmax)
    z = np.array([*host_stats(wav, sr, fmax)[:2], host_rate(wav, sr)], dtype=np.float64)
    head = np.array(host_head_stats(wav, sr, fmin=F0_MIN, fmax=fmax), dtype=np.float32)
    return {
        "fbank": host_fbank(encoder_wav)[None].astype(np.float32),
        "tilt": np.array([host_tilt(wav, sr)], dtype=np.float32),
        "head_stats": head[None],
        "blend_weights": blend_weight_vector(state.blend, z)[None],
    }


def assemble_voicepack(timbre_half: np.ndarray, predictor_rows: np.ndarray) -> np.ndarray:
    """Concatenate the timbre and predictor halves into a [510, 1, 256] pack."""
    timbre_half = np.asarray(timbre_half, dtype=np.float32).reshape(HALF)
    predictor_rows = np.asarray(predictor_rows, dtype=np.float32)
    _require(
        predictor_rows.shape == (ROWS, HALF),
        f"predictor rows must be [{ROWS}, {HALF}]",
    )
    timbre = np.broadcast_to(timbre_half, (ROWS, HALF))
    return np.ascontiguousarray(
        np.concatenate([timbre, predictor_rows], axis=-1)[:, None, :]
    )


# --------------------------------------------------------------------------
# Tuner metadata and enroller declaration
# --------------------------------------------------------------------------


def tuner_metadata_arrays(state: TunerState) -> dict[str, np.ndarray]:
    return {
        "blend_stats": state.blend.stats.astype(np.float32),
        "blend_grades": state.blend.grades.astype(np.float32),
        "blend_scale": state.blend.scale.astype(np.float32),
        "blend_gate": np.asarray(state.blend.gate, dtype=np.float32),
        "grade_pen": np.asarray(state.blend.grade_pen, dtype=np.float32),
    }


def tuner_metadata_document(
    state: TunerState, spec: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": 1,
        "id": str(spec["id"]),
        "kind": str(spec["kind"]),
        "version": state.version,
        "name": state.name,
        "blend_names": list(state.blend.names),
        "blend_gate": state.blend.gate,
        "blend_scale": [float(value) for value in state.blend.scale],
        "grade_pen": state.blend.grade_pen,
        "tilt_norm": {"mean_db_per_oct": state.tilt_mean, "sd": state.tilt_sd},
        "output": {
            "format": VOICEPACK_FORMAT,
            "shape": [ROWS, 1, STYLE],
            "dtype": "float32",
        },
        "source": pinned_sources(spec),
        "license": {
            "declared": ARTIFACT_LICENSE,
            "components": LICENSE_COMPONENTS,
        },
    }


def write_tuner_metadata(
    state: TunerState, out_dir: Path, spec: Mapping[str, Any]
) -> tuple[Path, Path]:
    npz_path = out_dir / TUNER_NPZ_FILENAME
    json_path = out_dir / TUNER_JSON_FILENAME
    arrays = tuner_metadata_arrays(state)
    _require(set(arrays) == set(TUNER_NPZ_FIELDS), "Tuner metadata fields drifted")
    with npz_path.open("wb") as file:
        np.savez(file, **arrays)
    with np.load(npz_path, allow_pickle=False) as archive:
        _require(
            set(archive.files) == set(TUNER_NPZ_FIELDS),
            "Tuner metadata archive fields drifted",
        )
    json_path.write_text(
        json.dumps(tuner_metadata_document(state, spec), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return npz_path, json_path


def enroller_metadata() -> dict[str, Any]:
    return {
        "id": ENROLLER_ID,
        "kind": ENROLLER_KIND,
        "input": "reference-audio",
        "transcript_required": False,
        "min_seconds": 3.0,
        "recommended_seconds": 5.0,
        "max_seconds": 30.0,
        "output": {
            "format": VOICEPACK_FORMAT,
            "shape": [ROWS, 1, STYLE],
            "dtype": "float32",
        },
        "model_component": "inno_voicepack",
        "metadata_component": "inno_tuner",
    }


def build_filenames() -> dict[str, str]:
    return {
        "model": GRAPH_FILENAME,
        "inno_tuner": TUNER_NPZ_FILENAME,
        "inno_tuner_config": TUNER_JSON_FILENAME,
        "parity_fixtures": PARITY_FIXTURES_FILENAME,
        "parity_report": PARITY_REPORT_FILENAME,
        "bundle": BUNDLE_FILENAME,
    }


# --------------------------------------------------------------------------
# Export (build environment only)
# --------------------------------------------------------------------------


def array_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float((left * right).sum() / denominator) if denominator else 1.0

    flat_expected = expected.reshape(len(expected), -1)
    flat_actual = actual.reshape(len(actual), -1)
    midpoint = flat_expected.shape[1] // 2
    return {
        "shape": list(actual.shape),
        "dtype_ok": expected.shape == actual.shape,
        "all_finite": bool(np.isfinite(actual).all()),
        "max_abs_error": float(np.abs(expected - actual).max()),
        "mean_abs_error": float(np.abs(expected - actual).mean()),
        "cosine_timbre_half": _cosine(
            flat_expected[:, :midpoint], flat_actual[:, :midpoint]
        ),
        "cosine_predictor_half": _cosine(
            flat_expected[:, midpoint:], flat_actual[:, midpoint:]
        ),
    }


def synthetic_reference_clips() -> list[tuple[str, np.ndarray]]:
    """Deterministic synthetic references for build-time parity fixtures."""
    tone = np.arange(SR_ENC * 12) / SR_ENC
    harmonic = sum(
        np.sin(2 * np.pi * 130 * index * tone) / index for index in range(1, 30)
    ) * (0.6 + 0.4 * np.cos(2 * np.pi * 3 * tone))
    rng = np.random.default_rng(0)
    noise_t = np.arange(SR_ENC * 8) / SR_ENC
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.cos(2 * np.pi * 2 * noise_t))
    return [
        ("synthetic_harmonic_12s", harmonic.astype(np.float32)),
        ("synthetic_modulated_noise_8s", (envelope * rng.standard_normal(len(noise_t))).astype(np.float32)),
    ]


def build_voicepack_module(state: TunerState) -> Any:
    """Reconstruct SpeakerEncoder + style head + blend/prosody path in torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class Block(nn.Module):
        def __init__(self, cin: int, channels: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(cin, channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.shortcut = nn.Sequential()
            if stride != 1 or cin != channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(cin, channels, 1, stride, bias=False),
                    nn.BatchNorm2d(channels),
                )

        def forward(self, x: Any) -> Any:
            out = functional.relu(self.bn1(self.conv1(x)))
            return functional.relu(self.bn2(self.conv2(out)) + self.shortcut(x))

    class ResNet34(nn.Module):
        def __init__(self, width: int = 32, embed_dim: int = 256) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, width, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(width)
            cin = width
            for index, (channels, stride) in enumerate(
                [(width, 1), (width * 2, 2), (width * 4, 2), (width * 8, 2)]
            ):
                blocks = []
                for block_stride in [stride] + [1] * ([3, 4, 6, 3][index] - 1):
                    blocks.append(Block(cin, channels, block_stride))
                    cin = channels
                setattr(self, f"layer{index + 1}", nn.Sequential(*blocks))
            self.seg_1 = nn.Linear(NMEL // 8 * width * 8 * 2, embed_dim)

        def forward(self, features: Any) -> Any:
            x = features.permute(0, 2, 1)[:, None]
            out = functional.relu(self.bn1(self.conv1(x)))
            for index in range(4):
                out = getattr(self, f"layer{index + 1}")(out)
            pooled = torch.cat(
                [
                    out.mean(-1).flatten(1),
                    torch.sqrt(out.var(-1) + 1e-7).flatten(1),
                ],
                1,
            )
            return self.seg_1(pooled)

    class VoicePackTuner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = ResNet34()
            self.proj = nn.Sequential(
                nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 512)
            )
            self.style = nn.Sequential(
                nn.Linear(512, 512), nn.GELU(), nn.Linear(512, STYLE)
            )
            self.register_buffer("tilt_dir", torch.from_numpy(state.tilt_dir.copy()))
            self.register_buffer("blend_rows", torch.from_numpy(state.blend_rows.copy()))
            self.register_buffer("head_weight", torch.from_numpy(state.head_weight.copy()))
            self.register_buffer("head_mu", torch.from_numpy(state.head_mu.copy()))
            self.register_buffer("head_sd", torch.from_numpy(state.head_sd.copy()))
            self.register_buffer(
                "tilt_norm", torch.tensor([state.tilt_mean, state.tilt_sd])
            )

        def forward(self, fbank: Any, tilt: Any, head_stats: Any, blend_weights: Any) -> Any:
            embedding = functional.normalize(self.proj(self.encoder(fbank)), dim=-1)
            z_tilt = (tilt - self.tilt_norm[0]) / self.tilt_norm[1]
            style = self.style(embedding) + self.tilt_dir * z_tilt[:, None]
            z_head = (head_stats - self.head_mu) / self.head_sd
            delta = torch.cat([z_head, torch.ones_like(z_head[:, :1])], 1) @ self.head_weight
            predictor = torch.einsum("bv,vhp->bhp", blend_weights, self.blend_rows)
            predictor = predictor + delta[:, None, :]
            timbre = style[:, None, :HALF].expand(-1, ROWS, -1)
            return torch.cat([timbre, predictor], -1).reshape(ROWS, 1, STYLE)

    module = VoicePackTuner()
    encoder_weights = {
        name.removeprefix("backbone."): torch.from_numpy(value.copy())
        for name, value in state.enc.items()
        if name.startswith("backbone.")
    }
    projection_weights = {
        name.removeprefix("proj."): torch.from_numpy(value.copy())
        for name, value in state.enc.items()
        if name.startswith("proj.")
    }
    _require(
        len(encoder_weights) + len(projection_weights) == len(state.enc),
        "SpeakerEncoder keys must live under enc.backbone.* or enc.proj.*",
    )
    missing, unexpected = module.encoder.load_state_dict(encoder_weights, strict=False)
    _require(
        not unexpected
        and all(key.endswith("num_batches_tracked") for key in missing),
        f"SpeakerEncoder reconstruction mismatch: missing {sorted(missing)}, "
        f"unexpected {sorted(unexpected)}",
    )
    module.proj.load_state_dict(projection_weights, strict=True)
    module.style.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in state.style.items()},
        strict=True,
    )
    return module.eval().requires_grad_(False)


def export_bundle(
    spec: Mapping[str, Any],
    target: Path,
    *,
    sources: Mapping[str, Path],
    opset: int = 17,
    run_checker: bool = True,
) -> dict[str, Any]:
    """Export the tuner ONNX graph, tuner metadata and parity fixtures."""
    import torch

    target.mkdir(parents=True, exist_ok=True)
    state = load_tuner_state(sources["model.safetensors"])
    module = build_voicepack_module(state)
    fixtures: dict[str, np.ndarray] = {}
    expected_packs = []
    inputs = []
    for name, wav in synthetic_reference_clips():
        features = reference_features(wav, SR_ENC, state)
        for field, value in features.items():
            fixtures[f"{name}.{field}"] = value
        inputs.append(features)
        with torch.no_grad():
            pack = module(
                torch.from_numpy(features["fbank"]),
                torch.from_numpy(features["tilt"]),
                torch.from_numpy(features["head_stats"]),
                torch.from_numpy(features["blend_weights"]),
            )
        expected = pack.numpy().astype(np.float32)
        _require(
            expected.shape == (ROWS, 1, STYLE) and np.isfinite(expected).all(),
            f"Synthetic fixture {name} produced an invalid voicepack",
        )
        expected_packs.append(expected)
        fixtures[f"{name}.voicepack"] = expected

    graph_path = target / GRAPH_FILENAME
    example = tuple(
        torch.from_numpy(inputs[0][field]) for field in GRAPH_INPUTS
    )
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            str(graph_path),
            input_names=list(GRAPH_INPUTS),
            output_names=[GRAPH_OUTPUT],
            dynamic_axes={"fbank": {1: "frames"}},
            opset_version=opset,
            dynamo=False,
        )

    parity_cases = []
    if run_checker:
        import onnx
        import onnxruntime as ort

        model = onnx.load(str(graph_path))
        onnx.checker.check_model(model)
        session = ort.InferenceSession(
            str(graph_path), providers=["CPUExecutionProvider"]
        )
        for (name, _), features, expected in zip(
            synthetic_reference_clips(), inputs, expected_packs
        ):
            (actual,) = session.run(
                [GRAPH_OUTPUT], {field: features[field] for field in GRAPH_INPUTS}
            )
            parity_cases.append(
                {"case": name, **array_metrics(expected, actual.astype(np.float32))}
            )
            fixtures[f"{name}.voicepack_onnx"] = actual.astype(np.float32)

    write_tuner_metadata(state, target, spec)
    fixtures_path = target / PARITY_FIXTURES_FILENAME
    with fixtures_path.open("wb") as file:
        np.savez(file, **fixtures)
    report = {
        "schema": 1,
        "opset": opset,
        "inputs": {
            "fbank": "float32 [1, frames, 80]",
            "tilt": "float32 [1]",
            "head_stats": "float32 [1, 2]",
            "blend_weights": "float32 [1, packs]",
        },
        "output": {"voicepack": f"float32 [{ROWS}, 1, {STYLE}]"},
        "cases": parity_cases,
        "fixtures": PARITY_FIXTURES_FILENAME,
    }
    (target / PARITY_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "tuner": {
            "id": ENROLLER_ID,
            "version": state.version,
            "blend_names": list(state.blend.names),
        },
        "exporter": {
            "graph": GRAPH_FILENAME,
            "opset": opset,
            "inputs": list(GRAPH_INPUTS),
            "outputs": [GRAPH_OUTPUT],
            "torch_version": torch.__version__,
            "parity_cases": len(parity_cases),
        },
    }


def build_inno_tuner(
    spec: Mapping[str, Any],
    out_dir: Path,
    *,
    sources: Mapping[str, Path],
    opset: int = 17,
    run_checker: bool = True,
) -> Path:
    """Build the Inno tuner augmentation artifacts into `out_dir`."""
    validate_source_pins(spec)
    verify_sources(spec, sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = export_bundle(
        spec, out_dir, sources=sources, opset=opset, run_checker=run_checker
    )
    leaked = sorted(
        path.name
        for path in out_dir.iterdir()
        if path.suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
    )
    _require(not leaked, "PyTorch artifacts leaked into the tuner build: " + ", ".join(leaked))

    components = []
    for asset in spec["assets"]:
        path = out_dir / str(asset["source"])
        _require(path.is_file(), f"Missing tuner build artifact: {path.name}")
        components.append(
            {
                "name": str(asset["name"]),
                "role": str(asset["role"]),
                "component": str(asset["component"]),
                "format": str(asset["format"]),
                "size": path.stat().st_size,
                "sha256": sha256(path),
                **({"quality": str(asset["quality"])} if asset.get("quality") else {}),
            }
        )
    bundle = {
        "schema": 1,
        "profile": str(spec["id"]),
        "kind": str(spec["kind"]),
        "version": str(spec["declared_version"]),
        "voice_mode": "static",
        "speakers": [],
        "components": components,
        "source_artifacts": pinned_sources(spec),
        "source_files": source_checksums(sources),
        "license": {
            "declared": ARTIFACT_LICENSE,
            "components": LICENSE_COMPONENTS,
        },
        "tuner": result["tuner"],
        "exporter": result["exporter"],
    }
    (out_dir / BUNDLE_FILENAME).write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/inno-tuner-v0.2"))
    parser.add_argument("--cache", type=Path, default=Path("build/.cache-inno-tuner"))
    parser.add_argument("--releases", type=Path, default=RELEASES)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-checker", action="store_true")
    args = parser.parse_args(argv)

    spec = load_augmentation(args.releases)
    validate_source_pins(spec)
    sources = resolve_sources(spec, args.cache)
    out = build_inno_tuner(
        spec,
        args.out,
        sources=sources,
        opset=args.opset,
        run_checker=not args.no_checker,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
