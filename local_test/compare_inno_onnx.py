#!/usr/bin/env python3
"""A/B gate: compare upstream Inno v0.2 enrollment with the ONNX tuner export.

For every reference clip (synthetic clips plus optional public WAVs) this
harness runs the pinned upstream v0.2 enrollment, computes the deterministic
host features, runs the exported ONNX graph, and compares the final voicepack
(feature parity, voicepack parity and optional synthesis A/B through the same
Kokoro v1.0 ONNX model via staged pykokoro). Only public or synthetic
references are used; no recordings are redistributed.

Upstream Inno v0.2 (PyTorch, safetensors, praat-parselmouth) is a build-time
oracle and is vendored at the pinned commit into the local cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import export_inno_tuner as inno

DEFAULT_BUILD_ROOT = ROOT / ".local-test" / "inno-build"
DEFAULT_OUTPUT_ROOT = ROOT / ".local-test" / "compare"
DEFAULT_CACHE_ROOT = ROOT / ".local-test" / "inno-cache"
TARGET_TEXT = (
    "It is a bright, calm morning, and the air is cool and clear. "
    "Down by the river, a few boats drift slowly past the old stone bridge."
)

# Plan tolerances for the ONNX voicepack versus upstream Inno v0.2. The feature
# gate is a drift bound, not a plan tolerance: upstream verifies its torch fbank
# only to 1e-4 against torchaudio Kaldi fbank, this NumPy port differs from
# float32 torch FFTs by up to ~1e-3 on log-mel features (which washes out to
# ~1e-7 at the pack), and 5e-3 still catches algorithmic errors like a wrong
# normalization axis, which show up orders of magnitude larger.
PACK_MAX_ABS_ERROR = 1e-4
PACK_MEAN_ABS_ERROR = 1e-5
PACK_HALF_COSINE = 0.9999
FEATURE_MAX_ABS_ERROR = 5e-3
SYNTH_CORRELATION = 0.9999
SYNTH_MAE = 1e-3
SYNTH_LEVEL_DELTA = 1e-3


def _load_upstream(source_dir: Path) -> Any:
    import importlib

    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return importlib.import_module("inno_kokoro.enroll")


def _reference_clips(extra: list[Path]) -> list[tuple[str, np.ndarray, int]]:
    clips = [
        (name, wav, inno.SR_ENC) for name, wav in inno.synthetic_reference_clips()
    ]
    for path in extra:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(-1)
        clips.append((path.stem, wav, sr))
    return clips


def _pack_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    metrics = inno.array_metrics(expected, actual)
    metrics["shape_exact"] = list(expected.shape) == list(actual.shape) == [510, 1, 256]
    metrics["dtype"] = str(np.asarray(actual).dtype)
    metrics["within_tolerance"] = (
        metrics["shape_exact"]
        and metrics["dtype_ok"]
        and metrics["all_finite"]
        and metrics["max_abs_error"] <= PACK_MAX_ABS_ERROR
        and metrics["mean_abs_error"] <= PACK_MEAN_ABS_ERROR
        and metrics["cosine_timbre_half"] >= PACK_HALF_COSINE
        and metrics["cosine_predictor_half"] >= PACK_HALF_COSINE
    )
    return metrics


def _waveform_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    size = min(len(expected), len(actual))
    left, right = expected[:size], actual[:size]
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    correlation = float((left * right).sum() / norm) if norm else 1.0
    return {
        "length_equal": len(expected) == len(actual),
        "lengths": [len(expected), len(actual)],
        "rms": [float(np.sqrt((left**2).mean())), float(np.sqrt((right**2).mean()))],
        "peak": [float(np.abs(left).max()), float(np.abs(right).max())],
        "rms_delta": float(
            abs(np.sqrt((left**2).mean()) - np.sqrt((right**2).mean()))
        ),
        "peak_delta": float(abs(np.abs(left).max() - np.abs(right).max())),
        "correlation": correlation,
        "mean_abs_error": float(np.abs(left - right).mean()),
    }


def _timing_signature(result: Any) -> tuple[list[float], str]:
    """Per-phoneme timing spans from a staged pykokoro result.

    pykokoro exposes phoneme_segments whose fields vary by version; numeric
    spans are read from `duration`/`onset`+`offset` and the total waveform
    length is the fallback timing signal.
    """
    spans: list[float] = []
    source = "waveform-length"
    for segment in getattr(result, "phoneme_segments", None) or []:
        if hasattr(segment, "duration"):
            spans.append(float(segment.duration))
            source = "phoneme_segments.duration"
        elif hasattr(segment, "onset") and hasattr(segment, "offset"):
            spans.append(float(segment.offset - segment.onset))
            source = "phoneme_segments.onset_offset"
    return spans, source


def _synthesis_pack_metrics(
    upstream_pack: np.ndarray,
    onnx_pack: np.ndarray,
    *,
    asset_dir: Path,
    text: str,
) -> dict[str, Any]:
    from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
    from pykokoro.tokenizer import TokenizerConfig

    model_path = asset_dir / "kokoro-v1.0.onnx"
    vocab_path = asset_dir / "vocab-v1.0.json"
    if not model_path.is_file() or not vocab_path.is_file():
        raise SystemExit(
            f"Staged v1.0 assets are missing from {asset_dir}; "
            "run local_test/prepare_local_assets.py v1.0 first"
        )

    archives = {}
    for label, pack in (("upstream", upstream_pack), ("onnx", onnx_pack)):
        path = asset_dir / f".compare-inno-{label}.npz"
        with path.open("wb") as file:
            np.savez(file, **{"inno_compare": pack.astype(np.float32)})
        archives[label] = path

    outputs = {}
    timings = {}
    for label, voices_path in archives.items():
        config = PipelineConfig(
            voice="inno_compare",
            model_path=model_path,
            voices_path=voices_path,
            model_config_path=vocab_path,
            model_source="github",
            model_variant="v1.0",
            model_quality="fp32",
            generation=GenerationConfig(lang="en-us", speed=1.0),
            return_trace=True,
        )
        with KokoroPipeline(config) as pipeline:
            result = pipeline.run(
                text,
                voice="inno_compare",
                generation=GenerationConfig(lang="en-us", speed=1.0),
                tokenizer_config=TokenizerConfig(),
            )
        outputs[label] = result.audio
        timings[label] = _timing_signature(result)

    waveform = _waveform_metrics(outputs["upstream"], outputs["onnx"])
    upstream_spans, timing_source = timings["upstream"]
    onnx_spans, _ = timings["onnx"]
    waveform["duration_timing_equal"] = upstream_spans == onnx_spans
    waveform["timing_source"] = timing_source
    waveform["within_tolerance"] = (
        waveform["length_equal"]
        and waveform["duration_timing_equal"]
        and waveform["correlation"] >= SYNTH_CORRELATION
        and waveform["mean_abs_error"] <= SYNTH_MAE
        and waveform["rms_delta"] <= SYNTH_LEVEL_DELTA
        and waveform["peak_delta"] <= SYNTH_LEVEL_DELTA
    )
    return waveform


def compare(
    *,
    build_root: Path,
    cache_root: Path,
    references: list[Path],
    synthesis_asset_dir: Path | None,
    text: str,
    run_synthesis: bool,
) -> dict[str, Any]:
    import onnxruntime as ort

    spec = inno.load_augmentation()
    inno.validate_source_pins(spec)
    sources = inno.resolve_sources(spec, cache_root)
    state = inno.load_tuner_state(sources["model.safetensors"])
    upstream = _load_upstream(inno.resolve_source_code(spec, cache_root))

    build_dir = build_root / "inno-tuner-v0.2"
    if not (build_dir / inno.GRAPH_FILENAME).is_file():
        inno.build_inno_tuner(spec, build_dir, sources=sources)
    session = ort.InferenceSession(
        str(build_dir / inno.GRAPH_FILENAME), providers=["CPUExecutionProvider"]
    )

    upstream_tuner = upstream.Tuner(str(sources["model.safetensors"]))
    upstream_encoder = upstream_tuner.encoder
    cases = []
    for name, wav, sr in _reference_clips(references):
        features = inno.reference_features(wav, sr, state)
        (onnx_pack,) = session.run(
            [inno.GRAPH_OUTPUT],
            {field: features[field] for field in inno.GRAPH_INPUTS},
        )

        import torch

        encoder_wav = inno.host_resample(wav[: inno.REF_MAX_S * sr], sr, inno.SR_ENC)
        with torch.no_grad():
            upstream_features = upstream_encoder.backbone.fbank(
                torch.from_numpy(encoder_wav)[None]
            ).numpy()[0]
        host_features = inno.host_fbank(encoder_wav)
        feature_error = float(
            np.abs(upstream_features - host_features).max()
        )
        upstream_pack, upstream_weights = upstream.enroll(
            torch.from_numpy(np.asarray(wav, dtype=np.float32)), sr, upstream_tuner
        )
        upstream_weights_dense = np.array(
            [[upstream_weights.get(name, 0.0) for name in state.blend.names]]
        )
        case = {
            "name": name,
            "sample_rate": sr,
            "seconds": round(len(wav) / sr, 3),
            "feature_parity": {
                "max_abs_error": feature_error,
                "within_tolerance": feature_error <= FEATURE_MAX_ABS_ERROR,
            },
            "blend_weights": {
                label: sorted(
                    (
                        (name, float(weight))
                        for name, weight in zip(state.blend.names, weights[0])
                        if weight > 0
                    ),
                    key=lambda item: -item[1],
                )
                for label, weights in (
                    ("upstream", upstream_weights_dense),
                    ("host", features["blend_weights"]),
                )
            },
            "voicepack": _pack_metrics(
                upstream_pack.numpy(), np.asarray(onnx_pack, dtype=np.float32)
            ),
        }
        if run_synthesis:
            if synthesis_asset_dir is None:
                raise SystemExit("--synthesis requires --asset-dir with staged v1.0 assets")
            case["synthesis"] = _synthesis_pack_metrics(
                upstream_pack.numpy(),
                np.asarray(onnx_pack, dtype=np.float32),
                asset_dir=synthesis_asset_dir,
                text=text,
            )
        cases.append(case)

    status = "pass"
    for case in cases:
        checks = [case["feature_parity"]["within_tolerance"], case["voicepack"]["within_tolerance"]]
        if "synthesis" in case:
            checks.append(case["synthesis"]["within_tolerance"])
        if not all(checks):
            status = "fail"
    return {
        "schema": 1,
        "status": status,
        "source_code": inno.pinned_sources(spec)["source_code"],
        "tolerances": {
            "pack_max_abs_error": PACK_MAX_ABS_ERROR,
            "pack_mean_abs_error": PACK_MEAN_ABS_ERROR,
            "pack_half_cosine": PACK_HALF_COSINE,
            "feature_max_abs_error": FEATURE_MAX_ABS_ERROR,
            "synth_correlation": SYNTH_CORRELATION,
            "synth_mean_abs_error": SYNTH_MAE,
            "synth_level_delta": SYNTH_LEVEL_DELTA,
        },
        "cases": cases,
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help="optional public reference WAV (repeatable)",
    )
    parser.add_argument(
        "--synthesis",
        action="store_true",
        help="also synthesize target text through the staged Kokoro v1.0 ONNX model",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / ".local-test" / "assets" / "v1.0",
        help="staged v1.0 asset directory for --synthesis",
    )
    parser.add_argument("--text", default=TARGET_TEXT)
    args = parser.parse_args(argv)

    report = compare(
        build_root=args.build_root,
        cache_root=args.cache_root,
        references=args.reference,
        synthesis_asset_dir=args.asset_dir,
        text=args.text,
        run_synthesis=args.synthesis,
    )
    output_dir = args.output_root / "en-inno-tuner-v0.2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for case in report["cases"]:
        pack = case["voicepack"]
        print(
            f"{case['name']}: feature {case['feature_parity']['max_abs_error']:.2e} "
            f"pack max {pack['max_abs_error']:.2e} mean {pack['mean_abs_error']:.2e} "
            f"cosine ({pack['cosine_timbre_half']:.6f}, {pack['cosine_predictor_half']:.6f})"
        )
    print(f"{report['status']}: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(run())
