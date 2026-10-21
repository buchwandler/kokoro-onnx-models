"""Prove AkinVox reference enrollment and synthesis through PyKokoro's public API.

The deterministic synthetic reference checks API wiring and runtime contracts only;
it is not evidence of subjective voice quality.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import TypedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE_KEY = "en-akinvox-cloning-v1"
DEFAULT_OUTPUT = ROOT / ".local-test" / "smoke" / PROFILE_KEY
SAMPLE_RATE = 24_000

COMPONENT_KEYS = (
    "reference_wavlm",
    "reference_encoders",
    "reference_mapper",
    "prosody",
    "curves",
    "decoder",
)

RELEASE_FILENAMES = {
    "reference_wavlm": "kokoro-akinvox-cloning-reference-wavlm-v1.0.1.onnx",
    "reference_encoders": "kokoro-akinvox-cloning-reference-encoders-v1.0.1.onnx",
    "reference_mapper": "kokoro-akinvox-cloning-reference-mapper-v1.0.1.onnx",
    "prosody": "kokoro-akinvox-cloning-prosody-v1.0.1.onnx",
    "curves": "kokoro-akinvox-cloning-curves-v1.0.1.onnx",
    "decoder": "kokoro-akinvox-cloning-decoder-v1.0.1.onnx",
    "source_params": "kokoro-akinvox-cloning-source-params-v1.0.1.npz",
    "config": "kokoro-akinvox-cloning-config-v1.0.1.json",
}

REFERENCE_TEXT = "I am speaking in a calm and steady voice."
TARGET_TEXTS = {
    "short": "Hello there.",
    "long": (
        "This is a longer test sentence for reference-conditioned speech synthesis. "
        "It has more words and enough phonemes to produce a longer waveform."
    ),
}


def _synthetic_reference(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = round(seconds * SAMPLE_RATE)
    time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    envelope = 0.25 + 0.2 * np.sin(2.0 * np.pi * 1.7 * time)
    wave = envelope * np.sin(2.0 * np.pi * 155.0 * time)
    wave += 0.02 * rng.standard_normal(count)
    return np.asarray(wave, dtype=np.float32)


def _candidate_artifacts(
    asset_dir: Path, *, strict_release_format: bool
) -> dict[str, str]:
    if strict_release_format:
        filenames = RELEASE_FILENAMES
    else:
        filenames = {
            **{component: f"{component}.onnx" for component in COMPONENT_KEYS},
            "source_params": "source-params.npz",
            "config": "config.json",
        }

    paths = {name: asset_dir / filename for name, filename in filenames.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing candidate artifact(s): " + ", ".join(missing))

    return {name: str(path) for name, path in paths.items()}


class AudioSummary(TypedDict):
    sample_rate: int
    audio_samples: int
    dtype: str


def _audio_summary(result, name: str) -> AudioSummary:
    if int(result.sample_rate) != SAMPLE_RATE:
        raise SystemExit(
            f"{name}: expected {SAMPLE_RATE} Hz audio, got {result.sample_rate}"
        )

    audio = np.asarray(result.audio)
    if audio.size == 0:
        raise SystemExit(f"{name}: PyKokoro returned empty audio")
    if not np.issubdtype(audio.dtype, np.floating) or not np.all(np.isfinite(audio)):
        raise SystemExit(
            f"{name}: audio must contain only finite floating-point samples"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        audio_fp32 = audio.astype(np.float32, copy=False)
    if not np.all(np.isfinite(audio_fp32)):
        raise SystemExit(f"{name}: audio cannot be represented as finite FP32 samples")

    return {
        "sample_rate": int(result.sample_rate),
        "audio_samples": int(audio_fp32.size),
        "dtype": str(audio_fp32.dtype),
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--reference-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict-release-format",
        action="store_true",
        help="Require all candidate assets to use their published release filenames",
    )
    args = parser.parse_args(argv)

    reference_seconds = float(args.reference_seconds)
    if not 3.0 <= reference_seconds <= 30.0:
        raise SystemExit("--reference-seconds must be between 3 and 30")

    artifacts = _candidate_artifacts(
        args.asset_dir,
        strict_release_format=args.strict_release_format,
    )

    from pykokoro import (
        GenerationConfig,
        KokoroSynthesizer,
        ReferenceVoice,
        SynthesisConfig,
    )

    config = SynthesisConfig(
        model_variant=PROFILE_KEY,
        generation=GenerationConfig(lang="en-us", speed=1.0),
        model_artifacts=artifacts,
        provider="cpu",
    )
    reference_audio = _synthetic_reference(reference_seconds, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    with KokoroSynthesizer(config) as synthesizer:
        reference_voice = synthesizer.enroll_voice(
            reference_audio,
            REFERENCE_TEXT,
            engine="akinvox",
            sample_rate=SAMPLE_RATE,
            language="en-us",
            name="akinvox-smoke",
        )
        if not isinstance(reference_voice, ReferenceVoice):
            raise SystemExit("AkinVox enrollment did not return a ReferenceVoice")
        if reference_voice.model_id != PROFILE_KEY:
            raise SystemExit(
                f"ReferenceVoice belongs to {reference_voice.model_id!r}, expected {PROFILE_KEY!r}"
            )

        results = {
            name: _audio_summary(
                synthesizer.synthesize_text(
                    text, language="en-us", voice=reference_voice
                ),
                name,
            )
            for name, text in TARGET_TEXTS.items()
        }
        if results["long"]["audio_samples"] <= results["short"]["audio_samples"]:
            raise SystemExit(
                "long target did not produce more audio than the short target"
            )

        with tempfile.TemporaryDirectory(prefix="akinvox-reference-state-") as temp_dir:
            state_path = Path(temp_dir) / "reference-voice.npz"
            reference_voice.save(state_path)
            restored_voice = ReferenceVoice.load(state_path)
            if restored_voice.model_id != PROFILE_KEY:
                raise SystemExit("reloaded ReferenceVoice belongs to the wrong model")
            results["reloaded_short"] = _audio_summary(
                synthesizer.synthesize_text(
                    TARGET_TEXTS["short"], language="en-us", voice=restored_voice
                ),
                "reloaded_short",
            )

    report = {
        "profile": PROFILE_KEY,
        "voice_mode": "reference",
        "consumer_api": "pykokoro.KokoroSynthesizer.enroll_voice+synthesize_text",
        "client_readiness": "proven",
        "reference_voice_model_id": PROFILE_KEY,
        "reference_sample_rate": SAMPLE_RATE,
        "sample_rates": {name: item["sample_rate"] for name, item in results.items()},
        "audio_samples": {
            name: item["audio_samples"] for name, item in results.items()
        },
        "audio_dtypes": {name: item["dtype"] for name, item in results.items()},
        "reference_seconds": reference_seconds,
        "seed": args.seed,
        "status": "pass",
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"short: {results['short']}")
    print(f"long: {results['long']}")
    print(f"reloaded_short: {results['reloaded_short']}")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
