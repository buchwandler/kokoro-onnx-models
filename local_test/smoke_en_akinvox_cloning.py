#!/usr/bin/env python3
"""Staged PyKokoro frontend and OnnxVoice gate for AkinVox cloning.

PyKokoro's prepared G2P frontend supplies real phone IDs, then OnnxVoice enrolls a
synthetic reference and synthesizes two target texts. This exercises the low-level
consumer path. PyKokoro's high-level synthesizer does not yet expose reference cloning,
so this gate does not claim full user-facing runtime readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROFILE_KEY = "en-akinvox-cloning-v1"
DEFAULT_OUTPUT = ROOT / ".local-test" / "smoke" / PROFILE_KEY

COMPONENT_KEYS = (
    "reference_wavlm",
    "reference_encoders",
    "reference_mapper",
    "prosody",
    "curves",
    "decoder",
)


REFERENCE_TEXT = "I am speaking in a calm and steady voice."
TARGET_TEXTS = {
    "short": "Hello there.",
    "long": (
        "This is a longer test sentence for reference-conditioned speech synthesis. "
        "It has more words and enough phonemes to produce a longer waveform."
    ),
}


def _load_runtime(asset_dir: Path, runtime_metadata: dict):
    from onnxvoice import OnnxVoice

    artifacts = {}
    for component in COMPONENT_KEYS:
        artifacts[f"model:{component}"] = str(asset_dir / f"{component}.onnx")
    artifacts["metadata:source_params"] = str(asset_dir / "source-params.npz")
    artifacts["config"] = str(asset_dir / "config.json")
    return OnnxVoice.open_local(
        system="kokoro",
        artifacts=artifacts,
        runtime=runtime_metadata,
        sample_rate=24000,
        provider="cpu",
    )


def _synthetic_reference(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = round(seconds * 24000)
    time = np.arange(count, dtype=np.float64) / 24000
    envelope = 0.25 + 0.2 * np.sin(2.0 * np.pi * 1.7 * time)
    wave = envelope * np.sin(2.0 * np.pi * 155.0 * time)
    wave += 0.02 * rng.standard_normal(count)
    return np.asarray(wave, dtype=np.float32)


def _resample_24k_to_16k(wave: np.ndarray) -> np.ndarray:
    count = round(len(wave) * 16000 / 24000)
    return np.interp(
        np.linspace(0, len(wave) - 1, count), np.arange(len(wave)), wave
    ).astype(np.float32)



def _phonemize(frontend, config, request_id: str, text: str) -> list[int]:
    from pykokoro import SynthesisSegment

    prepared = frontend.phonemize(
        SynthesisSegment(id=request_id, text=text, language="en-us"), config
    )
    tokens = list(prepared.token_ids)
    if not tokens:
        raise SystemExit(f"{request_id}: PyKokoro returned no token IDs")
    return tokens

def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--reference-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict-release-format",
        action="store_true",
        help="Require the released component filenames and reference-only metadata",
    )
    args = parser.parse_args(argv)

    asset_dir = args.asset_dir
    published = {
        "reference_wavlm": "kokoro-akinvox-cloning-reference-wavlm-v1.0.1.onnx",
        "reference_encoders": "kokoro-akinvox-cloning-reference-encoders-v1.0.1.onnx",
        "reference_mapper": "kokoro-akinvox-cloning-reference-mapper-v1.0.1.onnx",
        "prosody": "kokoro-akinvox-cloning-prosody-v1.0.1.onnx",
        "curves": "kokoro-akinvox-cloning-curves-v1.0.1.onnx",
        "decoder": "kokoro-akinvox-cloning-decoder-v1.0.1.onnx",
    }
    if args.strict_release_format:
        links = {}
        for component, name in published.items():
            source = asset_dir / name
            if not source.is_file():
                raise SystemExit(f"Missing released component: {source}")
            links[f"model:{component}"] = str(source)
        links["metadata:source_params"] = str(
            asset_dir / "kokoro-akinvox-cloning-source-params-v1.0.1.npz"
        )
        links["config"] = str(asset_dir / "kokoro-akinvox-cloning-config-v1.0.1.json")
    else:
        links = None


    from pykokoro import GenerationConfig, SynthesisConfig, TokenizerConfig
    from pykokoro.prepared_g2p import PreparedG2PAdapter

    frontend = PreparedG2PAdapter()
    frontend_config = SynthesisConfig(
        model_variant="v1.0",
        generation=GenerationConfig(lang="en-us"),
        tokenizer_config=TokenizerConfig(use_spacy=False),
    )
    reference_tokens = _phonemize(frontend, frontend_config, "reference", REFERENCE_TEXT)
    targets = {
        name: _phonemize(frontend, frontend_config, name, text)
        for name, text in TARGET_TEXTS.items()
    }
    runtime_metadata = {
        "layout": "cloning-onnx-v1",
        "voice_mode": "reference",
        "max_tokens": 510,
        "speed_supported": False,
        "style_dimensions": {"acoustic": 128, "duration": 128},
        "reference": {
            "format": "akinvox-cloning-reference-v1",
            "sample_rate": 24000,
            "identity_sample_rate": 16000,
            "min_seconds": 3.0,
            "max_seconds": 30.0,
            "memory_width": 192,
            "style_width": 256,
        },
    }
    if links is None:
        runtime = _load_runtime(asset_dir, runtime_metadata)
    else:
        from onnxvoice import OnnxVoice

        runtime = OnnxVoice.open_local(
            system="kokoro",
            artifacts=links,
            runtime=runtime_metadata,
            sample_rate=24000,
            provider="cpu",
        )

    wave24 = _synthetic_reference(args.reference_seconds, args.seed)
    wave16 = _resample_24k_to_16k(wave24)
    reference = runtime.prepare_reference(
        reference_tokens, audio_24k=wave24, audio_16k=wave16
    )
    results = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, tokens in targets.items():
        outcome = runtime.infer(tokens, reference=reference, speed=1.0, seed=args.seed)
        if not np.all(np.isfinite(outcome.audio)):
            raise SystemExit(f"{name}: non-finite audio")
        results[name] = {
            "tokens": len(tokens),
            "durations": int(np.sum(outcome.timings)),
            "audio_samples": int(np.asarray(outcome.audio).size),
            "sample_rate": int(outcome.sample_rate),
        }
        print(f"{name}: {results[name]}")

    if results["short"]["audio_samples"] >= results["long"]["audio_samples"]:
        raise SystemExit("long target did not produce more audio than the short target")

    report = {
        "profile": PROFILE_KEY,
        "voice_mode": "reference",
        "consumer_api": "onnxvoice.prepare_reference+infer",
        "client_readiness": "not_proven",
        "frontend": "pykokoro.PreparedG2PAdapter",
        "frontend_model": "v1.0",
        "language": "en-us",
        "reference_text": REFERENCE_TEXT,
        "target_texts": TARGET_TEXTS,
        "reference_seconds": args.reference_seconds,
        "seed": args.seed,
        "status": "pass",
        "synthesis": results,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output / "report.json")
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
