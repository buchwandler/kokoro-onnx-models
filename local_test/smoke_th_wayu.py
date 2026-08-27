#!/usr/bin/env python3
"""Run the local smoke test for the Thai Wayu split runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from runtimes.thai_wayu_split import SAMPLE_RATE, ThaiWayuRuntime

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / ".local-test" / "assets" / "th-wayu"
OUTPUT_DIR = ROOT / ".local-test" / "wav" / "th-wayu"
SENTENCE = "สวัสดี นี่คือการทดสอบเสียงแบบภายในก่อนเผยแพร่โมเดล"
VOICES = (
    "f_teen_bright",
    "f_young_bright",
    "f_young_clear",
    "f_young_warm",
    "f_mid_clear",
    "f_mid_warm",
    "f_elderly_soft",
    "f_elderly_low",
    "m_teen_bright",
    "m_young_clear",
    "m_mid_warm",
    "m_elderly_deep",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--voice", action="append", default=[])
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    runtime = ThaiWayuRuntime(args.asset_dir.resolve())
    selected = args.voice or list(VOICES)
    if set(runtime.voices) != set(VOICES):
        raise RuntimeError("Thai voicepack roster does not match the release roster")
    phonemes = runtime.phonemize(SENTENCE)
    unknown = sorted({symbol for symbol in phonemes if symbol not in runtime.vocab})
    if unknown:
        raise RuntimeError(f"Thai frontend emitted unsupported symbols: {unknown}")
    token_count = sum(symbol in runtime.vocab for symbol in phonemes)
    if token_count > runtime.max_tokens:
        raise RuntimeError(f"Thai frontend emitted {token_count} tokens")

    if not args.no_write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for voice in selected:
        audio = runtime.synthesize(SENTENCE, voice)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError(f"Thai synthesis failed for {voice}")
        if not args.no_write:
            sf.write(args.output_dir / f"{voice}.wav", audio, SAMPLE_RATE)
        print(f"{voice}: {audio.size} samples at {SAMPLE_RATE} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
