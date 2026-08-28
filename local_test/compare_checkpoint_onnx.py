#!/usr/bin/env python3
"""Compare a checkpoint-derived Kokoro model with its ONNX export."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_kokoro
from scripts.export_validation import compare_waveform_structure, waveform_metrics

DEFAULT_BUILD_ROOT = ROOT / ".local-test" / "compare-build"
DEFAULT_OUTPUT_ROOT = ROOT / ".local-test" / "compare"


def _safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise build_kokoro.BuildError(f"Invalid {label}: {value!r}")
    return value


def eligible_cases(
    profile_key: str,
    profile: Mapping[str, Any],
    requested: list[str] | None,
) -> list[Mapping[str, Any]]:
    model = profile.get("model") or {}
    if model.get("kind") != "checkpoint":
        raise build_kokoro.BuildError(
            f"Profile {profile_key!r} has no native checkpoint reference; "
            "only checkpoint profiles support direct comparison"
        )
    cases = list((profile.get("export_validation") or {}).get("cases") or [])
    if not cases:
        raise build_kokoro.BuildError(
            f"Profile {profile_key!r} has no frozen export_validation cases"
        )
    if requested is None:
        return cases
    by_name = {str(case.get("name", "case")): case for case in cases}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise build_kokoro.BuildError(
            f"Unknown comparison case(s) for {profile_key!r}: {', '.join(missing)}"
        )
    return [by_name[name] for name in requested]


def _duration_record(duration: np.ndarray) -> dict[str, Any]:
    values = np.asarray(duration).reshape(-1)
    return {
        "frames": [int(value) for value in values],
        "count": int(values.size),
        "total_frames": int(np.sum(values)),
    }


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    values = np.asarray(audio, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size == 0:
        raise build_kokoro.BuildError(f"Cannot write invalid audio to {path}")
    if not np.isfinite(values).all():
        raise build_kokoro.BuildError(f"Cannot write non-finite audio to {path}")
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def _health_kwargs(validation: Mapping[str, Any]) -> dict[str, float]:
    return build_kokoro._health_kwargs(validation)


def _validate_audio(
    audio: np.ndarray,
    duration: np.ndarray,
    *,
    token_count: int,
    sample_rate: int,
    validation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    timing = build_kokoro.validate_duration_audio_consistency(
        audio, duration, token_count=token_count
    )
    health = build_kokoro.validate_waveform_health(
        audio, sample_rate, **_health_kwargs(validation)
    )
    return timing, health


def _stage_record(
    audio: np.ndarray,
    duration: np.ndarray,
    *,
    native_audio: np.ndarray,
    token_count: int,
    sample_rate: int,
    validation: Mapping[str, Any],
    wav_path: Path | None,
    wav_root: Path,
) -> dict[str, Any]:
    timing, health = _validate_audio(
        audio,
        duration,
        token_count=token_count,
        sample_rate=sample_rate,
        validation=validation,
    )
    record: dict[str, Any] = {
        "duration": _duration_record(duration),
        "timing": timing,
        "metrics": waveform_metrics(audio, sample_rate),
        "health": health,
    }
    if wav_path is not None:
        _write_wav(wav_path, audio, sample_rate)
        record["wav"] = _relative_path(wav_path, wav_root)
    else:
        record["wav"] = None
    if audio is not native_audio:
        record["comparison_to_native"] = compare_waveform_structure(
            native_audio, audio, sample_rate
        )
    return record


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _report_markdown(report: Mapping[str, Any], output_dir: Path) -> str:
    lines = [
        f"# Native Torch versus ONNX comparison: {report['profile']}",
        "",
        f"Seed: `{report['seed']}`. ONNX runs per case: `{report['runs']}`.",
        "",
        "Listening status: `not-recorded`.",
        "",
    ]
    for case in report["cases"]:
        lines.extend([f"## Case: {case['name']}", ""])
        stages = [("Native", case["native"])]
        if case.get("patched") is not None:
            stages.append(("Patched", case["patched"]))
        for label, stage in stages:
            if stage.get("wav"):
                lines.append(f"{label}: `{output_dir / stage['wav']}`")
        for index, stage in enumerate(case["onnx"], 1):
            if stage.get("wav"):
                lines.append(f"ONNX {index}: `{output_dir / stage['wav']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def compare_profile(
    profile_key: str,
    profile: Mapping[str, Any],
    *,
    build_root: Path,
    output_dir: Path,
    requested_cases: list[str] | None,
    voice_name: str | None,
    runs: int,
    seed: int,
    provider: str,
    write_wav: bool,
) -> dict[str, Any]:
    if runs < 1:
        raise build_kokoro.BuildError("--runs must be at least 1")
    cases = eligible_cases(profile_key, profile, requested_cases)
    sample_rate = int(profile.get("sample_rate", 24000))
    validation = profile.get("export_validation") or {}
    build_dir = build_root / profile_key
    cache_dir = build_dir / ".hf"
    model_path = build_dir / "model.onnx"
    build_dir.mkdir(parents=True, exist_ok=True)

    voices = build_kokoro.resolve_voices(dict(profile), cache_dir)
    if voice_name is None:
        voice_name = str(
            (profile.get("release") or {}).get("default_voice") or next(iter(voices))
        )
    if voice_name not in voices:
        raise build_kokoro.BuildError(
            f"Unknown voice {voice_name!r}; available voices: {', '.join(voices)}"
        )
    voice = voices[voice_name]

    config_name = str(profile["model"].get("config") or "")
    if not config_name:
        raise build_kokoro.BuildError("Checkpoint comparison requires model.config")
    config_path = build_kokoro.hf_download(
        str(profile["repo_id"]),
        config_name,
        str(profile.get("revision", "main")),
        cache_dir,
    )
    build_kokoro.verify_source_hash(
        config_path,
        profile["model"].get("config_sha256"),
        label=f"model config {config_name}",
    )
    checkpoint_path = build_kokoro.hf_download(
        str(profile["repo_id"]),
        str(profile["model"]["path"]),
        str(profile.get("revision", "main")),
        cache_dir,
    )
    build_kokoro.verify_source_hash(
        checkpoint_path,
        profile["model"].get("sha256"),
        label=f"model checkpoint {profile['model']['path']}",
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    native_model = build_kokoro.load_checkpoint_native(checkpoint_path, config)
    from kokoro.model import KModelForONNX

    native_cases = build_kokoro.capture_native_reference_cases(
        KModelForONNX(native_model).eval(),
        cases,
        voice,
        max_phonemes=510,
        seed=seed,
    )

    export_provenance: dict[str, Any] = {}
    build_kokoro.resolve_model(
        dict(profile),
        cache_dir,
        model_path,
        opset=17,
        seq_len=64,
        voice=voice,
        export_provenance=export_provenance,
    )

    patched_model = build_kokoro.load_checkpoint_native(checkpoint_path, config)
    patch_metadata = build_kokoro.install_exact_onnx_istft(patched_model)
    patched_wrapper = KModelForONNX(patched_model).eval()
    patched_cases = build_kokoro.validate_patched_pytorch_against_native(
        patched_wrapper,
        native_cases,
        sample_rate=sample_rate,
        validation=validation,
        seed=seed,
    )["cases"]

    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=[provider])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_cases: list[dict[str, Any]] = []
    for native_case, patched_case, case in zip(native_cases, patched_cases, cases):
        case_name = _safe_component(str(case.get("name", "case")), "case name")
        case_dir = output_dir / case_name
        native_audio = np.asarray(native_case["audio"])
        native_duration = np.asarray(native_case["duration"])
        native_record = _stage_record(
            native_audio,
            native_duration,
            native_audio=native_audio,
            token_count=int(native_case["tokens"].shape[1]),
            sample_rate=sample_rate,
            validation=validation,
            wav_path=case_dir / "torch-native.wav" if write_wav else None,
            wav_root=output_dir,
        )
        patched_audio, patched_duration = build_kokoro.run_patched_pytorch_outputs(
            patched_wrapper, native_case, seed=seed
        )
        patched_record = _stage_record(
            patched_audio,
            patched_duration,
            native_audio=native_audio,
            token_count=int(native_case["tokens"].shape[1]),
            sample_rate=sample_rate,
            validation=validation,
            wav_path=case_dir / "torch-patched.wav" if write_wav else None,
            wav_root=output_dir,
        )
        patched_record["comparison_to_native"] = patched_case["waveform_structure"]

        onnx_records: list[dict[str, Any]] = []
        for index in range(1, runs + 1):
            audio, duration = build_kokoro.run_onnx_case(session, native_case)
            if audio.shape != native_audio.shape:
                raise build_kokoro.BuildError(
                    f"Audio shape mismatch for {case_name!r}: "
                    f"native {native_audio.shape}, ONNX {audio.shape}"
                )
            if not np.array_equal(duration, native_duration):
                raise build_kokoro.BuildError(
                    f"Duration parity failed for {case_name!r}"
                )
            onnx_record = _stage_record(
                audio,
                duration,
                native_audio=native_audio,
                token_count=int(native_case["tokens"].shape[1]),
                sample_rate=sample_rate,
                validation=validation,
                wav_path=case_dir / f"onnx-{index:02d}.wav" if write_wav else None,
                wav_root=output_dir,
            )
            onnx_records.append(onnx_record)
        report_cases.append(
            {
                "name": case_name,
                "tokens": int(native_case["tokens"].shape[1]),
                "speed": float(case.get("speed", 1.0)),
                "native": native_record,
                "patched": patched_record,
                "onnx": onnx_records,
            }
        )

    report: dict[str, Any] = {
        "schema": 1,
        "profile": profile_key,
        "source": {
            "repo_id": profile["repo_id"],
            "revision": profile.get("revision", "main"),
            "checkpoint": profile["model"]["path"],
            "voice": voice_name,
        },
        "reference": {"runtime": "torch", "native": True, "disable_complex": False},
        "patched": {"runtime": "torch", "export_specific": True, **patch_metadata},
        "onnx": {"runtime": "onnxruntime", "provider": provider},
        "sample_rate": sample_rate,
        "seed": seed,
        "runs": runs,
        "cases": report_cases,
        "listening": {"required": True, "status": "not-recorded"},
        "exporter": export_provenance,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        _report_markdown(report, output_dir), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native Torch and ONNX audio for a checkpoint profile"
    )
    parser.add_argument("profile", help="Checkpoint profile key")
    parser.add_argument("--profiles", type=Path, default=build_kokoro.PROFILE_FILE)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--voice")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--keep-build", action="store_true")
    parser.add_argument("--no-wav", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = build_kokoro.load_profiles(args.profiles)
    if args.profile not in profiles:
        raise build_kokoro.BuildError(f"Unknown profile: {args.profile}")
    profile = profiles[args.profile]
    validation = profile.get("export_validation") or {}
    seed = int(args.seed if args.seed is not None else validation.get("export_seed", 0))
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.profile
    try:
        report = compare_profile(
            args.profile,
            profile,
            build_root=args.build_root,
            output_dir=output_dir,
            requested_cases=args.cases,
            voice_name=args.voice,
            runs=args.runs,
            seed=seed,
            provider=args.provider,
            write_wav=not args.no_wav,
        )
    finally:
        if not args.keep_build:
            shutil.rmtree(args.build_root / args.profile, ignore_errors=True)
    for case in report["cases"]:
        print(f"Case {case['name']}:")
        for stage in [case["native"], case.get("patched"), *case["onnx"]]:
            if stage and stage.get("wav"):
                print(f"  {output_dir / stage['wav']}")
    print(f"Report: {output_dir / 'report.json'}")
    print(f"Summary: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except build_kokoro.BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
