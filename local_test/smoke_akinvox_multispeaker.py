#!/usr/bin/env python3
"""Run every AKinVox voice through its pinned language-specific ONNX graph."""

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

from scripts import build_kokoro, runtime_contracts

PROFILE_KEY = "akinvox-multispeaker-v1"
DEFAULT_OUTPUT = ROOT / ".local-test" / "smoke" / PROFILE_KEY
LANGUAGE_PREFIXES = {"en": "english-", "de": "german-"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return data


def _artifact_paths(
    profile: dict[str, Any], asset_dir: Path, *, strict_release_format: bool
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    release = profile["release"]
    model_assets = release["model_assets"]
    if {asset["component"] for asset in model_assets} != {"model_en", "model_de"}:
        raise SystemExit("Profile release must declare exactly model_en and model_de")
    models = {
        asset["component"]: asset_dir
        / (asset["filename"] if strict_release_format else asset["source"])
        for asset in model_assets
    }
    voice_asset = release["voice_assets"][0]
    voices = asset_dir / (
        voice_asset["filename"] if strict_release_format else voice_asset["source"]
    )
    missing = [path for path in [*models.values(), voices] if not path.is_file()]
    if missing:
        raise SystemExit("Missing runtime asset(s): " + ", ".join(map(str, missing)))

    if strict_release_format:
        manifest = _read_json(asset_dir / "release-manifest.json")
        runtime = manifest.get("runtime")
        contract = manifest.get("onnx_contract")
        components = [
            asset.get("component")
            for asset in manifest.get("assets", [])
            if asset.get("role") == "model"
        ]
        runtime_contracts.validate_component_set(
            runtime.get("layout") if isinstance(runtime, dict) else None,
            contract.get("components") if isinstance(contract, dict) else None,
            components,
        )
        if manifest.get("profile") != PROFILE_KEY:
            raise SystemExit(
                "Release manifest profile does not match the requested target"
            )
    else:
        config = _read_json(asset_dir / "config.json")
        runtime = config.get("runtime")
        contract = config.get("onnx_contract")
        components = list(models)
        if config.get("model_id") != PROFILE_KEY:
            raise SystemExit(
                "Build config model_id does not match the requested target"
            )

    if not isinstance(runtime, dict) or not isinstance(contract, dict):
        raise SystemExit("Runtime metadata or ONNX contract is missing")
    try:
        runtime_contracts.validate_language_router_constraints(runtime)
        runtime_contracts.validate_component_set(
            runtime.get("layout"), contract.get("components"), components
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return models, voices, {"runtime": runtime, "contract": contract}


def _feed_for_voice(
    session: Any, case: dict[str, Any], voice: np.ndarray
) -> dict[str, np.ndarray]:
    token_names = {"tokens", "input_ids"}
    style_names = {"style", "ref_s"}
    speed_names = {"speed"}
    inputs = {value.name for value in session.get_inputs()}
    aliases = {
        "tokens": token_names,
        "style": style_names,
        "speed": speed_names,
    }
    feed: dict[str, np.ndarray] = {}
    selected: dict[str, str] = {}
    for semantic, possible_names in aliases.items():
        matches = sorted(inputs & possible_names)
        if len(matches) != 1:
            raise SystemExit(f"ONNX graph must expose one {semantic} input alias")
        selected[semantic] = matches[0]

    tokens = np.asarray(case["tokens"], dtype=np.int64)
    expected_tokens = np.asarray([0, *tokens[1:-1], 0], dtype=np.int64)
    if tokens.size < 2 or not np.array_equal(tokens, expected_tokens):
        raise SystemExit(f"Invalid boundary-token fixture {case.get('name')!r}")
    phonemes = str(case["phonemes"])[:510]
    if not phonemes:
        raise SystemExit("Smoke fixture has no phonemes")
    style_index = min(len(phonemes), 510) - 1
    if style_index >= voice.shape[0]:
        raise SystemExit("Voice archive cannot supply the fixture's style row")
    feed[selected["tokens"]] = tokens[None, :]
    feed[selected["style"]] = np.ascontiguousarray(voice[style_index], dtype=np.float32)
    feed[selected["speed"]] = np.asarray(
        [float(case.get("speed", 1.0))], dtype=np.float32
    )
    return feed


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument(
        "--profiles", type=Path, default=ROOT / "scripts/kokoro_profiles.json"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-release-format", action="store_true")
    args = parser.parse_args(argv)

    profiles = build_kokoro.load_profiles(args.profiles)
    try:
        profile = profiles[PROFILE_KEY]
    except KeyError as exc:
        raise SystemExit(f"Missing profile {PROFILE_KEY!r}") from exc
    models, voice_path, runtime_info = _artifact_paths(
        profile, args.asset_dir, strict_release_format=args.strict_release_format
    )
    runtime = runtime_info["runtime"]
    contract = runtime_info["contract"]
    expected_voices = list(profile["voices"]["items"])
    with np.load(voice_path, allow_pickle=False) as archive:
        if archive.files != expected_voices:
            raise SystemExit("Voice archive names/order differ from the pinned roster")
        voices = {name: archive[name] for name in archive.files}

    for name, voice in voices.items():
        if voice.shape != (510, 1, 256) or voice.dtype != np.dtype("float32"):
            raise SystemExit(f"{name} has the wrong shape or dtype")
        if not voice.flags.c_contiguous or not np.isfinite(voice).all():
            raise SystemExit(f"{name} is non-contiguous or contains non-finite values")
        first_row = voice[0].tobytes(order="C")
        if any(voice[index].tobytes(order="C") != first_row for index in range(1, 510)):
            raise SystemExit(f"{name} rows are not byte-identical")

    import onnx
    import onnxruntime as ort

    export_cases = profile["export_validation"]["cases"]
    expected_languages = {
        name: language
        for language in ("en", "de")
        for name in profile["model"]["languages"][language]["speaker_names"]
    }
    results: dict[str, Any] = {}
    for language, prefix in LANGUAGE_PREFIXES.items():
        component = f"model_{language}"
        model_path = models[component]
        graph = onnx.load(str(model_path), load_external_data=True)
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        output_types = {
            value.name: onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
            for value in graph.graph.output
        }
        if output_types != {"audio": "FLOAT", "duration": "INT64"}:
            raise SystemExit(
                f"{component} has unexpected graph outputs: {output_types}"
            )
        cases = [
            case
            for case in export_cases
            if str(case.get("name", "")).startswith(prefix)
            and float(case.get("speed", 1.0)) == 1.0
        ]
        if not cases:
            raise SystemExit(f"No speed-1.0 short smoke fixture exists for {language}")
        case = cases[0]
        voice_results: dict[str, Any] = {}
        for name in profile["model"]["languages"][language]["speaker_names"]:
            if expected_languages[name] != language:
                raise SystemExit(f"Voice {name} is assigned to the wrong language")
            audio, duration = session.run(
                ["audio", "duration"], _feed_for_voice(session, case, voices[name])
            )
            geometry = build_kokoro.validate_duration_audio_consistency(
                audio,
                duration,
                token_count=len(case["tokens"]),
                samples_per_frame=int(contract["timing"]["samples_per_frame"]),
            )
            audio_values = np.asarray(audio)
            if audio_values.size == 0 or not np.any(audio_values):
                raise SystemExit(f"{component}/{name} produced empty or silent audio")
            voice_results[name] = {
                "fixture": case["name"],
                "speed": float(case.get("speed", 1.0)),
                "audio_samples": int(audio_values.size),
                "audio_rms": float(
                    np.sqrt(np.mean(np.square(audio_values, dtype=np.float64)))
                ),
                "duration_frames": int(np.asarray(duration).sum()),
                "geometry": geometry,
            }
        results[language] = {
            "component": component,
            "onnx_checker": "passed",
            "voice_count": len(voice_results),
            "voices": voice_results,
        }

    forbidden_suffixes = {".pt", ".pth", ".py", ".whl"}
    forbidden = [
        path.name
        for path in args.asset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if forbidden:
        raise SystemExit(
            f"Runtime asset directory contains forbidden build inputs: {forbidden}"
        )

    report = {
        "profile": PROFILE_KEY,
        "runtime_layout": runtime["layout"],
        "language_models": runtime["language_models"],
        "voice_count": len(expected_voices),
        "status": "pass",
        "languages": results,
        "forbidden_runtime_assets": [],
    }
    report_path = args.output / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
