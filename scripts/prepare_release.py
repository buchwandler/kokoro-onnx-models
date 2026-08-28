#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "scripts" / "kokoro_profiles.json"
TARGET_REPOSITORY = "buchwandler/kokoro-onnx-models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profiles() -> dict[str, dict[str, Any]]:
    return json.loads(PROFILES.read_text(encoding="utf-8"))


def _bundle_voices(path: Path, fallback: list[str]) -> list[str]:
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    speakers = bundle.get("speakers")
    if isinstance(speakers, list):
        names = [
            item["name"]
            for item in speakers
            if isinstance(item, dict) and item.get("name")
        ]
        if names:
            return names
    return fallback


def _asset_metadata(asset: dict[str, Any], path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "name": path.name,
        "role": str(asset["role"]),
        "format": str(asset["format"]),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }
    if asset.get("quality") is not None:
        metadata["quality"] = str(asset["quality"])
    if metadata["role"] == "voices" and metadata["format"] == "raw-float32-le":
        metadata["handling"] = {
            "dtype": "float32",
            "shape": [510, 256],
            "endianness": "little",
        }
    return metadata


def _validate_checkpoint_contract(
    profile: dict[str, Any],
    contract: dict[str, Any],
    bundle: dict[str, Any],
) -> None:
    if profile.get("model", {}).get("kind") != "checkpoint":
        return
    if not contract:
        raise SystemExit("Checkpoint profile is missing onnx_contract")
    outputs = contract.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("audio") != "float32":
        raise SystemExit("Checkpoint onnx_contract must declare audio output")
    timing = contract.get("timing")
    if not isinstance(timing, dict):
        raise SystemExit("Checkpoint onnx_contract must declare timing")
    if timing.get("kind") != "token-duration-v1":
        raise SystemExit("Checkpoint timing kind must be token-duration-v1")
    timing_output = timing.get("output")
    if outputs.get(timing_output) != "int64":
        raise SystemExit(
            "Checkpoint timing output must be declared as int64 in outputs"
        )
    if timing.get("unit") != "frame" or timing.get("samples_per_frame") != 600:
        raise SystemExit("Checkpoint timing must use 600-sample synthesis frames")
    if not isinstance(timing.get("includes_boundary_tokens"), bool):
        raise SystemExit("Checkpoint timing must declare boundary-token semantics")
    exporter = bundle.get("exporter") or {}
    exported_outputs = exporter.get("outputs")
    if not isinstance(exported_outputs, list) or timing_output not in exported_outputs:
        raise SystemExit(
            "Checkpoint exporter provenance does not expose the declared timing output"
        )
    exporter_timing = exporter.get("timing") or {}
    if exporter_timing.get("output") != timing_output:
        raise SystemExit("Checkpoint exporter timing output does not match contract")
    if exporter_timing.get("validated") is not True:
        raise SystemExit("Checkpoint exporter timing has not been validated")


def _write_checksums(out: Path, assets: list[dict[str, Any]]) -> None:
    lines = [f"{asset['sha256']}  {asset['name']}" for asset in assets]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_release_notes(out: Path, manifest: dict[str, Any]) -> None:
    runtime = manifest["runtime"]
    models = [
        asset.get("quality", "unknown")
        for asset in manifest["assets"]
        if asset["role"] == "model"
    ]
    notes = [
        f"# {manifest['profile']} {manifest['model_version']}",
        "",
        f"- Language(s): {', '.join(runtime['language_codes'])}",
        f"- Frontend: {runtime['frontend']}",
        f"- Model qualities: {', '.join(models)}",
        f"- Voices: {', '.join(runtime['voices'])}",
        f"- Source: {manifest['source']['repository']} @ {manifest['source']['revision']}",
        f"- License: {manifest['license']}",
        f"- SHA-256: recorded for {len(manifest['assets'])} assets",
    ]
    if manifest.get("builder"):
        notes.append(
            f"- Builder: {manifest['builder']['repository']} @ {manifest['builder']['commit']}"
        )
    (out / "release-notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _runtime_metadata(
    profile: dict[str, Any], bundle_path: Path, release: dict[str, Any]
) -> dict[str, Any]:
    frontend = profile.get("frontend") or {}
    fallback_voices = list(release.get("voices") or ["default"])
    contract = dict(profile.get("onnx_contract") or {})
    contract.setdefault(
        "inputs", {"tokens": "int64", "style": "float32", "speed": "float32"}
    )
    contract.setdefault("outputs", {"audio": "float32"})
    contract.setdefault("max_tokens", int(release.get("max_tokens", 510)))
    return {
        "language_codes": [str(profile.get("language", "und"))],
        "sample_rate": int(profile.get("sample_rate", 24000)),
        "frontend": str(
            profile.get("frontend_id") or frontend.get("name") or "pykokoro-native-v1"
        ),
        "frontend_experimental": bool(frontend.get("experimental", False)),
        "tokenizer_vocab_version": str(release.get("tokenizer_vocab_version", "1.0")),
        "vocabulary_source": str(release.get("vocabulary_source", "downloaded-config")),
        "max_tokens": int(contract["max_tokens"]),
        "default_voice": str(release.get("default_voice", fallback_voices[0])),
        "voices": _bundle_voices(bundle_path, fallback_voices),
        "layout": str(release.get("runtime_layout", "single-onnx-v1")),
        "postprocess": profile.get("postprocess", {}),
        "runtime_hints": profile.get("runtime_hints", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package a built profile into release assets"
    )
    parser.add_argument("profile")
    parser.add_argument("--build-root", type=Path, default=Path("build"))
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--allow-restricted", action="store_true")
    args = parser.parse_args()

    profiles = load_profiles()
    if args.profile not in profiles:
        raise SystemExit(f"Unknown profile: {args.profile}")
    profile = profiles[args.profile]
    release = profile.get("release") or {}
    if not release.get("enabled", False) and not args.allow_restricted:
        raise SystemExit(
            f"Release disabled for {args.profile}. Review MODEL_LICENSES.md and pass "
            "--allow-restricted only if redistribution is permitted."
        )

    src = args.build_root / args.profile
    voice_assets = release.get("voice_assets") or [
        {
            "source": "voices.bin",
            "filename": release["voices_filename"],
            "format": "unknown",
        }
    ]
    auxiliary_assets = release.get("auxiliary_assets") or []
    required = [src / "model.onnx", src / "bundle.json"]
    required += [src / str(asset["source"]) for asset in voice_assets]
    required += [src / str(asset["source"]) for asset in auxiliary_assets]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing build artifacts: " + ", ".join(missing))

    tag = str(release["tag"])
    out = args.dist / tag
    out.mkdir(parents=True, exist_ok=True)
    mapping: dict[Path, tuple[Path, dict[str, Any]]] = {
        src / "model.onnx": (
            out / str(release["model_filename"]),
            {"role": "model", "format": "onnx", "quality": "fp32"},
        ),
        src / "bundle.json": (
            out / "bundle.json",
            {"role": "bundle", "format": "json"},
        ),
    }
    for asset in voice_assets:
        mapping[src / str(asset["source"])] = (
            out / str(asset["filename"]),
            {"role": "voices", "format": str(asset.get("format", "unknown"))},
        )
    for asset in auxiliary_assets:
        mapping[src / str(asset["source"])] = (
            out / str(asset["filename"]),
            {
                "role": str(asset.get("role", "metadata")),
                "format": str(asset.get("format", "unknown")),
            },
        )
    if (src / "config.json").is_file() and release.get("config_filename"):
        mapping[src / "config.json"] = (
            out / str(release["config_filename"]),
            {"role": "config", "format": "json"},
        )

    for source, (target, _) in mapping.items():
        shutil.copy2(source, target)

    asset_metadata = []
    for source, (target, metadata) in sorted(
        mapping.items(), key=lambda item: item[1][0].name
    ):
        asset_metadata.append(_asset_metadata(metadata, target))

    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))

    contract = dict(profile.get("onnx_contract") or {})
    contract.setdefault(
        "inputs", {"tokens": "int64", "style": "float32", "speed": "float32"}
    )
    if profile.get("model", {}).get("kind") != "checkpoint":
        contract.setdefault("outputs", {"audio": "float32"})
    contract.setdefault("max_tokens", int(release.get("max_tokens", 510)))
    _validate_checkpoint_contract(profile, contract, bundle)
    manifest: dict[str, Any] = {
        "schema": 2,
        "runtime_contract": 1,
        "repository": TARGET_REPOSITORY,
        "tag": tag,
        "profile": args.profile,
        "model_version": str(
            release.get(
                "model_version", tag.rsplit("-v", 1)[-1] if "-v" in tag else tag
            )
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "type": str(profile.get("source_type", "huggingface")),
            "repository": str(profile["repo_id"]),
            "revision": str(profile.get("revision", "main")),
        },
        "license": str(profile["license"]),
        "publication": {"enabled": bool(release.get("enabled", False))},
        "runtime": _runtime_metadata(profile, out / "bundle.json", release),
        "onnx_contract": contract,
        "assets": asset_metadata,
        "provenance": {
            "source_artifacts": bundle.get("source_artifacts", {}),
            "exporter": bundle.get("exporter", {}),
        },
    }
    builder_commit = os.environ.get("GITHUB_SHA")
    if builder_commit:
        manifest["builder"] = {
            "repository": os.environ.get("GITHUB_REPOSITORY", TARGET_REPOSITORY),
            "commit": builder_commit,
        }
    (out / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_checksums(out, asset_metadata)
    _write_release_notes(out, manifest)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
