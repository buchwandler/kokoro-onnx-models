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
    return metadata


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
        "default_voice": fallback_voices[0],
        "voices": _bundle_voices(bundle_path, fallback_voices),
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

    contract = dict(profile.get("onnx_contract") or {})
    contract.setdefault(
        "inputs", {"tokens": "int64", "style": "float32", "speed": "float32"}
    )
    contract.setdefault("outputs", {"audio": "float32"})
    contract.setdefault("max_tokens", int(release.get("max_tokens", 510)))
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
