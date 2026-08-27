#!/usr/bin/env python3
"""Generate or check a registry GitHub distribution from a release manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "catalog" / "models.json"
RELEASES = ROOT / "catalog" / "releases.json"
TARGET_REPOSITORY = "buchwandler/kokoro-onnx-models"


class RegistryReleaseError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryReleaseError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryReleaseError(f"{path} must contain an object")
    return value


def distribution_from_manifest(
    manifest: dict[str, Any], release: dict[str, Any], existing: dict[str, Any] | None = None
) -> dict[str, Any]:
    tag = str(manifest["tag"])
    old_artifacts = {item["id"]: item for item in (existing or {}).get("artifacts", [])}
    artifacts = []
    for item in manifest["assets"]:
        name = str(item["name"])
        artifact_id = f"{item['role']}-{Path(name).stem}"
        artifact = {
            "id": artifact_id,
            "role": "voice" if item["role"] == "voice" else item["role"],
            "url": f"https://github.com/{TARGET_REPOSITORY}/releases/download/{tag}/{name}",
            "local_name": name,
            "format": str(item["format"]),
            "size": int(item["size"]),
            "sha256": str(item["sha256"]),
        }
        for field in ("quality", "component"):
            if item.get(field) is not None:
                artifact[field] = item[field]
        if artifact_id in old_artifacts and old_artifacts[artifact_id].get("handling"):
            artifact["handling"] = old_artifacts[artifact_id]["handling"]
        artifacts.append(artifact)
    return {
        "id": f"github-{tag}",
        "provider": "github-release",
        "transport": "https",
        "release_key": str(manifest["profile"]),
        "release_tag": tag,
        "runtime_ready": True,
        "artifacts": artifacts,
    }


def sync_release(
    candidate: Path, *, profile: str | None, registry_path: Path, releases_path: Path, update: bool
) -> None:
    manifest = _load(candidate / "release-manifest.json")
    registry = _load(registry_path)
    releases = _load(releases_path)
    model_id = profile or str(manifest.get("profile", ""))
    if model_id not in registry.get("models", {}):
        raise RegistryReleaseError(f"Unknown registry profile: {model_id}")
    release = releases.get("releases", {}).get(model_id)
    if release is None:
        raise RegistryReleaseError(f"Profile {model_id} has no release catalog entry")
    if manifest.get("tag") != release.get("tag"):
        raise RegistryReleaseError(f"Manifest tag {manifest.get('tag')!r} does not match catalog tag {release.get('tag')!r}")
    model = registry["models"][model_id]
    existing = next((d for d in model["distributions"] if d.get("provider") == "github-release"), None)
    generated = distribution_from_manifest(manifest, release, existing)
    if update:
        model["distributions"] = [
            d for d in model["distributions"] if d.get("provider") != "github-release"
        ]
        model["distributions"].append(generated)
        model["runtime_available"] = True
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated {registry_path} for {model_id}")
    elif existing != generated:
        raise RegistryReleaseError(f"Registry GitHub distribution differs from {candidate / 'release-manifest.json'}")
    else:
        print(f"registry GitHub distribution matches {candidate}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync registry metadata from release-manifest.json")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--releases", type=Path, default=RELEASES)
    args = parser.parse_args(argv)
    try:
        sync_release(args.candidate, profile=args.profile, registry_path=args.registry, releases_path=args.releases, update=args.update)
    except RegistryReleaseError as exc:
        print(f"registry release update failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
