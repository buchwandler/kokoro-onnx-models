#!/usr/bin/env python3
"""Validate a published GitHub release and synchronize its registry metadata."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

if __package__:
    from .update_registry_from_release import RegistryReleaseError, sync_release
    from .verify_candidate import CandidateError, verify_candidate
    from .verify_model_registry import RegistryError, verify_registry
else:
    from update_registry_from_release import RegistryReleaseError, sync_release
    from verify_candidate import CandidateError, verify_candidate
    from verify_model_registry import RegistryError, verify_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "catalog" / "models.json"
RELEASES = ROOT / "catalog" / "releases.json"
TARGET_REPOSITORY = "buchwandler/kokoro-onnx-models"


class ReleaseSyncError(ValueError):
    """Raised when a published release cannot be synchronized."""


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseSyncError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseSyncError(f"{path} must contain an object")
    return value


def _release_url(repository: str, tag: str, name: str) -> str:
    return (
        f"https://github.com/{repository}/releases/download/"
        f"{quote(tag, safe='')}/{quote(name, safe='')}"
    )


def _download(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "kokoro-onnx-models-registry-sync"})
    partial = target.with_name(target.name + ".part")
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _download_release(
    candidate: Path, *, repository: str, tag: str, manifest: dict
) -> None:
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ReleaseSyncError("Release manifest has no assets")
    names = ["SHA256SUMS"]
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            raise ReleaseSyncError("Release manifest contains an invalid asset")
        names.append(asset["name"])
    if len(names) != len(set(names)):
        raise ReleaseSyncError("Release manifest contains duplicate asset names")
    for name in names:
        path = Path(name)
        if path.name != name:
            raise ReleaseSyncError(
                f"Release asset name is not a flat filename: {name!r}"
            )
        _download(_release_url(repository, tag, name), candidate / name)


def sync_published_release(
    release_key: str,
    *,
    registry_path: Path = REGISTRY,
    releases_path: Path = RELEASES,
    repository: str | None = None,
) -> None:
    releases = _load(releases_path)
    release_entries = releases.get("releases")
    if not isinstance(release_entries, dict) or release_key not in release_entries:
        raise ReleaseSyncError(f"Unknown release key: {release_key}")
    release = release_entries[release_key]
    tag = str(release["tag"])
    profile = str(release.get("profile", release_key))
    repository = repository or str(releases.get("target_repository", TARGET_REPOSITORY))

    with tempfile.TemporaryDirectory(prefix="kokoro-release-") as temporary:
        candidate = Path(temporary)
        manifest_path = candidate / "release-manifest.json"
        _download(_release_url(repository, tag, manifest_path.name), manifest_path)
        manifest = _load(manifest_path)
        _download_release(candidate, repository=repository, tag=tag, manifest=manifest)
        try:
            verify_candidate(
                candidate,
                expected_tag=tag,
                expected_profile=profile,
            )
        except (CandidateError, KeyError, TypeError) as exc:
            raise ReleaseSyncError(
                f"Published release failed validation: {exc}"
            ) from exc

        try:
            sync_release(
                candidate,
                profile=profile,
                registry_path=registry_path,
                releases_path=releases_path,
                update=True,
            )
            verify_registry(registry_path, releases_path=releases_path)
        except (RegistryError, RegistryReleaseError) as exc:
            raise ReleaseSyncError(f"Registry synchronization failed: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a published GitHub release and update catalog/models.json"
    )
    parser.add_argument("release_key", help="Entry from catalog/releases.json")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--releases", type=Path, default=RELEASES)
    parser.add_argument("--repository", help="GitHub repository override")
    args = parser.parse_args(argv)
    try:
        sync_published_release(
            args.release_key,
            registry_path=args.registry,
            releases_path=args.releases,
            repository=args.repository,
        )
    except (ReleaseSyncError, OSError, ValueError) as exc:
        print(f"release registry sync failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
