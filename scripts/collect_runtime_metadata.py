#!/usr/bin/env python3
"""Collect and check immutable runtime artifact metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "catalog" / "models.json"


class MetadataError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MetadataError(f"{path} must contain an object")
    return value


def _download(url: str, target: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    request = Request(url, headers={"User-Agent": "kokoro-onnx-models-metadata"})
    with urlopen(request, timeout=120) as response, target.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _validate_format(path: Path, artifact: dict[str, Any]) -> None:
    fmt = artifact["format"]
    if fmt == "json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetadataError(f"{artifact['id']}: invalid JSON: {exc}") from exc
    elif fmt == "numpy-npz":
        if not zipfile.is_zipfile(path):
            raise MetadataError(f"{artifact['id']}: expected a NumPy zip archive")
    elif fmt == "raw-float32-le":
        if path.stat().st_size % 4:
            raise MetadataError(
                f"{artifact['id']}: raw float32 file is not 4-byte aligned"
            )
        handling = artifact.get("handling") or {}
        shape = handling.get("shape")
        if shape and path.stat().st_size != shape[0] * shape[1] * 4:
            raise MetadataError(
                f"{artifact['id']}: raw voice size does not match declared shape"
            )
    elif fmt == "onnx":
        try:
            import onnx
        except ImportError:
            return
        try:
            onnx.checker.check_model(str(path))
        except Exception as exc:
            raise MetadataError(f"{artifact['id']}: invalid ONNX: {exc}") from exc


def _iter_artifacts(
    registry: dict[str, Any], profile: str | None, artifact_id: str | None
):
    models = registry.get("models", {})
    if profile and profile not in models:
        raise MetadataError(f"Unknown model profile: {profile}")
    selected = [profile] if profile else list(models)
    found = False
    for model_id in selected:
        for distribution in models[model_id]["distributions"]:
            for artifact in distribution["artifacts"]:
                if artifact_id and artifact["id"] != artifact_id:
                    continue
                found = True
                yield model_id, distribution, artifact
    if artifact_id and not found:
        raise MetadataError(f"Unknown artifact: {artifact_id}")


def _collect(
    registry: dict[str, Any], profile: str | None, artifact_id: str | None, update: bool
) -> int:
    changed = False
    with tempfile.TemporaryDirectory(prefix="kokoro-registry-") as temporary:
        temporary_dir = Path(temporary)
        for model_id, distribution, artifact in _iter_artifacts(
            registry, profile, artifact_id
        ):
            target = (
                temporary_dir / model_id / distribution["id"] / artifact["local_name"]
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {model_id}/{distribution['id']}/{artifact['id']}")
            try:
                size, digest = _download(artifact["url"], target)
                _validate_format(target, artifact)
            except Exception as exc:
                raise MetadataError(f"{model_id}/{artifact['id']}: {exc}") from exc
            if artifact.get("size") is not None and artifact["size"] != size:
                raise MetadataError(
                    f"{model_id}/{artifact['id']}: upstream size changed from {artifact['size']} to {size}"
                )
            if artifact.get("sha256") is not None and artifact["sha256"] != digest:
                raise MetadataError(
                    f"{model_id}/{artifact['id']}: upstream SHA-256 changed"
                )
            print(f"  size={size} sha256={digest}")
            if update and (
                artifact.get("size") != size or artifact.get("sha256") != digest
            ):
                artifact["size"] = size
                artifact["sha256"] = digest
                changed = True
    if update and changed:
        registry_path = REGISTRY
        registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"updated {registry_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    global REGISTRY
    parser = argparse.ArgumentParser(
        description="Collect runtime artifact sizes and SHA-256 values"
    )
    parser.add_argument("profile", nargs="?")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="check committed metadata without downloading",
    )
    mode.add_argument(
        "--update", action="store_true", help="download and update metadata"
    )
    parser.add_argument("--artifact", help="limit update/check to one artifact ID")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args(argv)
    REGISTRY = args.registry
    try:
        registry = _load(REGISTRY)
        if args.check:
            from verify_model_registry import verify_registry

            verify_registry(REGISTRY)
            print(
                f"Checked {sum(len(d['artifacts']) for m in registry['models'].values() for d in m['distributions'])} committed runtime artifacts"
            )
            return 0
        return _collect(registry, args.profile, args.artifact, update=True)
    except MetadataError as exc:
        print(f"metadata collection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
