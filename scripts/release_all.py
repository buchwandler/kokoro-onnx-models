#!/usr/bin/env python3
"""Plan and compare the catalog-driven set of independent releases."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "releases.json"


class ReleaseConflictError(ValueError):
    """Raised when an existing release differs from a candidate."""


def load_catalog(path: Path = CATALOG) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("releases"), dict):
        raise TypeError("Release catalog must contain a releases object")
    return data


def publishable_matrix(catalog: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    releases = catalog["releases"]
    entries = [
        {
            "release_key": str(key),
            "profile": str(spec.get("profile", key)),
            "tag": str(spec["tag"]),
            "kind": str(spec["kind"]),
        }
        for key, spec in releases.items()
        if spec.get("publish", True) is True
    ]
    return {"include": sorted(entries, key=lambda item: item["release_key"])}


def _comparable_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(manifest))
    result.pop("generated_at", None)
    return result


def _asset_digests(manifest: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    return {
        str(asset["name"]): (int(asset["size"]), str(asset["sha256"]))
        for asset in manifest.get("assets", [])
    }


def compare_existing_release(
    candidate: Mapping[str, Any], existing: Mapping[str, Any] | None
) -> str:
    """Return ``publish`` or ``skip`` and reject a non-equivalent release."""
    if existing is None:
        return "publish"
    if _comparable_manifest(candidate) != _comparable_manifest(existing):
        raise ReleaseConflictError("existing release differs from candidate")
    if _asset_digests(candidate) != _asset_digests(existing):
        raise ReleaseConflictError(
            "existing release asset hashes differ from candidate"
        )
    return "skip"


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "compare":
        parser = argparse.ArgumentParser(
            description="Compare an existing release manifest"
        )
        parser.add_argument("compare")
        parser.add_argument("--candidate", type=Path, required=True)
        parser.add_argument("--existing", type=Path, required=True)
        args = parser.parse_args(arguments)
        try:
            candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
            existing = json.loads(args.existing.read_text(encoding="utf-8"))
            result = compare_existing_release(candidate, existing)
        except (OSError, ValueError, json.JSONDecodeError, ReleaseConflictError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(result)
        return 0

    parser = argparse.ArgumentParser(
        description="Plan all publishable catalog releases"
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(arguments)
    try:
        matrix = publishable_matrix(load_catalog(args.catalog))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"release-all plan failed: {exc}", file=sys.stderr)
        return 1
    output = (
        json.dumps(matrix, sort_keys=True, separators=(",", ":"))
        if args.json
        else json.dumps(matrix, indent=2)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
