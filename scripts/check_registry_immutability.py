#!/usr/bin/env python3
"""Check that unchanged GitHub release tags retain immutable catalog metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_IMMUTABLE_ARTIFACT_FIELDS = (
    "id",
    "role",
    "url",
    "local_name",
    "format",
    "size",
    "sha256",
    "quality",
    "component",
    "handling",
    "voice",
)


class RegistryImmutabilityError(ValueError):
    """Raised when an unchanged release tag has changed artifact metadata."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryImmutabilityError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryImmutabilityError(f"{path} must contain an object")
    return value


def _artifact_signature(distribution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["id"]): {
            field: item.get(field) for field in _IMMUTABLE_ARTIFACT_FIELDS
        }
        for item in distribution.get("artifacts", [])
    }


def _github_distributions(model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        distribution
        for distribution in model.get("distributions", [])
        if distribution.get("provider") == "github-release"
    ]


def check_immutability(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Reject changed artifact identity for GitHub distributions with the same tag."""
    before_models = before.get("models", {})
    after_models = after.get("models", {})
    if not isinstance(before_models, dict) or not isinstance(after_models, dict):
        raise RegistryImmutabilityError("Catalogs must contain a models object")

    for model_id in before_models.keys() & after_models.keys():
        old_distributions = _github_distributions(before_models[model_id])
        new_by_tag = {
            distribution.get("release_tag"): distribution
            for distribution in _github_distributions(after_models[model_id])
        }
        for old_distribution in old_distributions:
            tag = old_distribution.get("release_tag")
            new_distribution = new_by_tag.get(tag)
            if new_distribution is None:
                continue
            if _artifact_signature(old_distribution) != _artifact_signature(
                new_distribution
            ):
                raise RegistryImmutabilityError(
                    f"Published release {tag!r} for model {model_id!r} changed immutable "
                    "artifact metadata. Publish a new release tag instead."
                )


def check_catalog(before_path: Path, after_path: Path) -> None:
    check_immutability(_load(before_path), _load(after_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        check_catalog(args.before, args.after)
    except RegistryImmutabilityError as exc:
        print(f"registry immutability check failed: {exc}", file=sys.stderr)
        return 1
    print(f"registry immutability check passed: {args.before} -> {args.after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
