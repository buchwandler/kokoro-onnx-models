from __future__ import annotations

import pytest

from scripts.release_all import (
    ReleaseConflictError,
    compare_existing_release,
    publishable_matrix,
)


def test_publishable_matrix_is_sorted_and_excludes_disabled_and_upstream_only() -> None:
    catalog = {
        "releases": {
            "z": {"kind": "build", "tag": "z-tag"},
            "disabled": {"kind": "build", "tag": "disabled", "publish": False},
            "a": {"kind": "mirror", "tag": "a-tag"},
        }
    }

    assert publishable_matrix(catalog) == {
        "include": [
            {"release_key": "a", "profile": "a", "tag": "a-tag", "kind": "mirror"},
            {"release_key": "z", "profile": "z", "tag": "z-tag", "kind": "build"},
        ]
    }


def _manifest(digest: str = "a" * 64) -> dict[str, object]:
    return {
        "tag": "model-files-test",
        "profile": "test",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "assets": [{"name": "model.onnx", "size": 4, "sha256": digest}],
    }


def test_existing_release_comparison_is_idempotent() -> None:
    candidate = _manifest()
    existing = {**candidate, "generated_at": "2026-02-01T00:00:00+00:00"}

    assert compare_existing_release(candidate, existing) == "skip"
    assert compare_existing_release(candidate, None) == "publish"


def test_existing_release_difference_fails() -> None:
    with pytest.raises(ReleaseConflictError, match="differs"):
        compare_existing_release(_manifest(), _manifest("b" * 64))
