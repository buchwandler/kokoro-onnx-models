from __future__ import annotations

import copy

import pytest

from scripts.check_registry_immutability import (
    RegistryImmutabilityError,
    check_immutability,
)


def _catalog(
    *, tag: str = "model-files-test", size: int = 4, sha: str = "a" * 64
) -> dict:
    return {
        "models": {
            "test": {
                "distributions": [
                    {
                        "id": f"github-{tag}",
                        "provider": "github-release",
                        "release_tag": tag,
                        "artifacts": [
                            {
                                "id": "model-model",
                                "role": "model",
                                "url": f"https://example.test/{tag}/model.onnx",
                                "local_name": "model.onnx",
                                "format": "onnx",
                                "size": size,
                                "sha256": sha,
                                "quality": "fp32",
                            }
                        ],
                    }
                ]
            }
        }
    }


def test_immutability_checker_allows_identical_catalog() -> None:
    catalog = _catalog()
    check_immutability(catalog, copy.deepcopy(catalog))


@pytest.mark.parametrize(
    ("field", "value"),
    [("size", 5), ("sha256", "b" * 64)],
)
def test_immutability_checker_rejects_same_tag_artifact_changes(
    field: str, value
) -> None:
    before = _catalog()
    after = copy.deepcopy(before)
    after["models"]["test"]["distributions"][0]["artifacts"][0][field] = value

    with pytest.raises(RegistryImmutabilityError, match="immutable"):
        check_immutability(before, after)


def test_immutability_checker_allows_changed_release_tag() -> None:
    check_immutability(
        _catalog(), _catalog(tag="model-files-test-v2", size=5, sha="b" * 64)
    )


def test_immutability_checker_allows_removed_distribution() -> None:
    before = _catalog()
    after = {"models": {"test": {"distributions": []}}}
    check_immutability(before, after)
