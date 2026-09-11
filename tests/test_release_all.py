from __future__ import annotations

from pathlib import Path

import pytest

from scripts.release_all import (
    ReleaseConflictError,
    compare_existing_release,
    publishable_matrix,
)


def test_publishable_matrix_is_sorted_and_excludes_disabled_and_upstream_only() -> None:
    catalog = {
        "releases": {
            "z": {
                "kind": "build",
                "tag": "z-tag",
                "model_version": "1",
                "release_version": 1,
            },
            "disabled": {
                "kind": "build",
                "tag": "disabled",
                "model_version": "1",
                "release_version": 1,
                "publish": False,
            },
            "a": {
                "kind": "mirror",
                "tag": "a-tag",
                "model_version": "1",
                "release_version": 1,
            },
        }
    }

    assert publishable_matrix(catalog) == {
        "include": [
            {
                "release_key": "a",
                "profile": "a",
                "tag": "a-tag",
                "kind": "mirror",
                "model_version": "1",
                "release_version": 1,
            },
            {
                "release_key": "z",
                "profile": "z",
                "tag": "z-tag",
                "kind": "build",
                "model_version": "1",
                "release_version": 1,
            },
        ]
    }


def test_publishable_matrix_includes_staged_releases() -> None:
    matrix = publishable_matrix(
        {
            "releases": {
                "de-anna": {
                    "kind": "build",
                    "tag": "anna-tag",
                    "model_version": "1",
                    "release_version": 1,
                    "publish": True,
                    "activate_runtime_registry": True,
                },
                "pl-mateusz": {
                    "kind": "build",
                    "tag": "mateusz-tag",
                    "model_version": "1",
                    "release_version": 1,
                    "publish": True,
                    "activate_runtime_registry": False,
                },
            }
        }
    )

    assert [item["release_key"] for item in matrix["include"]] == [
        "de-anna",
        "pl-mateusz",
    ]


def test_sync_workflow_skips_staged_runtime_activation() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "release-all.yml"
    ).read_text(encoding="utf-8")

    assert 'get("activate_runtime_registry", True)' in workflow
    assert 'if [ "$activate" != "true" ]; then' in workflow
    assert "continue" in workflow

def test_catalog_writers_share_safe_concurrency_and_push_contract() -> None:
    root = Path(__file__).parents[1] / ".github" / "workflows"
    workflows = {
        name: (root / name).read_text(encoding="utf-8")
        for name in ("release-all.yml", "publish-release.yml", "refresh-catalog.yml")
    }
    for workflow in workflows.values():
        assert "kokoro-model-catalog-v1" in workflow
        assert "git pull --rebase origin main" in workflow
        assert "git push origin HEAD:main" in workflow
        assert "--force" not in workflow
    publish = workflows["publish-release.yml"]
    assert publish.index("  publish:\n") < publish.index("  sync-catalog:\n")
    assert "scripts/update_registry_from_release.py" in publish
    assert "scripts/verify_model_registry.py" in publish
    assert "scripts/collect_runtime_metadata.py --check" in publish
    assert "git show HEAD^:catalog/models.json" in publish
    assert "scripts/sync_registry_from_release.py" in workflows["refresh-catalog.yml"]


def test_release_all_matrix_exposes_both_versions() -> None:
    matrix = publishable_matrix(
        {
            "releases": {
                "test": {
                    "kind": "build",
                    "tag": "test-tag",
                    "model_version": "1",
                    "release_version": 2,
                }
            }
        }
    )
    assert matrix["include"][0]["model_version"] == "1"
    assert matrix["include"][0]["release_version"] == 2



def test_anna_consumer_gates_install_espeak_ng() -> None:
    root = Path(__file__).parents[1] / ".github" / "workflows"
    for workflow_name in ("build-release.yml", "release-all.yml"):
        workflow = (root / workflow_name).read_text(encoding="utf-8")
        install = "sudo apt-get install --no-install-recommends -y espeak-ng"
        gate = "German Anna pykokoro consumer gate"
        assert install in workflow
        assert workflow.index(install) < workflow.index(gate)


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
