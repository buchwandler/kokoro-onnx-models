from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import sync_registry_from_release as sync
from scripts.update_registry_from_release import (
    RegistryReleaseError,
    distribution_from_manifest,
    sync_release,
)


def test_distribution_from_manifest_contains_artifacts() -> None:
    manifest = {
        "tag": "model-files-test",
        "profile": "test",
        "assets": [
            {
                "name": "model.onnx",
                "role": "model",
                "format": "onnx",
                "size": 4,
                "sha256": "a" * 64,
                "quality": "fp32",
            }
        ],
    }

    distribution = distribution_from_manifest(manifest, {})

    assert distribution["artifacts"] == [
        {
            "id": "model-model",
            "role": "model",
            "url": "https://github.com/buchwandler/kokoro-onnx-models/releases/download/model-files-test/model.onnx",
            "local_name": "model.onnx",
            "format": "onnx",
            "size": 4,
            "sha256": "a" * 64,
            "quality": "fp32",
        }
    ]


def test_download_release_fetches_manifest_checksums_and_assets(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        sync, "_download", lambda url, target: calls.append((url, target))
    )

    sync._download_release(
        tmp_path,
        repository="buchwandler/kokoro-onnx-models",
        tag="model-files-test",
        manifest={"assets": [{"name": "model.onnx"}, {"name": "voices.npz"}]},
    )

    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == [
        "SHA256SUMS",
        "model.onnx",
        "voices.npz",
    ]


def test_sync_release_copies_timing_contract(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    timing = {
        "kind": "token-duration-v1",
        "output": "duration",
        "unit": "frame",
        "samples_per_frame": 600,
        "includes_boundary_tokens": True,
    }
    contract = {
        "inputs": {"tokens": "int64"},
        "outputs": {"audio": "float32", "duration": "int64"},
        "timing": timing,
        "max_tokens": 510,
    }
    (candidate / "release-manifest.json").write_text(
        json.dumps(
            {
                "tag": "model-files-test",
                "profile": "test",
                "onnx_contract": contract,
                "assets": [],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "models": {
                    "test": {
                        "onnx_contract": {"outputs": {"audio": "float32"}},
                        "distributions": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    releases = tmp_path / "releases.json"
    releases.write_text(
        json.dumps({"releases": {"test": {"tag": "model-files-test"}}}),
        encoding="utf-8",
    )
    sync_release(
        candidate,
        profile="test",
        registry_path=registry,
        releases_path=releases,
        update=True,
    )
    updated = json.loads(registry.read_text(encoding="utf-8"))
    assert updated["models"]["test"]["onnx_contract"] == contract


def _sync_fixture(
    tmp_path: Path,
    *,
    existing_size: int,
    existing_sha: str,
    generated_size: int,
    generated_sha: str,
    existing_tag: str = "model-files-test",
    generated_tag: str = "model-files-test",
) -> tuple[Path, Path, Path]:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = {
        "tag": generated_tag,
        "profile": "test",
        "onnx_contract": {"outputs": {"audio": "float32"}},
        "assets": [
            {
                "name": "model.onnx",
                "role": "model",
                "format": "onnx",
                "size": generated_size,
                "sha256": generated_sha,
                "quality": "fp32",
            }
        ],
    }
    (candidate / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    existing_manifest = {
        **manifest,
        "tag": existing_tag,
        "assets": [
            {**manifest["assets"][0], "size": existing_size, "sha256": existing_sha}
        ],
    }
    existing = distribution_from_manifest(existing_manifest, {})
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps({"models": {"test": {"distributions": [existing]}}}),
        encoding="utf-8",
    )
    releases = tmp_path / "releases.json"
    releases.write_text(
        json.dumps({"releases": {"test": {"tag": generated_tag}}}), encoding="utf-8"
    )
    return candidate, registry, releases


def test_sync_release_allows_identical_existing_release_tag(tmp_path: Path) -> None:
    candidate, registry, releases = _sync_fixture(
        tmp_path,
        existing_size=4,
        existing_sha="a" * 64,
        generated_size=4,
        generated_sha="a" * 64,
    )

    sync_release(
        candidate,
        profile="test",
        registry_path=registry,
        releases_path=releases,
        update=True,
    )


def test_sync_release_rejects_changed_size_for_existing_release_tag(
    tmp_path: Path,
) -> None:
    candidate, registry, releases = _sync_fixture(
        tmp_path,
        existing_size=4,
        existing_sha="a" * 64,
        generated_size=5,
        generated_sha="a" * 64,
    )

    with pytest.raises(RegistryReleaseError, match="immutable"):
        sync_release(
            candidate,
            profile="test",
            registry_path=registry,
            releases_path=releases,
            update=True,
        )


def test_sync_release_rejects_changed_sha_for_existing_release_tag(
    tmp_path: Path,
) -> None:
    candidate, registry, releases = _sync_fixture(
        tmp_path,
        existing_size=4,
        existing_sha="a" * 64,
        generated_size=4,
        generated_sha="b" * 64,
    )

    with pytest.raises(RegistryReleaseError, match="immutable"):
        sync_release(
            candidate,
            profile="test",
            registry_path=registry,
            releases_path=releases,
            update=True,
        )


def test_sync_release_allows_new_release_tag_with_new_artifact_hashes(
    tmp_path: Path,
) -> None:
    candidate, registry, releases = _sync_fixture(
        tmp_path,
        existing_size=4,
        existing_sha="a" * 64,
        generated_size=5,
        generated_sha="b" * 64,
        existing_tag="model-files-test-v1",
        generated_tag="model-files-test-v2",
    )

    sync_release(
        candidate,
        profile="test",
        registry_path=registry,
        releases_path=releases,
        update=True,
    )
    updated = json.loads(registry.read_text(encoding="utf-8"))
    generated = updated["models"]["test"]["distributions"][0]
    assert generated["release_tag"] == "model-files-test-v2"
    assert generated["artifacts"][0]["sha256"] == "b" * 64
