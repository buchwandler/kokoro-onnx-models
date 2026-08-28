from __future__ import annotations

import json
from pathlib import Path

from scripts import sync_registry_from_release as sync
from scripts.update_registry_from_release import (
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
