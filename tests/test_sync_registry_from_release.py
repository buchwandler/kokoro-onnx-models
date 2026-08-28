from __future__ import annotations

from pathlib import Path

from scripts import sync_registry_from_release as sync
from scripts.update_registry_from_release import distribution_from_manifest


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
