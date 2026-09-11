from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    distribution = distribution_from_manifest(manifest, {"release_version": 2})
    assert distribution["release_version"] == 2

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
                "model_version": "1.0",
                "release_version": 1,
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
        json.dumps(
            {
                "releases": {
                    "test": {
                        "tag": "model-files-test",
                        "model_version": "1.0",
                        "release_version": 1,
                    }
                }
            }
        ),
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


def test_sync_release_updates_stale_runtime_identity(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    contract = {
        "inputs": {"tokens": "int64"},
        "outputs": {"audio": "float32"},
        "max_tokens": 510,
    }
    runtime = {
        "language_codes": ["pt-pt"],
        "sample_rate": 24000,
        "frontend": "tts-eu-pt-v1",
        "layout": "single-onnx-v1",
        "max_tokens": 510,
        "default_voice": "pt_eu",
        "voices": ["pt_eu"],
    }
    (candidate / "release-manifest.json").write_text(
        json.dumps(
            {
                "tag": "model-files-test",
                "profile": "test",
                "model_version": "1.0",
                "release_version": 1,
                "runtime": runtime,
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
                        "language_codes": ["pt"],
                        "frontend": "tts-eu-pt-v1",
                        "sample_rate": 24000,
                        "runtime": {
                            "layout": "single-onnx-v1",
                            "max_tokens": 510,
                            "default_voice": "pt_eu",
                            "voices": ["pt_eu"],
                        },
                        "onnx_contract": contract,
                        "distributions": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    releases = tmp_path / "releases.json"
    releases.write_text(
        json.dumps(
            {
                "releases": {
                    "test": {
                        "tag": "model-files-test",
                        "model_version": "1.0",
                        "release_version": 1,
                    }
                }
            }
        ),
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
    model = updated["models"]["test"]
    assert model["language_codes"] == ["pt-pt"]
    assert model["frontend"] == "tts-eu-pt-v1"
    assert model["runtime"]["voices"] == ["pt_eu"]


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
        "model_version": "1.0",
        "release_version": 1,
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
    existing = distribution_from_manifest(
        existing_manifest, {"model_version": "1.0", "release_version": 1}
    )
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps({"models": {"test": {"distributions": [existing]}}}),
        encoding="utf-8",
    )
    releases = tmp_path / "releases.json"
    releases.write_text(
        json.dumps(
            {
                "releases": {
                    "test": {
                        "tag": generated_tag,
                        "model_version": "1.0",
                        "release_version": 1,
                    }
                }
            }
        ),
    )
    return candidate, registry, releases


def test_sync_release_does_not_activate_staged_release(tmp_path: Path) -> None:
    candidate, registry, releases = _sync_fixture(
        tmp_path,
        existing_size=4,
        existing_sha="a" * 64,
        generated_size=5,
        generated_sha="b" * 64,
        existing_tag="model-files-test-v1",
        generated_tag="model-files-test-v2",
    )
    release_data = json.loads(releases.read_text(encoding="utf-8"))
    release_data["releases"]["test"]["activate_runtime_registry"] = False
    releases.write_text(json.dumps(release_data), encoding="utf-8")
    before = registry.read_text(encoding="utf-8")

    sync_release(
        candidate,
        profile="test",
        registry_path=registry,
        releases_path=releases,
        update=True,
    )

    assert registry.read_text(encoding="utf-8") == before


def test_sync_release_activates_de_anna(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    tag = "model-files-german-software-mansion-anna-v1"
    contract = {
        "inputs": {
            "tokens": "int64",
            "style": "float32",
            "speed": "float32",
        },
        "outputs": {"audio": "float32", "duration": "int64"},
        "timing": {
            "kind": "token-duration-v1",
            "output": "duration",
            "unit": "frame",
            "samples_per_frame": 600,
            "includes_boundary_tokens": True,
        },
        "max_tokens": 510,
    }
    manifest = {
        "tag": tag,
        "profile": "de-anna",
        "model_version": "1",
        "release_version": 1,
        "onnx_contract": contract,
        "assets": [
            {
                "name": "model.onnx",
                "role": "model",
                "format": "onnx",
                "size": 4,
                "sha256": "a" * 64,
                "quality": "fp32",
            },
            {
                "name": "voices.npz",
                "role": "voices",
                "format": "numpy-npz",
                "size": 4,
                "sha256": "b" * 64,
            },
            {
                "name": "config.json",
                "role": "config",
                "format": "json",
                "size": 4,
                "sha256": "c" * 64,
            },
            {
                "name": "bundle.json",
                "role": "bundle",
                "format": "json",
                "size": 4,
                "sha256": "d" * 64,
            },
        ],
    }
    (candidate / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "models": {
                    "de-anna": {
                        "runtime_available": False,
                        "frontend": "german-ipa-v1",
                        "distributions": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    releases = tmp_path / "releases.json"
    releases.write_text(
        json.dumps(
            {
                "releases": {
                    "de-anna": {
                        "tag": tag,
                        "model_version": "1",
                        "release_version": 1,
                        "activate_runtime_registry": True,
                        "source_repository": "software-mansion/react-native-executorch-kokoro",
                        "source_revision": "9a8b5878012e01a26dad2618068dc61215994785",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    sync_release(
        candidate,
        profile="de-anna",
        registry_path=registry,
        releases_path=releases,
        update=True,
    )

    model = json.loads(registry.read_text(encoding="utf-8"))["models"]["de-anna"]
    distribution = model["distributions"][0]
    assert model["runtime_available"] is True
    assert distribution["provider"] == "github-release"
    assert distribution["runtime_ready"] is True
    assert distribution["release_key"] == "de-anna"
    assert distribution["release_tag"] == tag
    assert model["onnx_contract"] == contract


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
    assert generated["release_version"] == 1


def test_sync_release_rejects_model_version_mismatch(tmp_path: Path) -> None:
    candidate, registry, releases = _sync_fixture(
        tmp_path,
        existing_size=4,
        existing_sha="a" * 64,
        generated_size=4,
        generated_sha="a" * 64,
    )
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_version"] = "2.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RegistryReleaseError, match="model_version"):
        sync_release(
            candidate,
            profile="test",
            registry_path=registry,
            releases_path=releases,
            update=True,
        )


def test_sync_release_rejects_release_version_mismatch(tmp_path: Path) -> None:
    candidate, registry, releases = _sync_fixture(
        tmp_path,
        existing_size=4,
        existing_sha="a" * 64,
        generated_size=4,
        generated_sha="a" * 64,
    )
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RegistryReleaseError, match="release_version"):
        sync_release(
            candidate,
            profile="test",
            registry_path=registry,
            releases_path=releases,
            update=True,
        )
