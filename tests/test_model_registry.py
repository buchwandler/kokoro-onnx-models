from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_model_registry import RegistryError, verify_registry


def load_registry() -> dict:
    return json.loads((ROOT / "catalog" / "models.json").read_text(encoding="utf-8"))


def test_committed_registry_is_valid() -> None:
    registry = verify_registry()
    assert len(registry["models"]) == 18


def test_github_distributions_match_release_catalog() -> None:
    registry = load_registry()
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())["releases"]
    for model in registry["models"].values():
        for distribution in model["distributions"]:
            if distribution["provider"] != "github-release":
                continue
            release = releases[distribution["release_key"]]
            assert isinstance(distribution["release_version"], int)
            assert distribution["release_version"] >= 1
            assert distribution["release_tag"] == release["tag"]
            assert distribution["release_version"] == release["release_version"]
            assert model["model_version"] == release["model_version"]


def test_software_mansion_anna_registry_metadata_is_ready_for_activation() -> None:
    model = load_registry()["models"]["de-anna"]

    assert model["language_codes"] == ["de"]
    assert model["frontend"] == "german-ipa-v1"
    assert model["runtime"]["default_voice"] == "df_anna"
    assert model["runtime"]["voices"] == ["df_anna"]
    assert model["onnx_contract"]["outputs"] == {
        "audio": "float32",
        "duration": "int64",
    }


def test_software_mansion_mateusz_remains_staged() -> None:
    model = load_registry()["models"]["pl-mateusz"]

    assert model["language_codes"] == ["pl"]
    assert model["frontend"] == "phonemis-pl-v1"
    assert model["runtime_available"] is False
    assert model["runtime"]["default_voice"] == "pm_mateusz"
    assert model["runtime"]["voices"] == ["pm_mateusz"]
    assert model["distributions"] == []


def test_european_portuguese_registry_exposes_token_durations() -> None:
    model = load_registry()["models"]["pt-eu-logus2k"]
    assert model["language_codes"] == ["pt-pt"]
    assert model["runtime_available"] is True
    assert model["frontend"] == "tts-eu-pt-v1"
    assert model["runtime"]["default_voice"] == "pt_eu"
    assert model["runtime"]["voice_metadata"] == {
        "pt_eu": {
            "gender": "female",
            "language": "pt",
            "locale": "pt-PT",
            "language_label": "European Portuguese",
        }
    }
    assert model["onnx_contract"]["timing"] == {
        "kind": "token-duration-v1",
        "output": "duration",
        "unit": "frame",
        "samples_per_frame": 600,
        "includes_boundary_tokens": True,
    }
    distribution = model["distributions"][0]
    assert distribution["release_key"] == "pt-eu-logus2k"
    assert any(asset["role"] == "model" for asset in distribution["artifacts"])


def test_ngoc_huyen_registry_exposes_token_durations() -> None:
    model = load_registry()["models"]["vi-ngoc-huyen"]
    assert model["runtime"]["default_voice"] == "ngoc_huyen"
    assert model["onnx_contract"]["timing"] == {
        "kind": "token-duration-v1",
        "output": "duration",
        "unit": "frame",
        "samples_per_frame": 600,
        "includes_boundary_tokens": True,
    }
    distribution = model["distributions"][0]
    assert distribution["release_key"] == "vi-ngoc-huyen"
    assert any(asset["role"] == "model" for asset in distribution["artifacts"])


def test_russian_uses_separate_checkpoint_build_releases() -> None:
    registry = load_registry()
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    base = registry["models"]["ru-zaakirio-base"]
    dima = registry["models"]["ru-zaakirio-dima"]

    assert "ru-zaakirio-base" in releases["releases"]
    assert "ru-zaakirio-dima" in releases["releases"]
    assert base["mirror_policy"] == dima["mirror_policy"] == "preferred"
    assert base["distributions"][0]["provider"] == "github-release"
    assert dima["distributions"][0]["provider"] == "github-release"
    assert base["distributions"][0]["release_key"] == "ru-zaakirio-base"
    assert dima["distributions"][0]["release_key"] == "ru-zaakirio-dima"
    assert base["distributions"][0]["artifacts"][0]["local_name"] == "bundle.json"
    assert dima["distributions"][0]["artifacts"][0]["local_name"] == "bundle.json"
    assert base["runtime"]["voices"] == ["sveta", "masha"]
    assert dima["runtime"]["voices"] == ["dima"]
    assert base["onnx_contract"]["outputs"] == {"audio": "float32", "duration": "int64"}
    assert dima["onnx_contract"]["timing"]["output"] == "duration"


def test_all_runtime_artifacts_have_pinned_metadata() -> None:
    registry = load_registry()
    for model in registry["models"].values():
        for distribution in model["distributions"]:
            for artifact in distribution["artifacts"]:
                assert artifact["url"].startswith("https://")
                assert artifact["size"] > 0
                assert len(artifact["sha256"]) == 64
                if distribution["provider"] == "huggingface":
                    assert "/resolve/main/" not in artifact["url"]


def test_raw_voice_declares_shape() -> None:
    registry = load_registry()
    for model in registry["models"].values():
        for distribution in model["distributions"]:
            for artifact in distribution["artifacts"]:
                if artifact["format"] == "raw-float32-le":
                    assert artifact["handling"]["dtype"] == "float32"
                    assert artifact["handling"]["shape"] == [510, 256]


def test_voice_metadata_matches_runtime_roster() -> None:
    registry = load_registry()
    for model in registry["models"].values():
        metadata = model["runtime"].get("voice_metadata")
        if metadata is not None:
            assert set(metadata) == set(model["runtime"]["voices"])
            assert model["runtime"]["default_voice"] in metadata


def test_invalid_voice_metadata_is_rejected(tmp_path: Path) -> None:
    registry = load_registry()
    model = registry["models"]["v1.0"]
    model["runtime"]["voice_metadata"]["not-a-voice"] = {
        "gender": "female",
        "language": "en",
        "locale": "en-US",
        "language_label": "American English",
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(RegistryError, match="outside the roster"):
        verify_registry(path)


def test_invalid_registry_cases_are_rejected(tmp_path: Path) -> None:
    registry = load_registry()
    registry["models"]["ru-zaakirio-base"]["distributions"][0]["artifacts"][0][
        "url"
    ] = registry["models"]["ru-zaakirio-base"]["distributions"][0]["artifacts"][0][
        "url"
    ].replace("https://", "http://", 1)
    path = tmp_path / "models.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(RegistryError, match="https://"):
        verify_registry(path)


def test_metadata_collector_fills_missing_values(monkeypatch, tmp_path: Path) -> None:
    from scripts import collect_runtime_metadata

    registry = load_registry()
    artifact = registry["models"]["ru-zaakirio-base"]["distributions"][0]["artifacts"][
        0
    ]
    artifact.pop("size")
    artifact.pop("sha256")
    registry_path = tmp_path / "models.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    def fake_download(url: str, target: Path) -> tuple[int, str]:
        target.write_bytes(b"registry-test")
        return 13, "0" * 64

    monkeypatch.setattr(
        collect_runtime_metadata, "_validate_format", lambda path, artifact: None
    )
    monkeypatch.setattr(collect_runtime_metadata, "_download", fake_download)
    collect_runtime_metadata.REGISTRY = registry_path
    assert (
        collect_runtime_metadata._collect(
            registry, "ru-zaakirio-base", "bundle-bundle", True
        )
        == 0
    )
    updated = json.loads(registry_path.read_text(encoding="utf-8"))
    collected = updated["models"]["ru-zaakirio-base"]["distributions"][0]["artifacts"][
        0
    ]
    assert collected["size"] == 13
    assert collected["sha256"] == "0" * 64


def test_thai_split_components_remain_explicit() -> None:
    thai = load_registry()["models"]["th-wayu"]
    assert thai["runtime"]["layout"] == "split-onnx-v1"
    assert {
        a["component"]
        for a in thai["distributions"][0]["artifacts"]
        if a["role"] == "model"
    } == {
        "prosody",
        "curves",
        "decoder",
    }
