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
    assert len(registry["models"]) == 20


def test_github_distributions_match_release_catalog() -> None:
    registry = load_registry()
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())["releases"]
    for model in registry["models"].values():
        for distribution in model["distributions"]:
            if distribution["provider"] != "github-release":
                continue
            release = releases[distribution["release_key"]]
            if release.get("release_version", 0) > distribution.get(
                "release_version", 0
            ):
                continue
            assert isinstance(distribution["release_version"], int)
            assert distribution["release_version"] >= 1
            assert distribution["release_tag"] == release["tag"]
            assert distribution["release_version"] == release["release_version"]
            assert model["model_version"] == release["model_version"]


def test_nabra_registry_matches_public_build_voice() -> None:
    registry = load_registry()
    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    model = registry["models"]["ar-nabra"]
    distribution = next(
        item for item in model["distributions"] if item["provider"] == "github-release"
    )

    assert tuple(profiles["ar-nabra"]["voices"]["items"]) == ("default",)
    assert model["model_version"] == "0.1"
    assert model["runtime"]["default_voice"] == "default"
    assert model["runtime"]["voices"] == ["default"]
    assert set(model["runtime"]["voice_metadata"]) == {"default"}
    assert distribution["release_tag"] == "model-files-arabic-nabra-v0.1-r2"
    assert distribution["release_version"] == 2


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
    if model["runtime_available"]:
        distribution = model["distributions"][0]
        assert distribution["release_key"] == "vi-ngoc-huyen"
        assert any(asset["role"] == "model" for asset in distribution["artifacts"])
    else:
        assert model["distributions"] == []


def test_russian_uses_separate_checkpoint_build_releases() -> None:
    registry = load_registry()
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    base = registry["models"]["ru-zaakirio-base"]
    dima = registry["models"]["ru-zaakirio-dima"]

    assert "ru-zaakirio-base" in releases["releases"]
    assert "ru-zaakirio-dima" in releases["releases"]
    assert base["mirror_policy"] == dima["mirror_policy"] == "preferred"
    assert base["runtime_available"] is True
    assert dima["runtime_available"] is True
    assert len(base["distributions"]) == 1
    assert len(dima["distributions"]) == 1
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
    with pytest.raises(
        RegistryError,
        match="voice_metadata must exactly cover the runtime voice roster",
    ):
        verify_registry(path)


def test_partial_voice_metadata_is_rejected(tmp_path: Path) -> None:
    registry = load_registry()
    model = registry["models"]["v1.0"]
    model["runtime"]["voice_metadata"].pop("af_alloy")
    path = tmp_path / "models.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(
        RegistryError,
        match="voice_metadata must exactly cover the runtime voice roster",
    ):
        verify_registry(path)


def test_invalid_registry_cases_are_rejected(tmp_path: Path) -> None:
    registry = load_registry()
    artifact = registry["models"]["v1.0"]["distributions"][0]["artifacts"][0]
    artifact["url"] = artifact["url"].replace("https://", "http://", 1)
    path = tmp_path / "models.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(RegistryError, match="https://"):
        verify_registry(path)


def test_metadata_collector_fills_missing_values(monkeypatch, tmp_path: Path) -> None:
    from scripts import collect_runtime_metadata

    registry = load_registry()
    artifact = registry["models"]["v1.0"]["distributions"][0]["artifacts"][0]
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
        collect_runtime_metadata._collect(registry, "v1.0", "model-kokoro-v1.0", True)
        == 0
    )
    updated = json.loads(registry_path.read_text(encoding="utf-8"))
    collected = updated["models"]["v1.0"]["distributions"][0]["artifacts"][0]
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


def test_registry_schema_accepts_reference_cloning_without_static_voices() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    schema = json.loads((ROOT / "schemas" / "model-registry.schema.json").read_text())
    registry = load_registry()
    jsonschema.Draft202012Validator(schema).validate(registry)

    model = registry["models"]["en-akinvox-cloning-v1"]
    assert model["runtime"]["voice_mode"] == "reference"
    assert model["runtime"]["layout"] == "cloning-onnx-v1"
    assert "voices" not in model["runtime"]
    assert "default_voice" not in model["runtime"]


def test_registry_schema_keeps_static_voice_requirements() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    schema = json.loads((ROOT / "schemas" / "model-registry.schema.json").read_text())
    registry = load_registry()

    missing_voices = json.loads(json.dumps(registry))
    missing_voices["models"]["en-oddadmix-7m-distill"]["runtime"].pop("voices")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(missing_voices)

    fake_reference_voice = json.loads(json.dumps(registry))
    fake_reference_voice["models"]["en-akinvox-cloning-v1"]["runtime"][
        "default_voice"
    ] = "reference"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(fake_reference_voice)


def test_cloning_contract_components_cover_the_exact_graph_set() -> None:
    model = load_registry()["models"]["en-akinvox-cloning-v1"]
    assert set(model["onnx_contract"]["components"]) == {
        "reference_wavlm",
        "reference_encoders",
        "reference_mapper",
        "prosody",
        "curves",
        "decoder",
    }
    assert model["runtime"]["speed_supported"] is False


def test_v1_0_registry_omits_inno_enroller_until_release_activation() -> None:
    registry = load_registry()
    runtime = registry["models"]["v1.0"]["runtime"]
    release_catalog = json.loads(
        (ROOT / "catalog" / "releases.json").read_text(encoding="utf-8")
    )
    release_runtime = release_catalog["releases"]["v1.0"]["runtime"]
    enrollers = release_runtime["voice_enrollers"]

    assert [item["id"] for item in enrollers] == ["inno-v0.2"]
    assert enrollers[0]["model_component"] == "inno_voicepack"
    assert enrollers[0]["metadata_component"] == "inno_tuner"
    assert "voice_enrollers" not in runtime
    assert registry["models"]["v1.0"]["distributions"][0]["release_version"] == 4
    assert runtime["layout"] == "single-onnx-v1"
    assert runtime["default_voice"] == "af_heart"
    assert len(runtime["voices"]) == 61
    for model in registry["models"].values():
        assert "voice_enrollers" not in model["runtime"]


def test_enroller_validation_rejects_duplicate_ids_and_unordered_durations(
    tmp_path: Path,
) -> None:
    import copy

    registry = load_registry()
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    enrollers = releases["releases"]["v1.0"]["runtime"]["voice_enrollers"]
    registry_path = tmp_path / "models.json"

    broken = copy.deepcopy(registry)
    duplicate_enrollers = copy.deepcopy(enrollers)
    duplicate_enrollers.append(dict(duplicate_enrollers[0]))
    broken["models"]["v1.0"]["runtime"]["voice_enrollers"] = duplicate_enrollers
    registry_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RegistryError, match="Duplicate enroller id"):
        verify_registry(registry_path)

    broken = copy.deepcopy(registry)
    invalid_enrollers = copy.deepcopy(enrollers)
    invalid_enrollers[0]["min_seconds"] = 9.0
    broken["models"]["v1.0"]["runtime"]["voice_enrollers"] = invalid_enrollers
    registry_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RegistryError, match="durations"):
        verify_registry(registry_path)


def test_enroller_validation_rejects_inno_on_incompatible_model(
    tmp_path: Path,
) -> None:
    registry = load_registry()
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    enrollers = releases["releases"]["v1.0"]["runtime"]["voice_enrollers"]
    registry["models"]["v1.1-zh"]["runtime"]["voice_enrollers"] = enrollers
    registry_path = tmp_path / "models.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(RegistryError, match="Kokoro v1.0"):
        verify_registry(registry_path)


def test_enroller_components_must_exist_in_current_distribution() -> None:
    from scripts.runtime_contracts import ContractError, validate_voice_enrollers

    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    enrollers = releases["releases"]["v1.0"]["runtime"]["voice_enrollers"]
    with pytest.raises(ContractError, match="model component"):
        validate_voice_enrollers(
            {"voice_enrollers": enrollers},
            model_components=set(),
            metadata_components={"inno_tuner"},
            model_id="v1.0",
            model_version="1.0",
        )
    with pytest.raises(ContractError, match="metadata component"):
        validate_voice_enrollers(
            {"voice_enrollers": enrollers},
            model_components={"inno_voicepack"},
            metadata_components=set(),
            model_id="v1.0",
            model_version="1.0",
        )
    validate_voice_enrollers(
        {"voice_enrollers": enrollers},
        model_components={"inno_voicepack"},
        metadata_components={"inno_tuner", "inno_tuner_config"},
        model_id="v1.0",
        model_version="1.0",
    )


@pytest.mark.parametrize(
    ("present_component", "missing_component_message"),
    [
        ("inno_tuner", "model component"),
        ("inno_voicepack", "metadata component"),
    ],
)
def test_stale_release_does_not_skip_enroller_artifact_validation(
    tmp_path: Path,
    present_component: str,
    missing_component_message: str,
) -> None:
    import copy

    registry = load_registry()
    releases_path = ROOT / "catalog" / "releases.json"
    releases = json.loads(releases_path.read_text(encoding="utf-8"))
    release = releases["releases"]["v1.0"]
    model = registry["models"]["v1.0"]
    model["runtime"]["voice_enrollers"] = release["runtime"]["voice_enrollers"]
    distribution = model["distributions"][0]
    assert release["release_version"] > distribution["release_version"]

    role = "model" if present_component == "inno_voicepack" else "metadata"
    artifact = {
        "id": f"{role}-{present_component}",
        "role": role,
        "url": f"https://github.com/example/release/{present_component}",
        "local_name": f"{present_component}.bin",
        "format": "onnx" if role == "model" else "numpy-npz",
        "size": 1,
        "sha256": "0" * 64,
        "component": present_component,
    }
    distribution["artifacts"].append(artifact)

    registry_path = tmp_path / "models.json"
    registry_path.write_text(json.dumps(copy.deepcopy(registry)), encoding="utf-8")
    with pytest.raises(RegistryError, match=missing_component_message):
        verify_registry(registry_path, releases_path=releases_path)


@pytest.mark.parametrize(
    ("duplicate_component", "expected_message"),
    [
        ("inno_voicepack", "exactly one model component"),
        ("inno_tuner", "exactly one metadata component"),
    ],
)
def test_enroller_support_components_must_be_unique(
    duplicate_component: str,
    expected_message: str,
) -> None:
    from scripts.runtime_contracts import (
        ContractError,
        validate_voice_enroller_artifacts,
    )

    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    enrollers = releases["releases"]["v1.0"]["runtime"]["voice_enrollers"]
    artifacts = [
        {"role": "model", "component": "inno_voicepack"},
        {"role": "metadata", "component": "inno_tuner"},
    ]
    artifacts.append(
        {
            "role": "model" if duplicate_component == "inno_voicepack" else "metadata",
            "component": duplicate_component,
        }
    )

    with pytest.raises(ContractError, match=expected_message):
        validate_voice_enroller_artifacts(
            {"voice_enrollers": enrollers},
            artifacts,
            model_id="v1.0",
            model_version="1.0",
        )
