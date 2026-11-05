from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import (
    akinvox_multispeaker,
    build_kokoro,
    mirror_release,
    prepare_release,
    runtime_contracts,
    update_registry_from_release,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "akinvox-multispeaker-v1"
ENGLISH_VOICES = [
    "AV_ariala",
    "AV_brody",
    "AV_david",
    "AV_elliott",
    "AV_ralph",
    "AV_cori",
    "AV_phil",
    "AV_helen",
]
GERMAN_VOICES = [
    "AV_thorsten",
    "AV_bernd",
    "AV_eva_k",
    "AV_karlsson",
    "AV_kerstin",
    "AV_mucky",
    "AV_hokuspokus",
    "AV_julia",
]


def test_pinned_download_isolates_identical_filenames_by_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = {
        "source/catalog": b'{"format":"catalog"}',
        "base/repository": b'{"format":"base config"}',
    }

    def fake_hf_download(
        repo_id: str, filename: str, revision: str, cache_dir: Path
    ) -> Path:
        path = cache_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[repo_id])
        return path

    monkeypatch.setattr(akinvox_multispeaker, "hf_download", fake_hf_download)
    catalog = akinvox_multispeaker._pinned_download(
        "source/catalog",
        "config.json",
        "revision",
        tmp_path,
        hashlib.sha256(contents["source/catalog"]).hexdigest(),
        "source catalog",
    )
    base_config = akinvox_multispeaker._pinned_download(
        "base/repository",
        "config.json",
        "revision",
        tmp_path,
        hashlib.sha256(contents["base/repository"]).hexdigest(),
        "base config",
    )

    assert catalog != base_config
    assert catalog.read_bytes() == contents["source/catalog"]
    assert base_config.read_bytes() == contents["base/repository"]


def test_profile_and_release_pin_the_multispeaker_target() -> None:
    profile = build_kokoro.load_profiles()[MODEL_ID]
    releases = json.loads((ROOT / "catalog/releases.json").read_text())
    release = releases["releases"][MODEL_ID]

    assert profile["repo_id"] == "AKinvox/kokoro-multispeaker"
    assert profile["revision"] == akinvox_multispeaker.SOURCE_REVISION
    assert profile["model_version"] == "2026.10.07"
    assert profile["model"]["base"]["revision"] == akinvox_multispeaker.BASE_REVISION
    assert profile["model"]["kind"] == "akinvox-multispeaker"
    assert list(profile["model"]["languages"]) == ["en", "de"]
    assert list(profile["voices"]["items"]) == ENGLISH_VOICES + GERMAN_VOICES
    assert release["release_version"] == 1
    assert release["model_version"] == "2026.10.07"
    assert release["source_revision"] == akinvox_multispeaker.SOURCE_REVISION
    assert release["activate_runtime_registry"] is False
    assert release["runtime"] == profile["release"]["runtime"]
    assert [asset["component"] for asset in release["model_assets"]] == [
        "model_en",
        "model_de",
    ]


def test_language_router_metadata_survives_release_and_registry_paths(
    tmp_path: Path,
) -> None:
    profile = build_kokoro.load_profiles()[MODEL_ID]
    release = json.loads((ROOT / "catalog/releases.json").read_text())["releases"][
        MODEL_ID
    ]
    prepared = prepare_release._runtime_metadata(
        profile, tmp_path / "missing-bundle.json", release
    )
    mirrored = mirror_release._runtime(release)

    for runtime in (prepared, mirrored):
        assert runtime["language_codes"] == ["en", "de"]
        assert runtime["language_models"] == {"en": "model_en", "de": "model_de"}
        runtime_contracts.validate_language_router_constraints(runtime)

    manifest = {"runtime": prepared, "onnx_contract": profile["onnx_contract"]}
    model = update_registry_from_release._new_model_from_release(
        MODEL_ID, manifest, release, profile["onnx_contract"]
    )
    assert model["runtime_available"] is False
    assert model["runtime"]["language_codes"] == ["en", "de"]
    assert model["runtime"]["language_models"] == {"en": "model_en", "de": "model_de"}

    update_registry_from_release._sync_runtime_identity(model, manifest, release)
    assert model["language_codes"] == ["en", "de"]
    assert model["runtime"]["language_codes"] == ["en", "de"]
    assert model["runtime"]["language_models"] == {"en": "model_en", "de": "model_de"}


def test_parity_pairs_use_language_specific_fixtures_and_voices() -> None:
    cases = build_kokoro.load_profiles()[MODEL_ID]["export_validation"]["cases"]

    for language, voices, prefix in (
        ("en", ENGLISH_VOICES, "english-"),
        ("de", GERMAN_VOICES, "german-"),
    ):
        language_cases = akinvox_multispeaker._language_cases(language, cases)
        pairs = akinvox_multispeaker._selected_parity_pairs(
            language, language_cases, voices
        )
        assert len(pairs) == 5
        assert all(case["name"].startswith(prefix) for case, _ in pairs)
        assert [voice for _, voice in pairs] == [
            voices[0],
            voices[1],
            voices[0],
            voices[1],
            voices[0],
        ]
        assert [case["speed"] for case, _ in pairs][-1] != 1.0


def test_language_router_runtime_is_strict_and_voice_scoped() -> None:
    profile = build_kokoro.load_profiles()[MODEL_ID]
    runtime = profile["release"]["runtime"]

    assert akinvox_multispeaker._runtime_contract(profile) == runtime
    runtime_contracts.validate_language_router_constraints(runtime)

    invalid_mapping = {
        **runtime,
        "language_models": {"en": "model_de", "de": "model_en"},
    }
    with pytest.raises(runtime_contracts.ContractError, match="language_models"):
        runtime_contracts.validate_language_router_constraints(invalid_mapping)

    invalid_roster = {**runtime, "voices": [*runtime["voices"], "AV_unknown"]}
    with pytest.raises(runtime_contracts.ContractError, match="voice roster"):
        runtime_contracts.validate_language_router_constraints(invalid_roster)

    invalid_metadata = {
        **runtime,
        "voice_metadata": {
            **runtime["voice_metadata"],
            "AV_mucky": {
                **runtime["voice_metadata"]["AV_mucky"],
                "language": "en",
            },
        },
    }
    with pytest.raises(runtime_contracts.ContractError, match="AV_mucky"):
        runtime_contracts.validate_language_router_constraints(invalid_metadata)


def test_adapter_fixture_records_exact_audited_tensor_inventory() -> None:
    fixture = json.loads(
        (
            ROOT / "tests/fixtures/akinvox_multispeaker_adapter_inventory.json"
        ).read_text()
    )

    assert fixture["format"] == "akinvox-adapter-structural-inventory/v1"
    assert fixture["tensor_count"] == 403
    assert len(fixture["tensors"]) == fixture["tensor_count"]
    assert fixture["tensors"] == sorted(fixture["tensors"], key=lambda item: item[0])
    assert fixture["tensors"][-1] == ["voicebank", [8, 510, 256], "float32"]

    for language, voices, task, update in (
        ("en", ENGLISH_VOICES, "en_multispeaker", 185000),
        ("de", GERMAN_VOICES, "de_multispeaker", 156254),
    ):
        adapter = fixture["languages"][language]
        assert adapter["root_type"] == "dict"
        assert adapter["speaker_names"] == voices
        assert adapter["task"] == task
        assert adapter["update"] == update
        assert adapter["target_count"] == 201
        assert adapter["state_tensor_count"] == 402
        assert (
            adapter["base_sha256"]
            == "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
        )


def test_language_router_schemas_require_both_graphs_and_exact_rosters() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    profile = build_kokoro.load_profiles()[MODEL_ID]
    release = json.loads((ROOT / "catalog/releases.json").read_text())["releases"][
        MODEL_ID
    ]
    schema = json.loads((ROOT / "schemas/release-manifest.schema.json").read_text())
    runtime = {
        **release["runtime"],
        "sample_rate": 24000,
        "frontend": profile["frontend_id"],
        "frontend_experimental": False,
    }
    manifest = {
        "schema": 2,
        "runtime_contract": 1,
        "repository": "buchwandler/kokoro-onnx-models",
        "tag": release["tag"],
        "profile": MODEL_ID,
        "model_version": "2026.10.07",
        "release_version": 1,
        "generated_at": "2026-10-07T00:00:00Z",
        "source": {
            "type": "huggingface",
            "repository": profile["repo_id"],
            "revision": profile["revision"],
        },
        "license": release["license"],
        "publication": {"enabled": True},
        "runtime": runtime,
        "onnx_contract": profile["onnx_contract"],
        "assets": [
            {
                "name": "model-en.onnx",
                "role": "model",
                "component": "model_en",
                "quality": "fp32",
                "format": "onnx",
                "size": 1,
                "sha256": "0" * 64,
            }
        ],
    }
    validator = jsonschema.Draft202012Validator(schema)
    assert list(validator.iter_errors(manifest)) == []

    malformed = json.loads(json.dumps(manifest))
    malformed["runtime"]["language_models"]["de"] = "model_extra"
    assert list(validator.iter_errors(malformed))

    registry_schema = json.loads(
        (ROOT / "schemas/model-registry.schema.json").read_text()
    )
    registry_model = {
        "model_version": "2026.10.07",
        "language_codes": ["en", "de"],
        "frontend": "akinvox-multispeaker-v1",
        "sample_rate": 24000,
        "runtime": {
            "layout": runtime["layout"],
            "voice_mode": runtime["voice_mode"],
            "language_codes": runtime["language_codes"],
            "language_models": runtime["language_models"],
            "max_tokens": runtime["max_tokens"],
            "default_voice": runtime["default_voice"],
            "voices": runtime["voices"],
            "voice_metadata": runtime["voice_metadata"],
        },
        "onnx_contract": profile["onnx_contract"],
        "license": {
            "declared": release["license"],
            "redistribution": "allowed-with-upstream-terms",
        },
        "mirror_policy": "optional",
        "distributions": [],
    }
    registry_validator = jsonschema.Draft202012Validator(registry_schema)
    registry = {
        "schema": 1,
        "runtime_contract": 1,
        "models": {MODEL_ID: registry_model},
    }
    assert list(registry_validator.iter_errors(registry)) == []
    malformed_registry = json.loads(json.dumps(registry))
    malformed_registry["models"][MODEL_ID]["runtime"]["language_models"]["de"] = (
        "model_extra"
    )
    assert list(registry_validator.iter_errors(malformed_registry))


def test_executable_python_does_not_import_upstream_adapter_package() -> None:
    python_paths = [
        *ROOT.joinpath("scripts").rglob("*.py"),
        *ROOT.joinpath("local_test").rglob("*.py"),
        *ROOT.joinpath("tests").rglob("*.py"),
    ]
    forbidden: list[str] = []
    for path in python_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue
            if any(
                module.split(".", 1)[0] == "kokoro_multispeaker" for module in modules
            ):
                forbidden.append(path.relative_to(ROOT).as_posix())

    assert forbidden == []
    assert "kokoro-multispeaker" not in (ROOT / "pyproject.toml").read_text().lower()
