from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = ROOT / "scripts" / "prepare_release.py"
PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_release", PREPARE_PATH)
assert PREPARE_SPEC is not None and PREPARE_SPEC.loader is not None
prepare_release = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(prepare_release)


def test_catalog_target_repo() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    assert data["target_repository"] == "buchwandler/kokoro-onnx-models"
    assert data["releases"]["v1.0"]["tag"] == "model-files-v1.0-timestamped-r5"
    assert data["releases"]["v1.1-zh"]["tag"] == "model-files-v1.1"


def test_release_entries_have_explicit_version_identity() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    for release in data["releases"].values():
        assert isinstance(release["tag"], str) and release["tag"]
        assert isinstance(release["model_version"], str) and release["model_version"]
        assert isinstance(release["release_version"], int)
        assert release["release_version"] >= 1
    assert data["releases"]["v1.0"]["release_version"] == 5


def test_v1_0_voice_asset_is_numpy_archive() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["v1.0"]
    voices = next(
        asset for asset in spec["assets"] if asset["name"] == "kokoro-v1.0.onnx"
    )

    assert voices["format"] == "onnx"
    assert (
        spec["source_repository"] == "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped"
    )
    assert spec["source_revision"] == "dd4401a9add81ac692d20e240d22ec9dda82cc29"
    assert spec["onnx_contract"]["outputs"] == {
        "waveform": "float32",
        "durations": "float32",
    }
    assert len(spec["runtime"]["voices"]) == 61
    assert spec["runtime"]["default_voice"] == "af_heart"
    metadata = spec["runtime"]["voice_metadata"]
    assert set(metadata) == set(spec["runtime"]["voices"])
    new_voices = {
        "af_ameliaearhart",
        "af_libritts5338",
        "am_libritts1272",
        "am_libritts6241",
        "am_vincentprice",
        "bf_janegoodall",
        "bm_davidattenborough",
    }
    assert new_voices <= set(metadata)
    assert metadata["af_ameliaearhart"] == {
        "gender": "female",
        "language": "en",
        "locale": "en-US",
        "language_label": "American English",
    }
    assert metadata["bf_janegoodall"]["locale"] == "en-GB"
    assert metadata["bm_davidattenborough"]["gender"] == "male"
    assert spec["voice_pack"]["target"] == "voices-v1.0.npz"
    assert spec["voice_pack"]["expected_count"] == 61
    assert spec["voice_pack"]["expected_rows"] == 510
    remsky = [
        item
        for item in spec["voice_pack"]["source_assets"]
        if item["name"]
        in {
            "af_ameliaearhart",
            "af_libritts5338",
            "am_libritts1272",
            "am_libritts6241",
            "am_vincentprice",
            "bf_janegoodall",
            "bm_davidattenborough",
        }
    ]
    assert len(remsky) == 7
    assert {item["repository"] for item in remsky} == {"remsky/kokoro-inno-clone-tuner"}
    assert {item["revision"] for item in remsky} == {
        "b8fc665a0a110d663b2cc9e313ec28bcf6bbbbdb"
    }
    assert {item["format"] for item in remsky} == {"torch-pt"}
    assert {item["name"] for item in remsky} == {
        "af_ameliaearhart",
        "af_libritts5338",
        "am_libritts1272",
        "am_libritts6241",
        "am_vincentprice",
        "bf_janegoodall",
        "bm_davidattenborough",
    }


def test_v1_1_zh_has_distinct_quality_matrix_and_voice_inventory() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["v1.1-zh"]
    qualities = {
        asset["quality"] for asset in spec["assets"] if asset["role"] == "model"
    }

    assert spec["source_repository"] == "onnx-community/Kokoro-82M-v1.1-zh-ONNX"
    assert len(spec["source_revision"]) == 40
    assert len(spec["runtime"]["voices"]) == 103
    assert qualities == {"fp32", "fp16", "q8", "int8", "q4", "q4f16", "uint8", "bnb4"}
    assert not {"q8f16", "uint8f16"} & qualities


def test_swedish_and_thorsten_release_metadata() -> None:
    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    catalog = json.loads((ROOT / "catalog" / "releases.json").read_text())
    swedish = profiles["sv-joakim"]
    thorsten = profiles["de-thorsten"]
    assert swedish["license"] == "Apache-2.0"
    assert swedish["voices"]["items"]["Björn"] == "voices/Björn.pt"
    assert swedish["release"]["default_voice"] == "Alice"
    assert swedish["postprocess"]["q"] == 35
    assert thorsten["license"] == "Apache-2.0"
    assert thorsten["model"]["path"] == "model.pth"
    assert (
        thorsten["model"]["sha256"]
        == "36dde15c4a800cfd1ab540ccb4476dbab604fe03ff7c937d976ebbf3b49e59ce"
    )
    assert (
        thorsten["model"]["config_sha256"]
        == "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"
    )
    assert thorsten["voices"]["items"] == {
        "thorsten": {
            "path": "voices/thorsten.pt",
            "sha256": "9d98b775ebce1cfc369e8f9a3ee8ee260cd612dffb477cba85749112362306d7",
        }
    }
    assert thorsten["release"]["default_voice"] == "thorsten"
    assert catalog["releases"]["sv-joakim"]["tag"] == "model-files-swedish-v1.1-r2"
    assert (
        catalog["releases"]["de-thorsten"]["tag"]
        == "model-files-german-thorsten-v1.1.4"
    )
    assert (
        catalog["releases"]["vi-ngoc-huyen"]["tag"]
        == "model-files-vietnamese-ngoc-huyen-v1.0-r2"
    )
    assert catalog["releases"]["vi-ngoc-huyen"]["release_version"] == 2
    assert (
        catalog["releases"]["vi-ngoc-huyen"]["source_repository"]
        == "dinhthuan/kokoro-vi-ngoc-huyen"
    )
    assert catalog["releases"]["vi-ngoc-huyen"]["onnx_contract"]["outputs"] == {
        "audio": "float32",
        "duration": "int64",
    }


def test_software_mansion_anna_release_is_runtime_activated() -> None:
    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    releases = json.loads(
        (ROOT / "catalog" / "releases.json").read_text(encoding="utf-8")
    )["releases"]
    release = releases["de-anna"]
    profile = profiles["de-anna"]

    assert release["kind"] == "build"
    assert release["profile"] == "de-anna"
    assert release["tag"] == "model-files-german-software-mansion-anna-v1"
    assert release["publish"] is True
    assert release["activate_runtime_registry"] is True
    assert release["frontend"] == "german-ipa-v1"
    assert release["source_repository"] == profile["repo_id"]
    assert release["source_revision"] == profile["revision"]
    assert release["runtime"]["default_voice"] == "df_anna"
    assert release["onnx_contract"] == profile["onnx_contract"]
    assert profile["frontend_id"] == "german-ipa-v1"
    assert profile["frontend"]["experimental"] is False


def test_software_mansion_mateusz_release_remains_staged() -> None:
    release = json.loads(
        (ROOT / "catalog" / "releases.json").read_text(encoding="utf-8")
    )["releases"]["pl-mateusz"]

    assert release["activate_runtime_registry"] is False
    assert release["frontend"] == "phonemis-pl-v1"


def test_build_profile_and_release_asset_names_match() -> None:
    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    catalog = json.loads((ROOT / "catalog" / "releases.json").read_text())
    profile_release = profiles["de-thorsten"]["release"]
    release = catalog["releases"]["de-thorsten"]

    assert "tag" not in profile_release
    assert "model_version" not in profile_release
    release_names = {asset["name"] for asset in release["assets"]}
    assert profile_release["model_filename"] in release_names
    assert profile_release["config_filename"] in release_names
    for voice in profile_release["voice_assets"]:
        assert voice["filename"] in release_names
    assert not any("thorsten-v1.0" in name for name in release_names)


def test_declared_build_release_metadata_matches_profiles() -> None:
    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    releases = json.loads(
        (ROOT / "catalog" / "releases.json").read_text(encoding="utf-8")
    )["releases"]

    for release_key, release in releases.items():
        if release.get("kind") != "build":
            continue
        profile = profiles[release.get("profile", release_key)]
        if "source_repository" in release:
            assert release["source_repository"] == profile["repo_id"]
        if "source_revision" in release:
            assert release["source_revision"] == profile["revision"]
        if "onnx_contract" in release:
            assert release["onnx_contract"] == profile["onnx_contract"]

    for release_key in ("ru-zaakirio-base", "ru-zaakirio-dima"):
        release = releases[release_key]
        profile = profiles[release["profile"]]
        assert release["source_repository"] == profile["repo_id"]
        assert release["source_revision"] == profile["revision"]
        assert release["onnx_contract"] == profile["onnx_contract"]


def test_thai_wayu_is_pinned_split_mirror() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["th-wayu"]
    assert spec["kind"] == "mirror"
    assert spec["source_type"] == "huggingface"
    assert spec["source_repository"] == "kunato/wayu-kokoro-thai-v1"
    assert spec["source_revision"] == "50d7f60e41ac118e5bb92b0ba52c30bb7830103c"
    assert spec["runtime_layout"] == "split-onnx-v1"
    assert len(spec["runtime"]["voices"]) == 12
    components = spec["onnx_contract"]["components"]
    assert set(components) == {"prosody", "curves", "decoder"}
    model_assets = [asset for asset in spec["assets"] if asset["role"] == "model"]
    assert {asset["component"] for asset in model_assets} == set(components)
    assert all(
        asset["size"] > 0 and len(asset["sha256"]) == 64 for asset in spec["assets"]
    )


def test_runtime_metadata_preserves_default_and_postprocess(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps({"speakers": [{"name": "Alice"}, {"name": "Björn"}]}),
        encoding="utf-8",
    )
    profile = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )["sv-joakim"]
    runtime = prepare_release._runtime_metadata(profile, bundle, profile["release"])
    assert runtime["default_voice"] == "Alice"
    assert runtime["voices"] == ["Alice", "Björn"]
    assert runtime["layout"] == "single-onnx-v1"
    assert runtime["postprocess"]["kind"] == "notch_filters"


def test_nabra_build_uses_prebuilt_onnx_source() -> None:
    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    profile = profiles["ar-nabra"]

    assert profile["repo_id"] == "marwanelamami/Nabra-82M-v0.1-ONNX"
    assert profile["model"]["kind"] == "onnx"
    assert profile["model"]["path"] == "nabra_fp32.onnx"
    assert "vocab.json" in profile["release"]["auxiliary_assets"][0]["source"]
    assert profile["frontend_id"] == "nabra-arabic-v1"


def test_nabra_release_spec_uses_r2_default_runtime_identity() -> None:
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())["releases"]
    release = releases["ar-nabra"]

    assert release["model_version"] == "0.1"
    assert release["release_version"] == 2
    assert release["tag"] == "model-files-arabic-nabra-v0.1-r2"
    assert release["runtime"] == {
        "default_voice": "default",
        "voices": ["default"],
    }


def test_hebrew_not_published_by_default() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    assert data["releases"]["he-hebrew-nc"]["publish"] is False


def test_martin_is_mirrored_from_godelaune() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["v1.2-de-martin"]

    assert spec["kind"] == "mirror"
    assert spec["source_type"] == "huggingface"
    assert spec["source_repository"] == "Godelaune/Kokoro-82M-ONNX-German-Martin"
    assert spec["source_revision"] == "main"
    assert spec["tag"] == "model-files-german-martin-v1.2"

    by_name = {item["name"]: item for item in spec["assets"]}
    assert by_name["kokoro-german-martin-v1.2.onnx"]["source"] == "kokoro-martin.onnx"
    assert by_name["voices-german-martin-v1.2.bin"]["source"] == "voices-martin.npz"


def test_nabra_release_includes_vocabulary_metadata(tmp_path, monkeypatch) -> None:
    build_dir = tmp_path / "build" / "ar-nabra"
    build_dir.mkdir(parents=True)
    (build_dir / "model.onnx").write_bytes(b"model")
    np.savez(build_dir / "voices.npz", default=np.zeros((510, 1, 256), dtype="<f4"))
    (build_dir / "voices.raw.bin").write_bytes(b"\x00" * (510 * 256 * 4))
    (build_dir / "bundle.json").write_text("{}\n", encoding="utf-8")
    (build_dir / "vocab.json").write_text('{"ʕ": 7, "ħ": 8}\n', encoding="utf-8")
    dist = tmp_path / "dist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_release.py",
            "ar-nabra",
            "--build-root",
            str(tmp_path / "build"),
            "--dist",
            str(dist),
        ],
    )

    assert prepare_release.main() == 0

    manifest_path = dist / "model-files-arabic-nabra-v0.1-r2" / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_version"] == "0.1"
    assert manifest["release_version"] == 2
    assert manifest["tag"] == "model-files-arabic-nabra-v0.1-r2"
    assert manifest["runtime"]["frontend"] == "nabra-arabic-v1"
    assert manifest["runtime"]["default_voice"] == "default"
    assert manifest["runtime"]["voices"] == ["default"]
    vocab = next(
        asset
        for asset in manifest["assets"]
        if asset["name"] == "vocab-arabic-nabra-v0.1.json"
    )
    assert vocab["role"] == "vocab"
    assert vocab["format"] == "json"
    assert (
        vocab["size"]
        == (dist / "model-files-arabic-nabra-v0.1-r2" / vocab["name"]).stat().st_size
    )


def test_german_v1_1_and_holgern_are_retired() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "MODEL_LICENSES.md",
        ROOT / "catalog" / "releases.json",
        ROOT / "docs" / "LOCAL_PYKOKORO_PRE_RELEASE_TESTING.md",
        ROOT / "docs" / "PYKOKORO_MIGRATION.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "v1.1-de" not in text
    assert "holgern/kokoro-onnx-model" not in text


def test_release_catalog_contains_russian_checkpoint_builds() -> None:
    catalog = json.loads(
        (ROOT / "catalog" / "releases.json").read_text(encoding="utf-8")
    )
    releases = catalog["releases"]
    for key, tag, voices in (
        (
            "ru-zaakirio-base",
            "model-files-russian-zaakirio-base-v2",
            ["sveta", "masha"],
        ),
        ("ru-zaakirio-dima", "model-files-russian-zaakirio-dima-v2", ["dima"]),
    ):
        release = releases[key]
        assert release["kind"] == "build"
        assert release["publish"] is True
        assert release["tag"] == tag
        assert release["source_repository"] == "zaakirio/kokoro-ru"
        assert release["source_revision"] == "d649c57b239b18c4c384378127cbf01dba039bc1"
        assert release["runtime"]["voices"] == voices
        assert release["onnx_contract"]["outputs"] == {
            "audio": "float32",
            "duration": "int64",
        }
        assert release["onnx_contract"]["timing"]["output"] == "duration"


def test_v1_sources_and_registry_provenance_are_current() -> None:
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    models = json.loads((ROOT / "catalog" / "models.json").read_text())

    for key, repository, revision in (
        (
            "v1.0",
            "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped",
            "dd4401a9add81ac692d20e240d22ec9dda82cc29",
        ),
        (
            "v1.1-zh",
            "onnx-community/Kokoro-82M-v1.1-zh-ONNX",
            "6cc0f0d2ebe369a68b0df87c2b65c1af8c0ac3e3",
        ),
    ):
        assert releases["releases"][key]["source_repository"] == repository
        assert releases["releases"][key]["source_revision"] == revision
        assert models["models"][key]["license"]["source_repository"] == repository
        provenance = models["models"][key]["distributions"][0]["provenance"]
        assert provenance["source_repository"] == repository
        assert provenance["source_revision"] == revision

    text = (ROOT / "catalog" / "models.json").read_text()
    assert "thewh1teagle/kokoro-onnx" not in text


def test_contextbox_release_is_reproducible_r2() -> None:
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    profiles = json.loads((ROOT / "scripts" / "kokoro_profiles.json").read_text())

    release = releases["releases"]["vi-contextbox"]
    profile = profiles["vi-contextbox"]

    assert release["model_version"] == "1.0"
    assert release["release_version"] == 2
    assert release["tag"] == "model-files-vietnamese-v1.0-r2"
    assert release["source_repository"] == profile["repo_id"]
    assert release["source_revision"] == profile["revision"]
    assert release["frontend"] == "vig2p-v1"


def test_anphunl_release_is_reproducible_r2() -> None:
    releases = json.loads((ROOT / "catalog" / "releases.json").read_text())
    profiles = json.loads((ROOT / "scripts" / "kokoro_profiles.json").read_text())
    models = json.loads((ROOT / "catalog" / "models.json").read_text())

    release = releases["releases"]["vi-anphunl"]
    profile = profiles["vi-anphunl"]
    model = models["models"]["vi-anphunl"]

    assert release["model_version"] == "1.0"
    assert release["release_version"] == 2
    assert release["tag"] == "model-files-vietnamese-anphunl-v1.0-r2"
    assert release["source_repository"] == profile["repo_id"]
    assert release["source_revision"] == profile["revision"]
    assert release["frontend"] == "vig2p-v1"
    assert model["license"]["source_repository"] == profile["repo_id"]


def test_runtime_metadata_uses_bundle_voice_as_implicit_default(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "speakers": [
                    {"sid": 0, "name": "diem_trinh"},
                    {"sid": 1, "name": "hung_thinh"},
                ]
            }
        ),
        encoding="utf-8",
    )

    runtime = prepare_release._runtime_metadata(
        {"language": "vi", "sample_rate": 24000, "frontend": {"name": "vig2p"}},
        bundle,
        {},
    )

    assert runtime["voices"] == ["diem_trinh", "hung_thinh"]
    assert runtime["default_voice"] == "diem_trinh"


def test_runtime_metadata_rejects_default_outside_bundle_roster(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps({"speakers": [{"sid": 0, "name": "diem_trinh"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="not in the voice roster"):
        prepare_release._runtime_metadata(
            {"language": "vi", "sample_rate": 24000, "frontend": {"name": "vig2p"}},
            bundle,
            {"default_voice": "default"},
        )


def _reference_profile() -> dict:
    return {
        "display_name": "Cloning",
        "repo_id": "org/model",
        "revision": "a" * 40,
        "license": "Apache-2.0",
        "language": "en",
        "sample_rate": 24000,
        "frontend_id": "pykokoro-native-v1",
        "onnx_contract": {
            "inputs": {"reference": "reference-conditioned"},
            "outputs": {"audio": "float32"},
            "max_tokens": 510,
            "components": {
                "decoder": {
                    "inputs": {"har": "float32"},
                    "outputs": {"audio": "float32"},
                }
            },
        },
        "release": {
            "config_filename": "config-cloning-v1.json",
            "runtime": {
                "layout": "cloning-onnx-v1",
                "voice_mode": "reference",
                "max_tokens": 510,
                "speed_supported": False,
                "style_dimensions": {"acoustic": 128, "duration": 128},
                "reference": {
                    "format": "akinvox-cloning-reference-v1",
                    "sample_rate": 24000,
                    "identity_sample_rate": 16000,
                    "min_seconds": 3.0,
                    "max_seconds": 30.0,
                    "memory_width": 192,
                    "style_width": 256,
                },
            },
            "model_assets": [
                {
                    "source": "decoder.onnx",
                    "filename": "decoder-cloning-v1.onnx",
                    "component": "decoder",
                    "quality": "fp32",
                    "format": "onnx",
                }
            ],
            "auxiliary_assets": [
                {
                    "source": "source-params.npz",
                    "filename": "source-params-cloning-v1.npz",
                    "role": "metadata",
                    "component": "source_params",
                    "format": "numpy-npz",
                }
            ],
        },
    }


def _run_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profiles: dict,
    releases: dict,
    profile_key: str,
) -> Path:
    build = tmp_path / "build" / profile_key
    build.mkdir(parents=True)
    (build / "bundle.json").write_text(
        json.dumps(
            {
                "exporter": {
                    "outputs": ["audio", "duration"],
                    "timing": {"output": "duration", "validated": True},
                }
            }
        ),
        encoding="utf-8",
    )
    (build / "decoder.onnx").write_bytes(b"onnx")
    (build / "model.onnx").write_bytes(b"onnx")
    (build / "voices.bin").write_bytes(b"voices")
    (build / "config.json").write_text("{}", encoding="utf-8")
    np.savez(
        build / "source-params.npz",
        weight=np.zeros((1, 9), dtype=np.float32),
        bias=np.zeros(1, dtype=np.float32),
        window=np.zeros(20, dtype=np.float32),
    )
    profiles_path = tmp_path / "profiles.json"
    releases_path = tmp_path / "releases.json"
    profiles_path.write_text(json.dumps(profiles), encoding="utf-8")
    releases_path.write_text(json.dumps({"releases": releases}), encoding="utf-8")
    monkeypatch.setattr(prepare_release, "PROFILES", profiles_path)
    monkeypatch.setattr(prepare_release, "RELEASES", releases_path)
    dist = tmp_path / "dist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_release.py",
            profile_key,
            "--build-root",
            str(tmp_path / "build"),
            "--dist",
            str(dist),
        ],
    )
    assert prepare_release.main() == 0
    return dist / releases[profile_key]["tag"]


def test_reference_release_packages_components_without_voices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = {"cloning": _reference_profile()}
    releases = {
        "cloning": {
            "kind": "build",
            "profile": "cloning",
            "tag": "model-files-cloning",
            "model_version": "1.0",
            "release_version": 1,
        }
    }
    out = _run_prepare(tmp_path, monkeypatch, profiles, releases, "cloning")

    manifest = json.loads((out / "release-manifest.json").read_text())
    runtime = manifest["runtime"]
    assert runtime["voice_mode"] == "reference"
    assert "voices" not in runtime
    assert "default_voice" not in runtime
    model_assets = [a for a in manifest["assets"] if a["role"] == "model"]
    assert [a["component"] for a in model_assets] == ["decoder"]
    assert model_assets[0]["quality"] == "fp32"
    support = [a for a in manifest["assets"] if a.get("component") == "source_params"]
    assert [a["role"] for a in support] == ["metadata"]
    assert not any(a["role"] == "voices" for a in manifest["assets"])
    assert "Voice mode: reference enrollment" in (out / "release-notes.md").read_text()
    assert not (out / "voices.bin").exists()


def test_legacy_single_model_release_keeps_static_voice_packaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = {
        "display_name": "Legacy",
        "repo_id": "org/legacy",
        "revision": "b" * 40,
        "license": "Apache-2.0",
        "language": "de",
        "sample_rate": 24000,
        "frontend_id": "pykokoro-native-v1",
        "model": {"kind": "checkpoint"},
        "onnx_contract": {
            "inputs": {"tokens": "int64", "style": "float32", "speed": "float32"},
            "outputs": {"audio": "float32", "duration": "int64"},
            "timing": {
                "kind": "token-duration-v1",
                "output": "duration",
                "unit": "frame",
                "samples_per_frame": 600,
                "includes_boundary_tokens": True,
            },
            "max_tokens": 510,
        },
        "release": {
            "model_filename": "legacy-v1.onnx",
            "voices_filename": "voices-legacy-v1.bin",
            "config_filename": "config-legacy-v1.json",
        },
    }
    profiles = {"legacy": profile}
    releases = {
        "legacy": {
            "kind": "build",
            "profile": "legacy",
            "tag": "model-files-legacy",
            "model_version": "1.0",
            "release_version": 1,
        }
    }
    out = _run_prepare(tmp_path, monkeypatch, profiles, releases, "legacy")

    manifest = json.loads((out / "release-manifest.json").read_text())
    model_assets = [a for a in manifest["assets"] if a["role"] == "model"]
    assert [a["name"] for a in model_assets] == ["legacy-v1.onnx"]
    assert "component" not in model_assets[0]
    assert [a["role"] for a in manifest["assets"] if a["role"] == "voices"] == [
        "voices"
    ]
    assert manifest["runtime"]["voices"] == ["default"]
    assert "Voices: default" in (out / "release-notes.md").read_text()


def test_v1_0_release_declares_inno_enroller_and_pinned_augmentation() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["v1.0"]
    assert spec["release_version"] == 5
    enrollers = spec["runtime"]["voice_enrollers"]
    assert [item["id"] for item in enrollers] == ["inno-v0.2"]
    assert enrollers[0] == {
        "id": "inno-v0.2",
        "kind": "kokoro-voicepack-tuner",
        "input": "reference-audio",
        "transcript_required": False,
        "min_seconds": 3.0,
        "recommended_seconds": 5.0,
        "max_seconds": 30.0,
        "output": {
            "format": "kokoro-voicepack-v1",
            "shape": [510, 1, 256],
            "dtype": "float32",
        },
        "model_component": "inno_voicepack",
        "metadata_component": "inno_tuner",
    }
    assert spec["onnx_contract"]["components"]["inno_voicepack"] == {
        "inputs": {
            "fbank": "float32",
            "tilt": "float32",
            "head_stats": "float32",
            "blend_weights": "float32",
        },
        "outputs": {"voicepack": "float32"},
    }

    augmentation = spec["augmentations"][0]
    assert augmentation["id"] == "inno-v0.2"
    assert augmentation["declared_version"] == "0.2.0"
    assert augmentation["source_repository"] == "remsky/kokoro-inno-clone-tuner"
    assert augmentation["source_revision"] == (
        "429617d18ce4d637acea948bdff4cce3ec6cf167"
    )
    assert augmentation["code_repository"] == "remsky/inno-kokoro"
    assert augmentation["code_revision"] == "892ef184bc932aa3ff9d72c1509d5b81ff6941e6"
    assert augmentation["source_files"]["model.safetensors"] == {
        "size": 23776728,
        "sha256": "71cb8e93544f697043197f27fa7f13f0f9f9161076b092aaff6d0894fec2b0e8",
    }
    assert augmentation["source_files"]["config.json"]["sha256"] == (
        "319569366235a0c7fed8c34188edb6b713b53f1d56a1334196e8170800f873c8"
    )
    by_component = {asset["component"]: asset for asset in augmentation["assets"]}
    assert set(by_component) == {"inno_voicepack", "inno_tuner", "inno_tuner_config"}
    assert by_component["inno_voicepack"]["role"] == "model"
    assert by_component["inno_voicepack"]["format"] == "onnx"
    assert "quality" not in by_component["inno_voicepack"]
    assert by_component["inno_tuner"]["role"] == "metadata"
    assert by_component["inno_tuner"]["format"] == "numpy-npz"
    assert by_component["inno_tuner_config"]["role"] == "metadata"
    assert by_component["inno_tuner_config"]["format"] == "json"

    assert spec["runtime"]["default_voice"] == "af_heart"
    assert len(spec["runtime"]["voices"]) == 61
    assert spec["voice_pack"]["expected_count"] == 61


def test_v1_0_release_requires_no_pytorch_runtime_artifacts() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["v1.0"]
    formats = {asset["format"] for asset in spec["assets"]}
    formats |= {asset["format"] for asset in spec["augmentations"][0]["assets"]}
    assert formats <= {"onnx", "json", "numpy-npz"}
    registry = json.loads((ROOT / "catalog" / "models.json").read_text())
    artifacts = registry["models"]["v1.0"]["distributions"][0]["artifacts"]
    assert not any(
        artifact["local_name"].endswith((".pt", ".pth", ".safetensors"))
        for artifact in artifacts
    )
    assert {artifact["format"] for artifact in artifacts} <= {
        "onnx",
        "json",
        "numpy-npz",
    }
