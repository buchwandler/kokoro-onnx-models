from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load("runtime_contracts", "scripts/runtime_contracts.py")
akinvox = _load("akinvox_cloning", "scripts/akinvox_cloning.py")
prepare_release = _load("prepare_release", "scripts/prepare_release.py")
build_kokoro = _load("build_kokoro", "scripts/build_kokoro.py")
verify_candidate = _load("verify_candidate", "scripts/verify_candidate.py")

PROFILE = json.loads((ROOT / "scripts" / "kokoro_profiles.json").read_text())[
    "en-akinvox-cloning-v1"
]
RELEASES = json.loads((ROOT / "catalog" / "releases.json").read_text())["releases"]
MODELS = json.loads((ROOT / "catalog" / "models.json").read_text())["models"]


# --------------------------------------------------------------------------
# Pins and contracts
# --------------------------------------------------------------------------


def test_profile_pins_every_upstream_source() -> None:
    akinvox.validate_source_pins(PROFILE)
    pins = akinvox.pinned_sources(PROFILE)

    assert set(pins) == {"akinvox", "kokoro_base", "wavlm", "source_code"}
    assert PROFILE["revision"] == "0094666f0a9038ce49789446eb8dd3ddfc848b43"
    assert pins["akinvox"]["release_tag"] == "v1.0.1"
    assert set(pins["akinvox"]["files"]) == {
        "adapter.pt",
        "reference_mapper.pt",
        "reference_encoders.pt",
        "config.json",
    }
    assert pins["kokoro_base"]["repo_id"] == "hexgrad/Kokoro-82M"
    assert pins["wavlm"]["repo_id"] == "microsoft/wavlm-base-plus-sv"
    assert pins["source_code"]["tag"] == "v1.0.1"
    for pin in ("akinvox", "kokoro_base", "wavlm"):
        assert len(pins[pin]["revision"]) == 40


def test_validate_source_pins_rejects_moving_or_unpinned_sources() -> None:
    mutable = json.loads(json.dumps(PROFILE))
    mutable["revision"] = "main"
    with pytest.raises(akinvox.AkinvoxBuildError, match="commit SHA"):
        akinvox.validate_source_pins(mutable)

    unpinned = json.loads(json.dumps(PROFILE))
    del unpinned["model"]["pins"]["wavlm"]["files"]["pytorch_model.bin"]
    with pytest.raises(akinvox.AkinvoxBuildError, match="WavLM pin is missing"):
        akinvox.validate_source_pins(unpinned)

    unchecked = json.loads(json.dumps(PROFILE))
    del unchecked["model"]["pins"]["akinvox"]["files"]["adapter.pt"]["sha256"]
    with pytest.raises(akinvox.AkinvoxBuildError, match="lowercase SHA-256"):
        akinvox.validate_source_pins(unchecked)


def test_component_contract_declares_all_six_graphs() -> None:
    contract = akinvox.component_contract()

    assert akinvox.COMPONENTS == (
        "reference_wavlm",
        "reference_encoders",
        "reference_mapper",
        "prosody",
        "curves",
        "decoder",
    )
    assert set(contract["components"]) == set(akinvox.COMPONENTS)
    assert set(akinvox.COMPONENT_CONTRACTS) == set(akinvox.COMPONENTS)
    for component, spec in akinvox.COMPONENT_CONTRACTS.items():
        assert set(spec["inputs"]) == set(contract["components"][component]["inputs"])
        assert set(spec["outputs"]) == set(contract["components"][component]["outputs"])


def test_prosody_contract_has_no_speed_input() -> None:
    assert "speed" not in akinvox.COMPONENT_CONTRACTS["prosody"]["inputs"]
    assert set(akinvox.COMPONENT_CONTRACTS["prosody"]["outputs"]) == {
        "pred_dur",
        "d",
        "t_en",
    }


def test_reference_mapper_derives_mask_and_rows_from_length_inputs() -> None:
    inputs = akinvox.COMPONENT_CONTRACTS["reference_mapper"]["inputs"]
    assert inputs["reference_lengths"] == "int64"
    assert inputs["mel_lengths"] == "int64"
    assert "phone_mask" not in inputs
    assert "rows" not in inputs


def test_runtime_metadata_declares_reference_enrollment_without_voices() -> None:
    runtime = akinvox.runtime_metadata()

    assert runtime["layout"] == "cloning-onnx-v1"
    assert runtime["voice_mode"] == "reference"
    assert runtime["max_tokens"] == 510
    assert runtime["speed_supported"] is False
    assert runtime["style_dimensions"] == {"acoustic": 128, "duration": 128}
    assert runtime["reference"] == {
        "format": "akinvox-cloning-reference-v1",
        "sample_rate": 24000,
        "identity_sample_rate": 16000,
        "min_seconds": 3.0,
        "max_seconds": 30.0,
        "memory_width": 192,
        "style_width": 256,
    }
    assert "voices" not in runtime
    assert "default_voice" not in runtime
    contracts.validate_reference_constraints(runtime)


def test_build_and_published_filenames_cover_the_same_assets() -> None:
    built = akinvox.build_filenames()
    published = akinvox.component_filenames()

    assert (
        set(built)
        == set(published)
        == set(akinvox.COMPONENTS)
        | {
            "source_params",
            "config",
        }
    )
    assert built["source_params"] == "source-params.npz"
    assert published["source_params"].endswith("-v1.0.1.npz")
    assert published["config"].endswith("-v1.0.1.json")


# --------------------------------------------------------------------------
# Source parameters
# --------------------------------------------------------------------------


def test_source_params_metadata_records_shapes_and_hashes(tmp_path: Path) -> None:
    arrays = {
        "weight": np.zeros((1, 9), dtype=np.float32),
        "bias": np.zeros(1, dtype=np.float32),
        "window": np.zeros(20, dtype=np.float32),
    }
    path = tmp_path / "source-params.npz"
    np.savez(path, **arrays)

    metadata = akinvox.source_params_metadata(arrays, path)

    assert metadata["format"] == "numpy-npz"
    assert metadata["allow_pickle"] is False
    assert len(metadata["sha256"]) == 64
    assert metadata["arrays"]["weight"]["shape"] == [1, 9]
    assert metadata["arrays"]["window"]["shape"] == [20]


def test_source_params_metadata_rejects_wrong_shapes() -> None:
    path = Path("unused.npz")
    with pytest.raises(akinvox.AkinvoxBuildError, match="harmonics"):
        akinvox.source_params_metadata(
            {
                "weight": np.zeros((1, 3), dtype=np.float32),
                "bias": np.zeros(1, dtype=np.float32),
                "window": np.zeros(20, dtype=np.float32),
            },
            path,
        )
    with pytest.raises(akinvox.AkinvoxBuildError, match="20 values"):
        akinvox.source_params_metadata(
            {
                "weight": np.zeros((1, 9), dtype=np.float32),
                "bias": np.zeros(1, dtype=np.float32),
                "window": np.zeros(19, dtype=np.float32),
            },
            path,
        )
    with pytest.raises(akinvox.AkinvoxBuildError, match="float32"):
        akinvox.source_params_metadata(
            {
                "weight": np.zeros((1, 9), dtype=np.float64),
                "bias": np.zeros(1, dtype=np.float32),
                "window": np.zeros(20, dtype=np.float32),
            },
            path,
        )


def test_source_parameter_archive_is_pickle_free(tmp_path: Path) -> None:
    path = tmp_path / "source-params.npz"
    np.savez(
        path,
        weight=np.ones((1, 9), dtype=np.float32),
        bias=np.zeros(1, dtype=np.float32),
        window=np.ones(20, dtype=np.float32),
    )
    with np.load(path, allow_pickle=False) as archive:
        assert sorted(archive.files) == ["bias", "weight", "window"]


# --------------------------------------------------------------------------
# Bundle assembly
# --------------------------------------------------------------------------


def _stub_exporter(out_dir: Path):
    def export(profile, target, *, cache_dir, opset, run_checker):
        target.mkdir(parents=True, exist_ok=True)
        records = []
        for component in akinvox.COMPONENTS:
            path = target / akinvox.build_filenames()[component]
            path.write_bytes(b"onnx-placeholder-" + component.encode())
            records.append(
                {
                    "component": component,
                    "path": path.name,
                    "sha256": akinvox.sha256(path),
                    "size": path.stat().st_size,
                }
            )
        np.savez(
            target / akinvox.SOURCE_PARAMS_FILENAME,
            weight=np.ones((1, 9), dtype=np.float32),
            bias=np.zeros(1, dtype=np.float32),
            window=np.ones(20, dtype=np.float32),
        )
        (target / "config.json").write_text("{}", encoding="utf-8")
        return {
            "source_params": akinvox.source_params_metadata(
                {
                    "weight": np.ones((1, 9), dtype=np.float32),
                    "bias": np.zeros(1, dtype=np.float32),
                    "window": np.ones(20, dtype=np.float32),
                },
                target / akinvox.SOURCE_PARAMS_FILENAME,
            ),
            "exporter": {"components": records},
        }

    return export


def test_build_emits_all_six_components_and_source_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(akinvox, "export_bundle", _stub_exporter(tmp_path))
    out = akinvox.build_akinvox_profile(
        "en-akinvox-cloning-v1",
        PROFILE,
        tmp_path / "build",
        opset=17,
        cache_dir=tmp_path / "cache",
        run_checker=False,
    )

    bundle = json.loads((out / "bundle.json").read_text())
    assert {item["component"] for item in bundle["components"]} == set(
        akinvox.COMPONENTS
    )
    assert all(item["quality"] == "fp32" for item in bundle["components"])
    assert all(item["format"] == "onnx" for item in bundle["components"])
    assert bundle["source_params"]["format"] == "numpy-npz"
    assert bundle["source_artifacts"] == akinvox.pinned_sources(PROFILE)
    assert bundle["speakers"] == []
    assert bundle["voice_mode"] == "reference"
    for component in akinvox.COMPONENTS:
        assert (out / akinvox.build_filenames()[component]).is_file()
    assert (out / akinvox.SOURCE_PARAMS_FILENAME).is_file()
    assert (out / "config.json").is_file()


def test_build_rejects_leaked_pytorch_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def export(profile, target, *, cache_dir, opset, run_checker):
        result = _stub_exporter(tmp_path)(
            profile, target, cache_dir=cache_dir, opset=opset, run_checker=run_checker
        )
        (target / "adapter.pt").write_bytes(b"leaked")
        return result

    monkeypatch.setattr(akinvox, "export_bundle", export)
    with pytest.raises(akinvox.AkinvoxBuildError, match="leaked into the bundle"):
        akinvox.build_akinvox_profile(
            "en-akinvox-cloning-v1",
            PROFILE,
            tmp_path / "build",
            opset=17,
            cache_dir=tmp_path / "cache",
            run_checker=False,
        )


def test_build_dispatches_akinvox_cloning_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake(profile_key, profile, out_root, *, opset, cache_dir, run_checker):
        calls.append(profile_key)
        return out_root / profile_key

    monkeypatch.setattr(akinvox, "build_akinvox_profile", fake)
    monkeypatch.setitem(sys.modules, "scripts.akinvox_cloning", akinvox)
    result = build_kokoro.build_profile(
        "en-akinvox-cloning-v1",
        PROFILE,
        tmp_path,
        opset=17,
        seq_len=64,
        run_checker=False,
    )
    assert calls == ["en-akinvox-cloning-v1"]
    assert result == tmp_path / "en-akinvox-cloning-v1"


# --------------------------------------------------------------------------
# Parity helpers
# --------------------------------------------------------------------------


def test_array_metrics_reports_error_and_cosine() -> None:
    reference = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    metrics = akinvox.array_metrics(reference, reference + 0.001)

    assert metrics["shape_exact"] is True
    assert metrics["all_finite"] is True
    assert metrics["max_abs_error"] == pytest.approx(0.001, abs=1e-6)
    assert metrics["cosine_similarity"] > 0.9999


def test_assert_component_parity_requires_high_cosine() -> None:
    report = {
        "decoder": {
            "audio": {
                "all_finite": True,
                "cosine_similarity": 0.5,
                "max_abs_error": 1.0,
            }
        }
    }
    with pytest.raises(akinvox.AkinvoxBuildError, match="cosine"):
        akinvox.assert_component_parity(report)


def test_assert_waveform_parity_accepts_stochastic_source_and_rejects_silence() -> None:
    rng = np.random.default_rng(7)
    expected = rng.standard_normal(4000).astype(np.float32) * 0.1
    metrics = akinvox.assert_waveform_parity(expected, expected * 1.05)
    assert 0.5 <= metrics["rms_ratio"] <= 2.0

    with pytest.raises(akinvox.AkinvoxBuildError, match="RMS ratio"):
        akinvox.assert_waveform_parity(expected, expected * 0.01)


def test_assert_end_to_end_parity_requires_exact_durations() -> None:
    cases = [
        {
            "name": "case",
            "reference_mask_exact": True,
            "synthesis": [
                {"name": "short", "durations_exact": False, "output_length_exact": True}
            ],
        }
    ]
    with pytest.raises(akinvox.AkinvoxBuildError, match="Duration parity"):
        akinvox.assert_end_to_end_parity(cases)


def test_parity_matrix_covers_three_reference_durations() -> None:
    cases = akinvox.parity_cases()
    assert [case["reference_seconds"] for case in cases] == [5.0, 10.0, 20.0]
    assert all(len(case["sentences"]) >= 2 for case in cases)
    assert all(
        len(case["sentences"][1]["tokens"]) > len(case["sentences"][0]["tokens"])
        for case in cases
    )


def test_synthetic_references_are_repository_owned_and_bounded() -> None:
    for seconds in (5.0, 10.0, 20.0):
        wave = akinvox.synthetic_reference(seconds, 20260926)
        assert wave.dtype == np.float32
        assert wave.size == int(seconds * 24000)
        assert float(np.max(np.abs(wave))) < 0.999


# --------------------------------------------------------------------------
# Runtime contracts
# --------------------------------------------------------------------------


def test_validate_component_set_requires_exact_match() -> None:
    contracts.validate_component_set(
        "cloning-onnx-v1",
        akinvox.component_contract()["components"],
        set(akinvox.COMPONENTS),
    )
    with pytest.raises(contracts.ContractError, match="do not match contract"):
        contracts.validate_component_set(
            "cloning-onnx-v1",
            akinvox.component_contract()["components"],
            set(akinvox.COMPONENTS) - {"decoder"},
        )
    contracts.validate_component_set(
        "split-onnx-v1",
        {"prosody": {}, "curves": {}, "decoder": {}},
        {"prosody", "curves", "decoder"},
    )
    with pytest.raises(contracts.ContractError, match="must be"):
        contracts.validate_component_set(
            "split-onnx-v1", {"prosody": {}, "curves": {}}, {"prosody", "curves"}
        )


def test_missing_voice_mode_defaults_to_static() -> None:
    assert contracts.voice_mode({}) == "static"
    assert contracts.is_reference_mode({}) is False
    assert contracts.is_reference_mode({"voice_mode": "reference"}) is True
    with pytest.raises(contracts.ContractError, match="Unsupported voice_mode"):
        contracts.voice_mode({"voice_mode": "cloned"})


def test_validate_reference_constraints_rejects_static_and_speed_support() -> None:
    runtime = akinvox.runtime_metadata()
    with pytest.raises(contracts.ContractError, match="speed_supported"):
        contracts.validate_reference_constraints({**runtime, "speed_supported": 1})
    with pytest.raises(contracts.ContractError, match="static voice roster"):
        contracts.validate_reference_constraints({**runtime, "voices": ["af_heart"]})
    with pytest.raises(contracts.ContractError, match="default voice"):
        contracts.validate_reference_constraints(
            {**runtime, "default_voice": "af_heart"}
        )
    with pytest.raises(contracts.ContractError, match="style_dimensions"):
        contracts.validate_reference_constraints(
            {**runtime, "style_dimensions": {"acoustic": 256, "duration": 0}}
        )


# --------------------------------------------------------------------------
# Release packaging
# --------------------------------------------------------------------------


def test_reference_release_runtime_has_no_static_voice_fields() -> None:
    release = PROFILE["release"]
    runtime = prepare_release._runtime_metadata(
        PROFILE, ROOT / "missing-bundle.json", release
    )

    assert runtime["voice_mode"] == "reference"
    assert runtime["layout"] == "cloning-onnx-v1"
    assert "voices" not in runtime
    assert "default_voice" not in runtime
    assert prepare_release._voices_line(runtime) == "- Voice mode: reference enrollment"


def test_static_release_notes_still_list_voices() -> None:
    line = prepare_release._voices_line({"voices": ["af_heart", "af_msa"]})
    assert line == "- Voices: af_heart, af_msa"


def test_asset_metadata_retains_component() -> None:
    import hashlib

    class FakePath:
        name = "kokoro-akinvox-cloning-decoder-v1.0.1.onnx"

        class _Stat:
            st_size = 12

        def stat(self):
            return self._Stat()

        def open(self, mode):
            import io

            return io.BytesIO(b"0123456789ab")

    metadata = prepare_release._asset_metadata(
        {"role": "model", "format": "onnx", "quality": "fp32", "component": "decoder"},
        FakePath(),
    )
    assert metadata["component"] == "decoder"
    assert metadata["quality"] == "fp32"
    assert metadata["sha256"] == hashlib.sha256(b"0123456789ab").hexdigest()


def test_profile_model_assets_match_published_filenames() -> None:
    published = akinvox.component_filenames()
    model_assets = PROFILE["release"]["model_assets"]

    assert len(model_assets) == 6
    assert {asset["component"] for asset in model_assets} == set(akinvox.COMPONENTS)
    for asset in model_assets:
        assert asset["filename"] == published[asset["component"]]
        assert asset["quality"] == "fp32"
        assert asset["format"] == "onnx"
    auxiliary = PROFILE["release"]["auxiliary_assets"]
    assert auxiliary == [
        {
            "source": "source-params.npz",
            "filename": published["source_params"],
            "role": "metadata",
            "component": "source_params",
            "format": "numpy-npz",
        }
    ]


def test_release_catalog_declares_every_component_and_support_asset() -> None:
    release = RELEASES["en-akinvox-cloning-v1"]

    assert release["model_version"] == "1.0.1"
    assert release["release_version"] == 1
    assert release["runtime"]["voice_mode"] == "reference"
    assert "voices" not in release["runtime"]
    assert "default_voice" not in release["runtime"]
    model_assets = [asset for asset in release["assets"] if asset["role"] == "model"]
    assert {asset["component"] for asset in model_assets} == set(akinvox.COMPONENTS)
    assert all(asset["quality"] == "fp32" for asset in model_assets)
    support = [
        asset
        for asset in release["assets"]
        if asset["role"] == "metadata" and asset["component"] == "source_params"
    ]
    assert len(support) == 1
    assert support[0]["format"] == "numpy-npz"
    assert release["onnx_contract"] == PROFILE["onnx_contract"]


def test_registry_entry_uses_reference_runtime_without_fake_voice() -> None:
    model = MODELS["en-akinvox-cloning-v1"]

    assert model["runtime"]["voice_mode"] == "reference"
    assert model["runtime"]["layout"] == "cloning-onnx-v1"
    assert model["runtime"]["speed_supported"] is False
    assert "voices" not in model["runtime"]
    assert "default_voice" not in model["runtime"]
    assert model["runtime_available"] is False
    assert model["distributions"] == []
    assert set(model["onnx_contract"]["components"]) == set(akinvox.COMPONENTS)
    contracts.validate_reference_constraints(model["runtime"])
