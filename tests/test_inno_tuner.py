from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_inno_tuner", ROOT / "scripts" / "export_inno_tuner.py"
)
assert SPEC and SPEC.loader
inno = importlib.util.module_from_spec(SPEC)
sys.modules["export_inno_tuner"] = inno
SPEC.loader.exec_module(inno)

AUGMENTATION = inno.load_augmentation()


# --------------------------------------------------------------------------
# Pins and provenance
# --------------------------------------------------------------------------


def test_source_pins_are_immutable_and_complete() -> None:
    inno.validate_source_pins(AUGMENTATION)

    pins = inno.pinned_sources(AUGMENTATION)
    assert AUGMENTATION["source_repository"] == "remsky/kokoro-inno-clone-tuner"
    assert AUGMENTATION["source_revision"] == (
        "429617d18ce4d637acea948bdff4cce3ec6cf167"
    )
    assert AUGMENTATION["code_repository"] == "remsky/inno-kokoro"
    assert AUGMENTATION["code_tag"] == "v0.2.0"
    assert AUGMENTATION["code_revision"] == (
        "892ef184bc932aa3ff9d72c1509d5b81ff6941e6"
    )
    assert pins["tuner"]["release_tag"] == "v0.2.0"
    assert set(pins["tuner"]["files"]) == {"model.safetensors", "config.json"}
    for details in pins["tuner"]["files"].values():
        assert len(details["sha256"]) == 64
        assert details["size"] > 0
    assert pins["declared_version"] == "0.2.0"


def test_validate_source_pins_rejects_moving_or_unchecked_sources() -> None:
    moving = json.loads(json.dumps(AUGMENTATION))
    moving["source_revision"] = "main"
    with pytest.raises(inno.InnoTunerBuildError, match="commit SHA"):
        inno.validate_source_pins(moving)

    unchecked = json.loads(json.dumps(AUGMENTATION))
    del unchecked["source_files"]["model.safetensors"]["sha256"]
    with pytest.raises(inno.InnoTunerBuildError, match="SHA-256"):
        inno.validate_source_pins(unchecked)

    wrong_version = json.loads(json.dumps(AUGMENTATION))
    wrong_version["declared_version"] = "0.3.0"
    with pytest.raises(inno.InnoTunerBuildError, match="0.2.0"):
        inno.validate_source_pins(wrong_version)


# --------------------------------------------------------------------------
# Weight loading
# --------------------------------------------------------------------------


def _write_safetensors(
    path: Path, tensors: dict[str, np.ndarray], meta: dict[str, str]
) -> None:
    header: dict[str, object] = {}
    offset = 0
    data = b""
    dtypes = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        raw = array.tobytes()
        header[name] = {
            "dtype": dtypes[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        data += raw
    header["__metadata__"] = meta
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def _state_fixture(path: Path, packs: int = 4) -> Path:
    rng = np.random.default_rng(0)
    tensors = {
        "enc.backbone.conv1.weight": rng.standard_normal(
            (32, 1, 3, 3)
        ).astype(np.float16),
        "enc.proj.0.weight": rng.standard_normal((1024, 256)).astype(np.float16),
        "heads.style.0.weight": rng.standard_normal((512, 512)).astype(np.float32),
        "heads.style.0.bias": np.zeros(512, dtype=np.float32),
        "heads.style.2.weight": rng.standard_normal((256, 512)).astype(np.float32),
        "heads.style.2.bias": np.zeros(256, dtype=np.float32),
        "heads.tilt.weight": rng.standard_normal((256, 1)).astype(np.float32),
        "blend.rows": rng.standard_normal((packs, 510, 128)).astype(np.float32),
        "blend.stats": rng.standard_normal((packs, 3)).astype(np.float32),
        "blend.grades": np.linspace(2.0, 4.0, packs).astype(np.float32),
        "head.W": rng.standard_normal((3, 128)).astype(np.float32),
        "head.mu": np.zeros(2, dtype=np.float32),
        "head.sd": np.ones(2, dtype=np.float32),
    }
    meta = {
        "blend_names": ",".join(f"pack{index}" for index in range(packs)),
        "blend_scale": "1.0,1.1,0.8",
        "blend_gate": "4.0",
        "grade_pen": "1.0",
        "tilt_mean": "-7.8",
        "tilt_sd": "1.8",
        "version": "0.2.0",
        "name": "inno_ref01",
    }
    _write_safetensors(path, tensors, meta)
    return path


def test_read_safetensors_is_pickle_free(tmp_path: Path) -> None:
    path = _state_fixture(tmp_path / "model.safetensors")
    tensors, meta = inno.read_safetensors(path)

    assert tensors["blend.rows"].shape == (4, 510, 128)
    assert tensors["enc.backbone.conv1.weight"].dtype == np.float16
    assert meta["version"] == "0.2.0"


def test_load_tuner_state_reconstructs_all_components(tmp_path: Path) -> None:
    state = inno.load_tuner_state(_state_fixture(tmp_path / "model.safetensors"))

    assert state.version == "0.2.0"
    assert state.blend.names == ("pack0", "pack1", "pack2", "pack3")
    assert state.blend_rows.shape == (4, 510, 128)
    assert state.blend.stats.shape == (4, 3)
    assert state.blend.gate == 4.0
    assert state.blend.grade_pen == 1.0
    assert state.tilt_dir.shape == (256,)
    assert state.head_weight.shape == (3, 128)
    assert state.head_mu.shape == (2,)
    assert state.head_sd.shape == (2,)
    assert state.tilt_mean == -7.8 and state.tilt_sd == 1.8
    assert set(state.style) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    assert all(value.dtype == np.float32 for value in state.enc.values())


def test_load_tuner_state_rejects_missing_encoder(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _state_fixture(path)
    tensors, meta = inno.read_safetensors(path)
    tensors = {
        name: value for name, value in tensors.items() if not name.startswith("enc.")
    }
    _write_safetensors(path, tensors, meta)
    with pytest.raises(inno.InnoTunerBuildError, match="enc\\.\\*"):
        inno.load_tuner_state(path)


# --------------------------------------------------------------------------
# Host features (the fbank frontend stays out of the ONNX graph)
# --------------------------------------------------------------------------


def test_host_fbank_is_deterministic_and_mean_normalized() -> None:
    rng = np.random.default_rng(1)
    wav = rng.standard_normal(inno.SR_ENC * 2).astype(np.float32)

    first = inno.host_fbank(wav)
    second = inno.host_fbank(wav)

    assert first.shape == ((len(wav) - inno.FBANK_FRAME) // inno.FBANK_HOP + 1, 80)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.allclose(first, second, atol=0)
    assert np.abs(first.mean(0)).max() < 1e-4
    assert inno.mel_filterbank().shape == (80, 257)


def test_prosody_measurements_match_upstream_self_checks() -> None:
    sr = inno.SR_ENC
    t = np.arange(sr * 2) / sr
    tone = sum(np.sin(2 * np.pi * 220 * index * t) / index for index in range(1, 6))
    mean, std, _ = inno.host_stats(tone.astype(np.float32), sr)
    assert abs(mean - 13.7) < 0.5 and std < 0.5

    missing = sum(np.sin(2 * np.pi * 110 * index * t) for index in range(2, 7))
    tracked = inno.host_stats(missing.astype(np.float32), sr)[0]
    assert abs(tracked - 1.65) < 0.7

    rng = np.random.default_rng(0)
    t = np.arange(sr * 4) / sr
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.cos(2 * np.pi * 5 * t))
    rate = inno.host_rate(
        (envelope * rng.standard_normal(len(t))).astype(np.float32), sr
    )
    assert abs(rate - 5) < 0.7

    white = rng.standard_normal(sr * 4).astype(np.float32)
    pink = np.fft.irfft(
        np.fft.rfft(white) / np.sqrt(np.arange(1, sr * 2 + 2))
    ).astype(np.float32)
    assert abs(inno.host_tilt(white, sr)) < 0.3
    assert abs(inno.host_tilt(pink, sr) + 3) < 0.3


def test_blend_weights_are_nonnegative_and_normalized() -> None:
    tables = inno.BlendTables(
        names=("low", "mid", "high"),
        stats=np.array([[5.0, 2.0, 4.0], [12.0, 1.5, 4.5], [20.0, 2.5, 5.0]]),
        grades=np.array([4.0, 2.0, 3.0]),
        scale=np.array([1.0, 1.1, 0.8]),
        gate=4.0,
        grade_pen=1.0,
    )
    z = np.array([12.2, 1.4, 4.4])

    named = inno.host_blend_weights(tables, z)
    vector = inno.blend_weight_vector(tables, z)

    assert set(named) <= set(tables.names)
    assert all(weight > 0 for weight in named.values())
    assert abs(sum(named.values()) - 1) < 1e-3
    assert vector.shape == (3,)
    assert np.isclose(vector.sum(), 1, atol=1e-3)
    assert vector.argmax() == 1


def test_generated_pack_shape_is_exactly_510_1_256() -> None:
    pack = inno.assemble_voicepack(
        np.arange(128, dtype=np.float32), np.zeros((510, 128), dtype=np.float32)
    )

    assert pack.shape == (510, 1, 256)
    assert pack.dtype == np.float32
    assert np.isfinite(pack).all()
    assert np.allclose(pack[:, 0, :128], np.arange(128, dtype=np.float32))

    with pytest.raises(inno.InnoTunerBuildError, match="predictor rows"):
        inno.assemble_voicepack(np.zeros(128, np.float32), np.zeros((256, 128)))


# --------------------------------------------------------------------------
# Tuner metadata and enroller declaration
# --------------------------------------------------------------------------


def test_tuner_metadata_is_pickle_free_with_json_sidecar(tmp_path: Path) -> None:
    state = inno.load_tuner_state(_state_fixture(tmp_path / "model.safetensors"))
    out = tmp_path / "out"
    out.mkdir()

    npz_path, json_path = inno.write_tuner_metadata(state, out, AUGMENTATION)

    assert npz_path.name == "inno-tuner-v0.2.npz"
    assert json_path.name == "inno-tuner-v0.2.json"
    with np.load(npz_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(inno.TUNER_NPZ_FIELDS)
        assert archive["blend_stats"].shape == (4, 3)
        assert archive["blend_grades"].shape == (4,)
        assert archive["blend_scale"].shape == (3,)
        assert float(archive["blend_gate"]) == 4.0
        assert float(archive["grade_pen"]) == 1.0
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["blend_names"] == ["pack0", "pack1", "pack2", "pack3"]
    assert document["version"] == "0.2.0"
    assert document["tilt_norm"] == {"mean_db_per_oct": -7.8, "sd": 1.8}
    assert document["output"] == {
        "format": "kokoro-voicepack-v1",
        "shape": [510, 1, 256],
        "dtype": "float32",
    }
    assert document["source"] == inno.pinned_sources(AUGMENTATION)


def test_enroller_metadata_declares_the_documented_contract() -> None:
    assert inno.enroller_metadata() == {
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
    assets = {asset["component"] for asset in AUGMENTATION["assets"]}
    enroller = inno.enroller_metadata()
    assert enroller["model_component"] in assets
    assert enroller["metadata_component"] in assets
    assert "inno_tuner_config" in assets


def test_license_metadata_includes_cc_by_sa_speaker_encoder() -> None:
    components = inno.LICENSE_COMPONENTS
    assert components["inno_code"]["license"] == "Apache-2.0"
    assert components["wespeaker_code"]["license"] == "Apache-2.0"
    assert components["voxceleb_init"]["license"] == "CC-BY-4.0"
    assert components["speaker_encoder"]["license"] == "CC-BY-SA-3.0"
    assert "CC-BY-SA-3.0" in inno.ARTIFACT_LICENSE
    assert "Apache-2.0" != inno.ARTIFACT_LICENSE

    licenses_doc = (ROOT / "MODEL_LICENSES.md").read_text(encoding="utf-8")
    assert "UniSpeech-SAT" in licenses_doc
    assert "CC BY-SA 3.0" in licenses_doc
    assert "inno-voicepack-v0.2.onnx" in licenses_doc
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "speaker encoder" in notice


# --------------------------------------------------------------------------
# Build assembly
# --------------------------------------------------------------------------


def _stub_sources(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in inno.SOURCE_FILE_NAMES:
        path = tmp_path / name
        path.write_bytes(f"fixture-{name}".encode("utf-8"))
        sources[name] = path
    return sources


def _fixture_pins(sources: dict[str, Path]) -> dict:
    spec = json.loads(json.dumps(AUGMENTATION))
    spec["source_files"] = {
        name: {"size": path.stat().st_size, "sha256": inno.sha256(path)}
        for name, path in sources.items()
    }
    return spec


def _stub_exporter(out_dir: Path):
    def export(spec, target, *, sources, opset, run_checker):
        target.mkdir(parents=True, exist_ok=True)
        for asset in spec["assets"]:
            (target / str(asset["source"])).write_bytes(
                b"tuner-placeholder-" + str(asset["component"]).encode("utf-8")
            )
        return {
            "tuner": {"id": "inno-v0.2", "version": "0.2.0", "blend_names": []},
            "exporter": {"graph": inno.GRAPH_FILENAME, "opset": opset},
        }

    return export


def test_build_emits_components_bundle_and_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _stub_sources(tmp_path / "cache")
    spec = _fixture_pins(sources)
    monkeypatch.setattr(inno, "export_bundle", _stub_exporter(tmp_path))

    out = inno.build_inno_tuner(
        spec, tmp_path / "build", sources=sources, run_checker=False
    )

    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["profile"] == "inno-v0.2"
    assert {item["component"] for item in bundle["components"]} == {
        "inno_voicepack",
        "inno_tuner",
        "inno_tuner_config",
    }
    assert all(item["format"] in {"onnx", "numpy-npz", "json"} for item in bundle["components"])
    assert all(len(item["sha256"]) == 64 for item in bundle["components"])
    assert bundle["source_artifacts"] == inno.pinned_sources(spec)
    assert set(bundle["source_files"]) == set(inno.SOURCE_FILE_NAMES)
    assert bundle["license"]["components"]["speaker_encoder"]["license"] == (
        "CC-BY-SA-3.0"
    )
    for asset in spec["assets"]:
        assert (out / str(asset["source"])).is_file()


def test_build_rejects_leaked_pytorch_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _stub_sources(tmp_path / "cache")
    spec = _fixture_pins(sources)

    def export(spec, target, *, sources, opset, run_checker):
        result = _stub_exporter(tmp_path)(
            spec, target, sources=sources, opset=opset, run_checker=run_checker
        )
        (target / "model.safetensors").write_bytes(b"leaked")
        return result

    monkeypatch.setattr(inno, "export_bundle", export)
    with pytest.raises(inno.InnoTunerBuildError, match="leaked"):
        inno.build_inno_tuner(spec, tmp_path / "build", sources=sources)


def test_build_rejects_unverified_sources(tmp_path: Path) -> None:
    sources = _stub_sources(tmp_path / "cache")
    spec = _fixture_pins(sources)
    sources["model.safetensors"].write_bytes(b"tampered")
    with pytest.raises(inno.InnoTunerBuildError, match="size"):
        inno.build_inno_tuner(spec, tmp_path / "build", sources=sources)


def test_array_metrics_reports_shape_finiteness_and_half_cosines() -> None:
    expected = np.zeros((510, 1, 256), dtype=np.float32)
    expected[..., 0] = 1.0
    expected[..., 128] = 1.0
    actual = expected.copy()
    actual[0, 0, 129] = 0.5

    metrics = inno.array_metrics(expected, actual)

    assert metrics["shape"] == [510, 1, 256]
    assert metrics["all_finite"] is True
    assert metrics["max_abs_error"] == pytest.approx(0.5)
    assert metrics["cosine_timbre_half"] == pytest.approx(1.0)
    assert metrics["cosine_predictor_half"] < 1.0
