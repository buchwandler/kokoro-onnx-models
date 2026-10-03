from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
compare_akinvox = _load(
    "compare_akinvox_cloning_onnx", "local_test/compare_akinvox_cloning_onnx.py"
)
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
    assert PROFILE["frontend"]["kind"] == "misaki"
    assert PROFILE["frontend"]["sherpa_text_compatible"] is False
    assert set(pins["akinvox"]["files"]) == {
        "adapter.pt",
        "reference_mapper.pt",
        "reference_encoders.pt",
        "config.json",
    }
    assert pins["kokoro_base"]["repo_id"] == "hexgrad/Kokoro-82M"
    assert pins["wavlm"]["repo_id"] == "microsoft/wavlm-base-plus-sv"
    assert pins["source_code"]["commit"] == "322c5c3901e6e1aee46670deb868854413c06aef"
    assert "README.md" in pins["wavlm"]["files"]
    assert pins["wavlm"]["license_source"]["license"] == "CC BY-SA 3.0"
    assert pins["wavlm"]["license_source"]["revision"] == (
        "6112826ac13a4327f4c9a7afa2a505e35b763514"
    )
    for pin in ("akinvox", "kokoro_base", "wavlm"):
        assert len(pins[pin]["revision"]) == 40


def test_validate_source_pins_rejects_moving_or_unpinned_sources() -> None:
    mutable = json.loads(json.dumps(PROFILE))
    mutable["revision"] = "main"
    with pytest.raises(akinvox.AkinvoxBuildError, match="commit SHA"):
        akinvox.validate_source_pins(mutable)


    moving_code = json.loads(json.dumps(PROFILE))
    moving_code["model"]["pins"]["source_code"]["commit"] = "main"
    with pytest.raises(akinvox.AkinvoxBuildError, match="source_code commit"):
        akinvox.validate_source_pins(moving_code)

    unpinned = json.loads(json.dumps(PROFILE))
    del unpinned["model"]["pins"]["wavlm"]["files"]["pytorch_model.bin"]
    with pytest.raises(akinvox.AkinvoxBuildError, match="WavLM pin is missing"):
        akinvox.validate_source_pins(unpinned)

    unchecked = json.loads(json.dumps(PROFILE))
    del unchecked["model"]["pins"]["akinvox"]["files"]["adapter.pt"]["sha256"]
    with pytest.raises(akinvox.AkinvoxBuildError, match="lowercase SHA-256"):
        akinvox.validate_source_pins(unchecked)



def test_copy_license_assets_preserves_pinned_notices(tmp_path: Path) -> None:
    source_root = tmp_path / "akinvox-source"
    upstream_files = {
        "LICENSE": "AkinVox license\n",
        "NOTICE": "AkinVox notice\n",
        "THIRD_PARTY_LICENSES.md": "Third-party attribution\n",
        "licenses/KOKORO_LICENSE": "Kokoro license\n",
        "licenses/STYLE_TTS2_LICENSE": "StyleTTS2 license\n",
    }
    for relative, content in upstream_files.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    model_card = tmp_path / "wavlm-model-card.md"
    model_card.write_text("Pinned WavLM model card\n", encoding="utf-8")
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    artifacts = akinvox.copy_license_assets(
        PROFILE,
        {"source_code": source_root, "wavlm:README.md": model_card},
        out_dir,
    )

    assert len(artifacts) == 8
    assert {record["path"] for record in artifacts} == {
        "licenses/AKINVOX_LICENSE.txt",
        "licenses/AKINVOX_NOTICE.txt",
        "licenses/AKINVOX_THIRD_PARTY_LICENSES.md",
        "licenses/KOKORO_LICENSE.txt",
        "licenses/STYLE_TTS2_LICENSE.txt",
        "licenses/WAVLM_CC-BY-SA-3.0.txt",
        "licenses/WAVLM_MODEL_CARD.md",
        "licenses/LICENSE_NOTICES.md",
    }
    assert (out_dir / "licenses/WAVLM_CC-BY-SA-3.0.txt").read_bytes() == (
        ROOT / PROFILE["model"]["pins"]["wavlm"]["license_source"]["local_file"]
    ).read_bytes()
    assert (out_dir / "licenses/WAVLM_MODEL_CARD.md").read_text() == (
        "Pinned WavLM model card\n"
    )


def test_native_wavlm_uses_the_passed_identity_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTorch:
        from_numpy = staticmethod(lambda values: values)
        nn = SimpleNamespace(
            functional=SimpleNamespace(
                normalize=staticmethod(
                    lambda values, dim, eps=1.0e-12: values
                    / np.linalg.norm(values, axis=dim, keepdims=True)
                )
            )
        )

    class FakeIdentity:
        def __call__(self, *, input_values):
            assert input_values.shape == (1, 16000)
            embeddings = SimpleNamespace(
                float=lambda: np.asarray([[3.0, 4.0]], dtype=np.float32)
            )
            return SimpleNamespace(embeddings=embeddings)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    fake_torchaudio = SimpleNamespace(
        functional=SimpleNamespace(
            resample=lambda values, old_rate, new_rate: SimpleNamespace(
                numpy=lambda: np.zeros(16000, dtype=np.float32)
            )
        )
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    result = akinvox._native_wavlm(np.zeros(24000, dtype=np.float32), FakeIdentity())

    np.testing.assert_allclose(result, [[0.6, 0.8]])


def test_exported_wavlm_wrapper_preserves_raw_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs: list[np.ndarray] = []

    class FakeModule:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def normalize(values, dim, eps=1.0e-12):
        return values / np.linalg.norm(values, axis=dim, keepdims=True)

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(
            Module=FakeModule,
            functional=SimpleNamespace(normalize=normalize),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(akinvox, "_WRAPPER_TYPES", None)

    class FakeIdentity:
        def __call__(self, *, input_values):
            inputs.append(np.asarray(input_values).copy())
            embeddings = SimpleNamespace(
                float=lambda: np.asarray([[3.0, 4.0]], dtype=np.float32)
            )
            return SimpleNamespace(embeddings=embeddings)

    wrapper = akinvox.wrapper_types().ReferenceWavLM(FakeIdentity())
    raw_audio = np.asarray([[0.1, 0.2, 0.4]], dtype=np.float32)

    result = wrapper(raw_audio)

    np.testing.assert_array_equal(inputs[0], raw_audio)
    np.testing.assert_allclose(result, [[0.6, 0.8]])


def test_wavlm_preprocessing_requires_raw_16khz_audio() -> None:
    akinvox.validate_wavlm_preprocessing({"sampling_rate": 16000, "do_normalize": False})
    with pytest.raises(akinvox.AkinvoxBuildError, match="preserve raw waveform"):
        akinvox.validate_wavlm_preprocessing(
            {"sampling_rate": 16000, "do_normalize": True}
        )

def test_base_checkpoint_keys_strip_module_and_translate_weight_norm() -> None:
    raw_state = {
        "module.encoder.weight": np.asarray([1.0], dtype=np.float32),
        "module.decoder.weight_g": np.asarray([2.0], dtype=np.float32),
        "module.decoder.weight_v": np.asarray([3.0], dtype=np.float32),
    }

    translated = akinvox._base_state_candidates(raw_state)

    assert set(translated) == {
        "encoder.weight",
        "decoder.parametrizations.weight.original0",
        "decoder.parametrizations.weight.original1",
    }
    np.testing.assert_array_equal(translated["encoder.weight"], raw_state["module.encoder.weight"])




def test_materialize_spectral_norm_preserves_frozen_weights() -> None:
    torch = pytest.importorskip("torch")
    from torch.nn.utils.parametrizations import spectral_norm

    layer = spectral_norm(torch.nn.Linear(3, 2)).eval()
    values = torch.randn(2, 3)
    expected = layer(values).detach().clone()

    report = akinvox.materialize_spectral_norm([layer])

    torch.testing.assert_close(layer(values), expected, rtol=0.0, atol=0.0)
    assert report == {
        "materialized_spectral_norm_modules": 1,
        "remaining_parametrizations": 0,
        "weights_frozen": True,
    }

def test_onnx_duration_encoder_uses_time_axis_for_style_rows() -> None:
    torch = pytest.importorskip("torch")
    value = torch.zeros(1, 512, 32)
    style = torch.zeros(1, 128)

    encoded = akinvox._onnx_duration_encoder(SimpleNamespace(lstms=[]), value, style)

    assert encoded.shape == (1, 32, 640)

def test_onnx_instance_norm_matches_torch_instance_norm() -> None:
    torch = pytest.importorskip("torch")
    values = torch.randn(2, 4, 32)
    expected = torch.nn.InstanceNorm1d(4, affine=False)(values)

    actual = akinvox._onnx_instance_norm(values, eps=1.0e-5)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)

def test_native_mels_pass_numpy_to_upstream_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    inputs: list[np.ndarray] = []

    def preprocess(wave: np.ndarray):
        assert isinstance(wave, np.ndarray)
        inputs.append(wave.copy())
        return torch.zeros(80, 100)

    monkeypatch.setitem(
        sys.modules,
        "kokoro_cloning.reference_mel",
        SimpleNamespace(preprocess=preprocess),
    )

    mel, encoder_mel = akinvox._native_mels(np.zeros(1200, dtype=np.float32))

    assert len(inputs) == 2
    assert mel.shape == (80, 100)
    assert encoder_mel.shape == (1, 1, 80, 100)

def test_onnx_half_downsample_matches_replicated_odd_frame_padding() -> None:
    torch = pytest.importorskip("torch")
    values = torch.randn(1, 3, 80, 109)
    padded = torch.cat([values, values[..., -1].unsqueeze(-1)], dim=-1)

    actual = akinvox._onnx_half_downsample(values)
    expected = torch.nn.functional.avg_pool2d(padded, 2)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)

def test_onnx_attention_supports_dynamic_query_and_key_lengths() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    attention = torch.nn.MultiheadAttention(
        16, 4, dropout=0.0, batch_first=True
    ).eval()
    query = torch.randn(1, 7, 16)
    key = torch.randn(1, 9, 16)
    mask = torch.tensor([[False] * 7 + [True] * 2])

    expected, _ = attention(
        query, key, key, key_padding_mask=mask, need_weights=False
    )
    actual = akinvox._onnx_multihead_attention(attention, query, key, key, mask)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

def test_onnx_transformer_encoder_matches_torch_with_padding_mask() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(23)
    layer = torch.nn.TransformerEncoderLayer(
        16,
        4,
        dim_feedforward=32,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    encoder = torch.nn.TransformerEncoder(
        layer, 2, enable_nested_tensor=False
    ).eval()
    values = torch.randn(1, 7, 16)
    mask = torch.tensor([[False] * 5 + [True] * 2])

    expected = encoder(values, src_key_padding_mask=mask)
    actual = akinvox._onnx_transformer_encoder(encoder, values, mask)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

def test_native_enrollment_uses_separate_mel_and_reference_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    mel = torch.zeros(1, 80, 401)
    encoder_mel = torch.zeros(1, 1, 80, 32)
    observed: dict[str, list[int]] = {}

    class FakeMapper:
        def __call__(self, wavlm, raw_sdec, raw_spred, row_indices, prompt):
            observed["row_indices"] = row_indices.tolist()
            return torch.zeros(1, 256)

    def encode_reference(mapper, features, lengths, reference_ids):
        observed["mel_lengths"] = lengths.tolist()
        observed["reference_ids"] = [reference_ids.shape[1]]
        return (
            torch.zeros(1, 101, 192),
            torch.zeros(1, 101, dtype=torch.bool),
            None,
        )

    monkeypatch.setattr(akinvox, "_native_mels", lambda wave: (mel, encoder_mel))
    monkeypatch.setattr(akinvox, "_native_encode_reference", encode_reference)
    pipeline = akinvox._NativePipeline({}, FakeMapper(), object())
    monkeypatch.setattr(
        pipeline,
        "_observations",
        lambda features, wave: (
            torch.zeros(1, 512),
            torch.zeros(1, 128),
            torch.zeros(1, 128),
        ),
    )

    result = pipeline.enroll(np.zeros(100, dtype=np.float32), np.arange(24))

    assert result["memory"].shape == (1, 101, 192)
    assert observed == {
        "mel_lengths": [401],
        "reference_ids": [26],
        "row_indices": [25],
    }

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
            "license_notice_artifacts": [],
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
    assert bundle["license_notices"] == PROFILE["release"]["license_notices"]
    assert bundle["license_notice_artifacts"] == []
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
    cache_dirs: list[Path] = []

    def fake(profile_key, profile, out_root, *, opset, cache_dir, run_checker):
        calls.append(profile_key)
        cache_dirs.append(cache_dir)
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
    assert cache_dirs == [tmp_path / ".cache" / "en-akinvox-cloning-v1"]
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
    source_params = [
        asset for asset in auxiliary if asset["component"] == "source_params"
    ]
    assert len(source_params) == 1
    assert source_params[0]["filename"] == published["source_params"]
    notices = PROFILE["release"]["license_notices"]
    notice_assets = [
        asset for asset in auxiliary if asset["role"] in {"license", "attribution"}
    ]
    assert {asset["filename"] for asset in notice_assets} == set(notices["assets"])
    assert (
        "does not imply that AkinVox relicenses dependencies" in notices["statement"]
    )
    assert "WavLM" in notices["statement"]


def test_release_catalog_declares_every_component_and_support_asset() -> None:
    release = RELEASES["en-akinvox-cloning-v1"]

    assert release["model_version"] == "1.0.1"
    assert release["release_version"] == 1
    assert release["publish"] is False
    assert "AkinVox Apache-2.0 contributions only" in release["license"]
    assert release["license_notices"] == PROFILE["release"]["license_notices"]
    assert release["activate_runtime_registry"] is False
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
    notices = [
        asset
        for asset in release["assets"]
        if asset["role"] in {"license", "attribution"}
    ]
    assert {asset["name"] for asset in notices} == set(
        release["license_notices"]["assets"]
    )
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
    assert model["license"]["redistribution"].startswith("publication disabled")
    assert set(model["onnx_contract"]["components"]) == set(akinvox.COMPONENTS)
    contracts.validate_reference_constraints(model["runtime"])


@pytest.mark.parametrize("keep_build", [False, True])
def test_comparison_cli_records_seed_and_respects_keep_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_build: bool
) -> None:
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps({compare_akinvox.PROFILE_KEY: {"export_validation": {"export_seed": 1}}}),
        encoding="utf-8",
    )
    captured: dict[str, int] = {}

    def build(profile_key, profile, out_root, **kwargs):
        captured["seed"] = profile["export_validation"]["export_seed"]
        target = out_root / profile_key
        target.mkdir(parents=True)
        report = {
            "status": "pass",
            "components": {},
            "seeded_cases": 0,
            "end_to_end": [],
        }
        (target / compare_akinvox.akinvox.PARITY_REPORT_FILENAME).write_text(
            json.dumps(report), encoding="utf-8"
        )

    monkeypatch.setattr(compare_akinvox.akinvox, "build_akinvox_profile", build)
    build_root = tmp_path / "build"
    output_root = tmp_path / "reports"
    arguments = [
        "--profile",
        compare_akinvox.PROFILE_KEY,
        "--profiles",
        str(profiles_path),
        "--build-root",
        str(build_root),
        "--output-root",
        str(output_root),
        "--seed",
        "98765",
    ]
    if keep_build:
        arguments.append("--keep-build")

    assert compare_akinvox.run(arguments) == 0

    summary = json.loads((output_root / compare_akinvox.PROFILE_KEY / "report.json").read_text())
    assert captured["seed"] == 98765
    assert summary["seed"] == 98765
    assert (build_root / compare_akinvox.PROFILE_KEY).is_dir() is keep_build



def test_akinvox_workflow_requires_gate_reports() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build-release.yml").read_text(
        encoding="utf-8"
    )

    assert "uv run --python 3.12 --extra build --extra test" in workflow
    assert (
        'test -f ".local-test/compare/en-akinvox-cloning-v1/report.json"' in workflow
    )
    assert (
        'test -f ".local-test/compare/en-akinvox-cloning-v1/parity-report.json"' in workflow
    )
    assert 'test -f ".local-test/smoke/en-akinvox-cloning-v1/report.json"' in workflow
    assert workflow.count(".local-test/compare/en-akinvox-cloning-v1") >= 3
    assert workflow.count(".local-test/smoke/en-akinvox-cloning-v1") >= 2
