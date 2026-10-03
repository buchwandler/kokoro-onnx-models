from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_candidate", ROOT / "scripts" / "verify_candidate.py"
)
assert SPEC and SPEC.loader
verify_candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_candidate)


def _write_candidate(tmp_path: Path, *, enabled: bool = True) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    model_path = candidate / "model.onnx"
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        model_path.write_bytes(b"model")
    else:
        graph = helper.make_graph(
            [helper.make_node("Identity", ["audio_input"], ["audio"])],
            "test",
            [
                helper.make_tensor_value_info("tokens", TensorProto.INT64, [1, None]),
                helper.make_tensor_value_info("audio_input", TensorProto.FLOAT, [None]),
            ],
            [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [None])],
        )
        onnx.save(helper.make_model(graph), model_path)
    np.savez(candidate / "voices.npz", af=np.zeros((1, 1), dtype=np.float32))
    (candidate / "bundle.json").write_text(
        '{"speakers": [{"name": "af"}]}\n', encoding="utf-8"
    )
    assets = []
    for name, role, fmt, quality in (
        ("model.onnx", "model", "onnx", "fp32"),
        ("voices.npz", "voices", "numpy-npz", None),
        ("bundle.json", "bundle", "json", None),
    ):
        path = candidate / name
        asset = {
            "name": name,
            "role": role,
            "format": fmt,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if quality:
            asset["quality"] = quality
        assets.append(asset)
    manifest = {
        "schema": 2,
        "runtime_contract": 1,
        "repository": "buchwandler/kokoro-onnx-models",
        "tag": "model-files-test",
        "profile": "test",
        "model_version": "1.0",
        "release_version": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "source": {"type": "test", "repository": "source/repo", "revision": "rev"},
        "license": "Apache-2.0",
        "publication": {"enabled": enabled},
        "runtime": {
            "language_codes": ["en"],
            "sample_rate": 24000,
            "frontend": "pykokoro-native-v1",
            "frontend_experimental": False,
            "max_tokens": 510,
            "default_voice": "af",
            "voices": ["af"],
        },
        "onnx_contract": {
            "inputs": {"tokens": "int64"},
            "outputs": {"audio": "float32"},
            "max_tokens": 510,
        },
        "assets": assets,
    }
    (candidate / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (candidate / "SHA256SUMS").write_text(
        "\n".join(f"{asset['sha256']}  {asset['name']}" for asset in assets) + "\n",
        encoding="utf-8",
    )
    return candidate


def test_verify_candidate_checks_manifest_assets_and_checksums(tmp_path: Path) -> None:
    result = verify_candidate.verify_candidate(_write_candidate(tmp_path))
    assert result["asset_count"] == 3


def test_validate_voice_asset_enforces_declared_rows(tmp_path: Path) -> None:
    path = tmp_path / "voices.npz"
    np.savez(path, af=np.zeros((2, 1), dtype=np.float32))
    asset = {
        "format": "numpy-npz",
        "handling": {"rows": 2},
    }
    verify_candidate._validate_voice_asset(path, asset, {"voices": ["af"]})

    asset["handling"]["rows"] = 3
    with pytest.raises(verify_candidate.CandidateError, match="3 rows"):
        verify_candidate._validate_voice_asset(path, asset, {"voices": ["af"]})


@pytest.mark.parametrize(
    ("members", "runtime_voices", "match"),
    [
        (["af"], ["af", "am"], "missing voices"),
        (["af", "am", "extra"], ["af", "am"], "unexpected voices"),
    ],
    ids=["missing-voice", "unexpected-voice"],
)
def test_validate_voice_asset_requires_exact_roster(
    tmp_path: Path, members: list[str], runtime_voices: list[str], match: str
) -> None:
    path = tmp_path / "voices.npz"
    np.savez(path, **{name: np.zeros((2, 1), dtype=np.float32) for name in members})
    asset = {"format": "numpy-npz", "handling": {"rows": 2}}
    with pytest.raises(verify_candidate.CandidateError, match=match):
        verify_candidate._validate_voice_asset(path, asset, {"voices": runtime_voices})


@pytest.mark.parametrize(
    ("handling", "match"),
    [
        ({"voice_count": 3}, "voice_count"),
        ({"members": ["am", "af"]}, "handling members"),
    ],
    ids=["wrong-count", "wrong-members"],
)
def test_validate_voice_asset_requires_matching_handling(
    tmp_path: Path, handling: dict[str, object], match: str
) -> None:
    path = tmp_path / "voices.npz"
    np.savez(
        path,
        af=np.zeros((2, 1), dtype=np.float32),
        am=np.zeros((2, 1), dtype=np.float32),
    )
    asset = {"format": "numpy-npz", "handling": {"rows": 2, **handling}}
    with pytest.raises(verify_candidate.CandidateError, match=match):
        verify_candidate._validate_voice_asset(path, asset, {"voices": ["af", "am"]})


def test_verify_candidate_requires_thorsten_provenance(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profile"] = "de-thorsten"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(verify_candidate.CandidateError, match="documented default"):
        verify_candidate.verify_candidate(candidate)


def _thorsten_provenance() -> dict[str, object]:
    return {
        "source_artifacts": {
            "model": {
                "path": "model.pth",
                "sha256": "36dde15c4a800cfd1ab540ccb4476dbab604fe03ff7c937d976ebbf3b49e59ce",
                "config_sha256": "1" * 64,
            },
            "voices": {
                "thorsten": {
                    "path": "voices/thorsten.pt",
                    "sha256": "9d98b775ebce1cfc369e8f9a3ee8ee260cd612dffb477cba85749112362306d7",
                },
            },
        },
        "exporter": {
            "kokoro_version": "0.9.4",
            "torch_version": "2.13.0",
            "onnx_version": "1.22.0",
            "onnxruntime_version": "1.29.0",
            "python_version": "3.13.14",
            "opset": 17,
            "outputs": ["audio", "duration"],
            "random_source_ops": ["RandomNormalLike"],
            "decoder_reconstruction": {
                "reference_backend": "torch.istft",
                "backend": "exact-convtranspose-istft-v1",
                "filter_length": 20,
                "hop_length": 5,
                "win_length": 20,
                "window": "hann-periodic",
                "center": True,
                "one_sided_bin_scaling": True,
                "window_envelope_normalization": True,
                "native_delegate_validation": {
                    "max_abs_error": 1.0e-7,
                    "cases": [{"name": "hallo"}],
                },
                "native_patched_validation": {
                    "max_abs_error": 1.0e-7,
                    "cases": [{"name": "hallo"}],
                },
            },
            "waveform_validation": {
                "cases": [
                    {
                        "name": "hallo",
                        "native": {},
                        "patched": {},
                        "onnx": {},
                    }
                ]
            },
            "checkpoint_load": {
                "strict": True,
                "components": {
                    component: {
                        "missing_keys": [],
                        "unexpected_keys": [],
                        "shape_mismatches": [],
                        "loaded_tensor_mismatches": [],
                    }
                    for component in (
                        "bert",
                        "bert_encoder",
                        "predictor",
                        "text_encoder",
                        "decoder",
                    )
                },
            },
            "native_reference_validation": {
                "status": "pass",
                "cases": [{"name": "hallo"}],
            },
        },
    }


def test_thorsten_provenance_rejects_missing_and_unsupported_random_ops() -> None:
    manifest = {"profile": "de-thorsten", "provenance": _thorsten_provenance()}
    exporter = manifest["provenance"]["exporter"]
    del exporter["random_source_ops"]
    with pytest.raises(verify_candidate.CandidateError, match="random_source_ops"):
        verify_candidate._validate_checkpoint_provenance(manifest)

    exporter["random_source_ops"] = ["Identity"]
    with pytest.raises(
        verify_candidate.CandidateError, match="unsupported random operators"
    ):
        verify_candidate._validate_checkpoint_provenance(manifest)

    exporter["random_source_ops"] = ["RandomNormalLike"]
    verify_candidate._validate_checkpoint_provenance(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_backend", "custom", "reference backend"),
        ("one_sided_bin_scaling", False, "one_sided_bin_scaling"),
        ("window_envelope_normalization", False, "window_envelope_normalization"),
    ],
)
def test_thorsten_provenance_rejects_invalid_decoder_reconstruction(
    field: str, value: object, message: str
) -> None:
    manifest = {"profile": "de-thorsten", "provenance": _thorsten_provenance()}
    decoder = manifest["provenance"]["exporter"]["decoder_reconstruction"]
    decoder[field] = value
    with pytest.raises(verify_candidate.CandidateError, match=message):
        verify_candidate._validate_checkpoint_provenance(manifest)


def test_thorsten_provenance_rejects_missing_native_metrics() -> None:
    manifest = {"profile": "de-thorsten", "provenance": _thorsten_provenance()}
    del manifest["provenance"]["exporter"]["waveform_validation"]["cases"][0]["native"]
    with pytest.raises(verify_candidate.CandidateError, match="native metrics"):
        verify_candidate._validate_checkpoint_provenance(manifest)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("native_delegate_validation", "native-delegate"),
        ("native_patched_validation", "native/patched"),
    ],
)
def test_thorsten_provenance_requires_both_validation_phases(
    field: str, message: str
) -> None:
    manifest = {"profile": "de-thorsten", "provenance": _thorsten_provenance()}
    del manifest["provenance"]["exporter"]["decoder_reconstruction"][field]
    with pytest.raises(verify_candidate.CandidateError, match=message):
        verify_candidate._validate_checkpoint_provenance(manifest)


def test_thorsten_provenance_allows_stochastic_pointwise_reconstruction_error() -> None:
    manifest = {"profile": "de-thorsten", "provenance": _thorsten_provenance()}
    manifest["provenance"]["exporter"]["decoder_reconstruction"][
        "native_patched_validation"
    ]["max_abs_error"] = 1.0e-3

    verify_candidate._validate_checkpoint_provenance(manifest)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, None, "bad", True])
def test_thorsten_provenance_rejects_invalid_reconstruction_error(
    value: object,
) -> None:
    manifest = {"profile": "de-thorsten", "provenance": _thorsten_provenance()}
    manifest["provenance"]["exporter"]["decoder_reconstruction"][
        "native_patched_validation"
    ]["max_abs_error"] = value

    with pytest.raises(
        verify_candidate.CandidateError,
        match="finite and non-negative",
    ):
        verify_candidate._validate_checkpoint_provenance(manifest)


def test_verify_candidate_rejects_size_mismatch(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"][0]["size"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(verify_candidate.CandidateError, match="Size mismatch"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_unmanifested_file(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    (candidate / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(
        verify_candidate.CandidateError, match="Unexpected candidate files"
    ):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_disabled_publication_by_default(
    tmp_path: Path,
) -> None:
    candidate = _write_candidate(tmp_path, enabled=False)
    with pytest.raises(
        verify_candidate.CandidateError, match="Publication is disabled"
    ):
        verify_candidate.verify_candidate(candidate)
    verify_candidate.verify_candidate(candidate, allow_restricted=True)


def test_verify_candidate_rejects_duplicate_quality(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    extra = candidate / "model2.onnx"
    extra.write_bytes(b"second")
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"].append(
        {
            "name": extra.name,
            "role": "model",
            "quality": "fp32",
            "format": "onnx",
            "size": extra.stat().st_size,
            "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        verify_candidate.CandidateError, match="Duplicate asset role/format slot"
    ):
        verify_candidate.verify_candidate(candidate)


def _write_split_candidate(tmp_path: Path) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    candidate = tmp_path / "split-candidate"
    candidate.mkdir()
    contracts = {
        "prosody": {
            "inputs": {
                "input_ids": "int64",
                "style_dur": "float32",
                "speed": "float32",
            },
            "outputs": {"pred_dur": "float32", "d": "float32", "t_en": "float32"},
        },
        "curves": {
            "inputs": {"en": "float32", "style_dur": "float32"},
            "outputs": {"f0_curve": "float32", "n_curve": "float32"},
        },
        "decoder": {
            "inputs": {
                "asr": "float32",
                "f0_curve": "float32",
                "n_curve": "float32",
                "style_acou": "float32",
                "har": "float32",
            },
            "outputs": {"audio": "float32"},
        },
    }
    for component, contract in contracts.items():
        input_infos = [
            helper.make_tensor_value_info(
                name,
                {"float32": TensorProto.FLOAT, "int64": TensorProto.INT64}[
                    expected_type
                ],
                [1],
            )
            for name, expected_type in contract["inputs"].items()
        ]
        source = (
            "style_dur" if component == "prosody" else next(iter(contract["inputs"]))
        )
        output_infos = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
            for name in contract["outputs"]
        ]
        nodes = [
            helper.make_node("Identity", [source], [name])
            for name in contract["outputs"]
        ]
        graph = helper.make_graph(nodes, component, input_infos, output_infos)
        onnx.save(helper.make_model(graph), candidate / f"{component}.onnx")
    np.savez(candidate / "voices.npz", f_young_clear=np.zeros((1, 1), dtype=np.float32))
    assets = []
    for path, role, quality, component in [
        (candidate / "prosody.onnx", "model", "fp32", "prosody"),
        (candidate / "curves.onnx", "model", "fp32", "curves"),
        (candidate / "decoder.onnx", "model", "fp32", "decoder"),
        (candidate / "voices.npz", "voices", None, None),
    ]:
        asset = {
            "name": path.name,
            "role": role,
            "format": "onnx" if role == "model" else "numpy-npz",
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if quality:
            asset["quality"] = quality
        if component:
            asset["component"] = component
        assets.append(asset)
    manifest = {
        "schema": 2,
        "runtime_contract": 1,
        "repository": "buchwandler/kokoro-onnx-models",
        "tag": "model-files-split-test",
        "profile": "split-test",
        "model_version": "1.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "source": {"type": "test", "repository": "source/repo", "revision": "rev"},
        "license": "Apache-2.0",
        "publication": {"enabled": True},
        "runtime": {
            "language_codes": ["th"],
            "sample_rate": 24000,
            "frontend": "test",
            "frontend_experimental": False,
            "max_tokens": 510,
            "default_voice": "f_young_clear",
            "voices": ["f_young_clear"],
            "layout": "split-onnx-v1",
        },
        "onnx_contract": {
            "inputs": {"input_ids": "int64"},
            "outputs": {"audio": "float32"},
            "max_tokens": 510,
            "components": contracts,
        },
        "assets": assets,
    }
    (candidate / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (candidate / "SHA256SUMS").write_text(
        "\n".join(f"{asset['sha256']}  {asset['name']}" for asset in assets) + "\n",
        encoding="utf-8",
    )
    return candidate


def _refresh_split_manifest(candidate: Path, manifest: dict) -> None:
    for asset in manifest["assets"]:
        path = candidate / asset["name"]
        asset["size"] = path.stat().st_size
        asset["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (candidate / "release-manifest.json").write_text(json.dumps(manifest))
    (candidate / "SHA256SUMS").write_text(
        "\n".join(f"{asset['sha256']}  {asset['name']}" for asset in manifest["assets"])
        + "\n"
    )


def test_verify_candidate_accepts_valid_split_model(tmp_path: Path) -> None:
    candidate = _write_split_candidate(tmp_path)
    result = verify_candidate.verify_candidate(candidate)
    assert result["asset_count"] == 4
    manifest = result["manifest"]
    assert sum(asset["role"] == "model" for asset in manifest["assets"]) == 3
    assert {
        asset["component"] for asset in manifest["assets"] if asset["role"] == "model"
    } == {"prosody", "curves", "decoder"}


def test_verify_candidate_rejects_missing_split_graph(tmp_path: Path) -> None:
    candidate = _write_split_candidate(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"] = [
        asset for asset in manifest["assets"] if asset.get("component") != "decoder"
    ]
    (candidate / "decoder.onnx").unlink()
    _refresh_split_manifest(candidate, manifest)
    with pytest.raises(verify_candidate.CandidateError, match="components"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_duplicate_split_component(tmp_path: Path) -> None:
    candidate = _write_split_candidate(tmp_path)
    extra = candidate / "prosody-copy.onnx"
    extra.write_bytes((candidate / "prosody.onnx").read_bytes())
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    asset = next(
        item for item in manifest["assets"] if item.get("component") == "prosody"
    )
    duplicate = dict(asset)
    duplicate["name"] = extra.name
    manifest["assets"].append(duplicate)
    _refresh_split_manifest(candidate, manifest)
    with pytest.raises(
        verify_candidate.CandidateError, match="Duplicate asset role/format slot"
    ):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_wrong_split_graph_contract(tmp_path: Path) -> None:
    candidate = _write_split_candidate(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"] = [
        asset for asset in manifest["assets"] if asset.get("component") != "prosody"
    ]
    curves = next(
        item for item in manifest["assets"] if item.get("component") == "curves"
    )
    curves["component"] = "prosody"
    (candidate / "prosody.onnx").unlink()
    _refresh_split_manifest(candidate, manifest)
    with pytest.raises(verify_candidate.CandidateError, match="input_ids"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_invalid_release_versions(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["release_version"] = 0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(verify_candidate.CandidateError, match="release_version"):
        verify_candidate.verify_candidate(candidate)

    manifest["release_version"] = "1"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(verify_candidate.CandidateError, match="release_version"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_enforces_expected_versions_and_legacy_compatibility(
    tmp_path: Path,
) -> None:
    candidate = _write_candidate(tmp_path)
    with pytest.raises(verify_candidate.CandidateError, match="model_version"):
        verify_candidate.verify_candidate(candidate, expected_model_version="2.0")
    with pytest.raises(verify_candidate.CandidateError, match="release_version"):
        verify_candidate.verify_candidate(candidate, expected_release_version=2)

    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["release_version"]
    manifest_path.write_text(json.dumps(manifest))
    verify_candidate.verify_candidate(candidate)
    with pytest.raises(
        verify_candidate.CandidateError, match="missing release_version"
    ):
        verify_candidate.verify_candidate(candidate, expected_release_version=1)


def test_expected_release_version_parser_rejects_non_positive_values() -> None:
    with pytest.raises(verify_candidate.argparse.ArgumentTypeError):
        verify_candidate._positive_release_version("0")
    with pytest.raises(verify_candidate.argparse.ArgumentTypeError):
        verify_candidate._positive_release_version("2.0")


CLONING_CONTRACTS = {
    "reference_wavlm": {
        "inputs": {"input_values": "float32"},
        "outputs": {"wavlm": "float32"},
    },
    "reference_encoders": {
        "inputs": {"mel": "float32"},
        "outputs": {"raw_sdec": "float32", "raw_spred": "float32"},
    },
    "reference_mapper": {
        "inputs": {
            "mel": "float32",
            "mel_lengths": "int64",
            "reference_ids": "int64",
            "reference_lengths": "int64",
            "wavlm": "float32",
            "raw_sdec": "float32",
            "raw_spred": "float32",
        },
        "outputs": {
            "style": "float32",
            "reference_memory": "float32",
            "reference_mask": "bool",
        },
    },
    "prosody": {
        "inputs": {
            "input_ids": "int64",
            "style_dur": "float32",
            "reference_memory": "float32",
            "reference_mask": "bool",
        },
        "outputs": {"pred_dur": "int64", "d": "float32", "t_en": "float32"},
    },
    "curves": {
        "inputs": {"en": "float32", "style_dur": "float32"},
        "outputs": {"f0_curve": "float32", "n_curve": "float32"},
    },
    "decoder": {
        "inputs": {
            "asr": "float32",
            "f0_curve": "float32",
            "n_curve": "float32",
            "style_acou": "float32",
            "har": "float32",
        },
        "outputs": {"audio": "float32"},
    },
}

REFERENCE_RUNTIME = {
    "language_codes": ["en"],
    "sample_rate": 24000,
    "frontend": "pykokoro-native-v1",
    "frontend_experimental": False,
    "max_tokens": 510,
    "voice_mode": "reference",
    "layout": "cloning-onnx-v1",
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
}


def _write_cloning_candidate(tmp_path: Path) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    candidate = tmp_path / "cloning-candidate"
    candidate.mkdir()
    types = {
        "float32": TensorProto.FLOAT,
        "int64": TensorProto.INT64,
        "bool": TensorProto.BOOL,
    }
    for component, contract in CLONING_CONTRACTS.items():
        input_infos = [
            helper.make_tensor_value_info(name, types[kind], [1])
            for name, kind in contract["inputs"].items()
        ]
        output_infos = [
            helper.make_tensor_value_info(name, types[kind], [1])
            for name, kind in contract["outputs"].items()
        ]
        source = next(iter(contract["inputs"]))
        nodes = [
            helper.make_node("Cast", [source], [name], to=types[kind])
            for name, kind in contract["outputs"].items()
        ]
        graph = helper.make_graph(nodes, component, input_infos, output_infos)
        onnx.save(helper.make_model(graph), candidate / f"{component}.onnx")
    np.savez(
        candidate / "source-params.npz",
        weight=np.zeros((1, 9), dtype=np.float32),
        bias=np.zeros(1, dtype=np.float32),
        window=np.zeros(20, dtype=np.float32),
    )
    (candidate / "config.json").write_text("{}", encoding="utf-8")

    assets = [
        {
            "name": f"{component}.onnx",
            "role": "model",
            "format": "onnx",
            "quality": "fp32",
            "component": component,
            "size": (candidate / f"{component}.onnx").stat().st_size,
            "sha256": hashlib.sha256(
                (candidate / f"{component}.onnx").read_bytes()
            ).hexdigest(),
        }
        for component in CLONING_CONTRACTS
    ]
    for name, role, format_name, component in [
        ("source-params.npz", "metadata", "numpy-npz", "source_params"),
        ("config.json", "config", "json", None),
    ]:
        asset = {
            "name": name,
            "role": role,
            "format": format_name,
            "size": (candidate / name).stat().st_size,
            "sha256": hashlib.sha256((candidate / name).read_bytes()).hexdigest(),
        }
        if component:
            asset["component"] = component
        assets.append(asset)
    manifest = {
        "schema": 2,
        "runtime_contract": 1,
        "repository": "buchwandler/kokoro-onnx-models",
        "tag": "model-files-cloning-test",
        "profile": "cloning-test",
        "model_version": "1.0.1",
        "release_version": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "source": {"type": "test", "repository": "source/repo", "revision": "rev"},
        "license": "Apache-2.0",
        "publication": {"enabled": True},
        "runtime": dict(REFERENCE_RUNTIME),
        "onnx_contract": {
            "inputs": {"reference": "reference-conditioned"},
            "outputs": {"audio": "float32"},
            "max_tokens": 510,
            "components": {
                name: {"inputs": c["inputs"], "outputs": c["outputs"]}
                for name, c in CLONING_CONTRACTS.items()
            },
        },
        "assets": assets,
    }
    _write_manifest(candidate, manifest)
    return candidate


def _write_manifest(candidate: Path, manifest: dict) -> None:
    (candidate / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (candidate / "SHA256SUMS").write_text(
        "".join(f"{a['sha256']}  {a['name']}\n" for a in manifest["assets"]),
        encoding="utf-8",
    )


def test_verify_candidate_accepts_reference_only_cloning_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _write_cloning_candidate(tmp_path)
    result = verify_candidate.verify_candidate(candidate)

    runtime = result["manifest"]["runtime"]
    assert runtime["voice_mode"] == "reference"
    assert "voices" not in runtime
    assert "default_voice" not in runtime
    assert not any(a["role"] == "voices" for a in result["manifest"]["assets"])
    assert result["asset_count"] == 8


def test_verify_candidate_checks_declared_license_notice_assets(tmp_path: Path) -> None:
    candidate = _write_cloning_candidate(tmp_path)
    manifest = json.loads((candidate / "release-manifest.json").read_text())
    for name, role, format_name in [
        ("LICENSE.txt", "license", "text"),
        ("ATTRIBUTION.md", "attribution", "markdown"),
    ]:
        path = candidate / name
        path.write_text(f"{name} notice\n", encoding="utf-8")
        manifest["assets"].append(
            {
                "name": name,
                "role": role,
                "format": format_name,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest["license_notices"] = {
        "statement": "Components retain independent terms.",
        "assets": ["LICENSE.txt", "ATTRIBUTION.md"],
    }
    _write_manifest(candidate, manifest)

    verify_candidate.verify_candidate(candidate)

    manifest["license_notices"]["assets"].remove("ATTRIBUTION.md")
    _write_manifest(candidate, manifest)
    with pytest.raises(verify_candidate.CandidateError, match="must exactly match"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_requires_source_params_for_cloning(
    tmp_path: Path,
) -> None:
    candidate = _write_cloning_candidate(tmp_path)
    manifest = json.loads((candidate / "release-manifest.json").read_text())
    manifest["assets"] = [
        a for a in manifest["assets"] if a.get("component") != "source_params"
    ]
    _write_manifest(candidate, manifest)
    (candidate / "source-params.npz").unlink()

    with pytest.raises(verify_candidate.CandidateError, match="source_params"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_cloning_component_drift(tmp_path: Path) -> None:
    candidate = _write_cloning_candidate(tmp_path)
    manifest = json.loads((candidate / "release-manifest.json").read_text())
    manifest["onnx_contract"]["components"].pop("decoder")
    manifest["assets"] = [
        a for a in manifest["assets"] if a.get("component") != "decoder"
    ]
    _write_manifest(candidate, manifest)
    (candidate / "decoder.onnx").unlink()

    with pytest.raises(verify_candidate.CandidateError, match="must be"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_checksums_cover_every_component(tmp_path: Path) -> None:
    candidate = _write_cloning_candidate(tmp_path)
    manifest = json.loads((candidate / "release-manifest.json").read_text())
    kept = [a for a in manifest["assets"] if a["name"] != "decoder.onnx"]
    (candidate / "SHA256SUMS").write_text(
        "".join(f"{a['sha256']}  {a['name']}\n" for a in kept), encoding="utf-8"
    )

    with pytest.raises(verify_candidate.CandidateError, match="SHA256SUMS"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_reference_runtime_with_static_voice(
    tmp_path: Path,
) -> None:
    candidate = _write_cloning_candidate(tmp_path)
    manifest = json.loads((candidate / "release-manifest.json").read_text())
    manifest["runtime"]["default_voice"] = "af_heart"
    _write_manifest(candidate, manifest)

    with pytest.raises(verify_candidate.CandidateError):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_still_requires_voices_for_static_models(
    tmp_path: Path,
) -> None:
    candidate = _write_split_candidate(tmp_path)
    manifest = json.loads((candidate / "release-manifest.json").read_text())
    manifest["assets"] = [a for a in manifest["assets"] if a["role"] != "voices"]
    _write_manifest(candidate, manifest)
    (candidate / "voices.npz").unlink()

    with pytest.raises(verify_candidate.CandidateError, match="voices asset"):
        verify_candidate.verify_candidate(candidate)


_INNO_ENROLLER = {
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


def _candidate_with_enroller(tmp_path: Path) -> Path:
    candidate = _write_candidate(tmp_path)
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        pytest.skip("onnx is required for enroller graph checks")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["fbank"], ["voicepack"])],
        "inno",
        [
            helper.make_tensor_value_info("fbank", TensorProto.FLOAT, [1, None, 80]),
            helper.make_tensor_value_info("tilt", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("head_stats", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("blend_weights", TensorProto.FLOAT, [1, None]),
        ],
        [helper.make_tensor_value_info("voicepack", TensorProto.FLOAT, [510, 1, 256])],
    )
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile"] = "v1.0"
    manifest["onnx_contract"]["components"] = {
        "inno_voicepack": {
            "inputs": {
                "fbank": "float32",
                "tilt": "float32",
                "head_stats": "float32",
                "blend_weights": "float32",
            },
            "outputs": {"voicepack": "float32"},
        }
    }
    manifest["runtime"]["voice_enrollers"] = [_INNO_ENROLLER]
    assets = manifest["assets"]
    for name, role, fmt, component, quality in (
        ("inno-voicepack-v0.2.onnx", "model", "onnx", "inno_voicepack", "fp32"),
        ("inno-tuner-v0.2.npz", "metadata", "numpy-npz", "inno_tuner", None),
        ("inno-tuner-v0.2.json", "metadata", "json", "inno_tuner_config", None),
    ):
        path = candidate / name
        if component == "inno_voicepack":
            onnx.save(helper.make_model(graph), str(path))
        elif component == "inno_tuner":
            np.savez(
                path,
                blend_stats=np.zeros((2, 3), dtype=np.float32),
                blend_grades=np.zeros(2, dtype=np.float32),
                blend_scale=np.ones(3, dtype=np.float32),
                blend_gate=np.asarray(4.0, dtype=np.float32),
                grade_pen=np.asarray(1.0, dtype=np.float32),
            )
        else:
            path.write_text(
                '{"blend_names": ["a", "b"], "version": "0.2.0"}', encoding="utf-8"
            )
        asset = {
            "name": name,
            "role": role,
            "format": fmt,
            "component": component,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if quality:
            asset["quality"] = quality
        assets.append(asset)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (candidate / "SHA256SUMS").write_text(
        "\n".join(f"{asset['sha256']}  {asset['name']}" for asset in assets) + "\n",
        encoding="utf-8",
    )
    return candidate


def _refresh_checksums(candidate: Path) -> None:
    manifest = json.loads(
        (candidate / "release-manifest.json").read_text(encoding="utf-8")
    )
    (candidate / "SHA256SUMS").write_text(
        "\n".join(f"{asset['sha256']}  {asset['name']}" for asset in manifest["assets"])
        + "\n",
        encoding="utf-8",
    )


def test_verify_candidate_accepts_inno_enrollers_with_components(
    tmp_path: Path,
) -> None:
    candidate = _candidate_with_enroller(tmp_path)
    result = verify_candidate.verify_candidate(candidate, expected_profile="v1.0")
    assert result["manifest"]["runtime"]["voice_enrollers"] == [_INNO_ENROLLER]


def test_verify_candidate_rejects_enroller_with_missing_metadata_component(
    tmp_path: Path,
) -> None:
    candidate = _candidate_with_enroller(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"] = [
        asset for asset in manifest["assets"] if asset["name"] != "inno-tuner-v0.2.npz"
    ]
    (candidate / "inno-tuner-v0.2.npz").unlink()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_checksums(candidate)
    with pytest.raises(verify_candidate.CandidateError, match="metadata component"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_unordered_enroller_durations(
    tmp_path: Path,
) -> None:
    candidate = _candidate_with_enroller(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"]["voice_enrollers"][0]["min_seconds"] = 9.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_checksums(candidate)
    with pytest.raises(verify_candidate.CandidateError, match="durations"):
        verify_candidate.verify_candidate(candidate)
