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


def test_verify_candidate_requires_thorsten_provenance(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profile"] = "de-thorsten"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(verify_candidate.CandidateError, match="epoch 5"):
        verify_candidate.verify_candidate(candidate)


def _thorsten_provenance() -> dict[str, object]:
    return {
        "source_artifacts": {
            "model": {
                "path": "model_ep5.pth",
                "sha256": "0" * 64,
                "config_sha256": "1" * 64,
            },
            "voices": {
                "thorsten": {
                    "path": "voices/thorsten_ep5.pt",
                    "sha256": "2" * 64,
                }
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
    [("native_delegate_validation", "native-delegate"), ("native_patched_validation", "native/patched")],
 )
def test_thorsten_provenance_requires_both_validation_phases(
    field: str, message: str
 ) -> None:
    manifest = {"profile": "de-thorsten", "provenance": _thorsten_provenance()}
    del manifest["provenance"]["exporter"]["decoder_reconstruction"][field]
    with pytest.raises(verify_candidate.CandidateError, match=message):
        verify_candidate._validate_checkpoint_provenance(manifest)

def test_thorsten_provenance_rejects_excessive_reconstruction_error() -> None:
    manifest = {"profile": "de-thorsten", "provenance": _thorsten_provenance()}
    manifest["provenance"]["exporter"]["decoder_reconstruction"][
        "native_patched_validation"
    ]["max_abs_error"] = 1.0e-3
    with pytest.raises(verify_candidate.CandidateError, match="exceeds"):
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
