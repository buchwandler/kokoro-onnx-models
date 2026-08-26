from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_kokoro.py"
SPEC = importlib.util.spec_from_file_location("build_kokoro", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
build_kokoro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_kokoro)


def test_normalizes_rank2_voice() -> None:
    got = build_kokoro.normalize_voice(
        np.zeros((17, 256), dtype=np.float32), name="test"
    )
    assert got.shape == (17, 1, 256)


def test_rejects_non_kokoro_checkpoint_shape() -> None:
    try:
        build_kokoro.normalize_voice(np.zeros((128, 512), dtype=np.float32), name="bad")
    except build_kokoro.BuildError:
        pass
    else:
        raise AssertionError("invalid voice shape was accepted")


def test_raw_round_trip_multiple_speakers() -> None:
    a = np.arange(12 * 256, dtype=np.float32).reshape(12, 1, 256)
    b = np.full((12, 1, 256), 7.0, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "voices.bin"
        shape = build_kokoro.write_sherpa_voices_bin({"a": a, "b": b}, path)
        assert shape == (12, 1, 256)
        reread = build_kokoro._read_raw_voice_file(path, ["a", "b"])
        np.testing.assert_array_equal(reread["a"], a)
        np.testing.assert_array_equal(reread["b"], b)


def test_named_numpy_archive_round_trip_multiple_speakers() -> None:
    a = np.arange(12 * 256, dtype=np.float32).reshape(12, 1, 256)
    b = np.full((12, 1, 256), 7.0, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "voices.npz"
        shape = build_kokoro.write_numpy_voice_archive({"a": a, "b": b}, path)
        assert shape == (12, 1, 256)
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == ["a", "b"]
            np.testing.assert_array_equal(archive["a"], a)
            np.testing.assert_array_equal(archive["b"], b)


def test_expected_profiles_exist() -> None:
    profiles = build_kokoro.load_profiles()
    assert set(profiles) == {
        "vi-contextbox",
        "vi-anphunl",
        "ar-nabra",
        "de-crane",
        "he-hebrew-nc",
    }
    assert profiles["he-hebrew-nc"]["release"]["enabled"] is False


def test_nabra_profile_uses_prebuilt_onnx_source() -> None:
    profiles = build_kokoro.load_profiles()
    nabra = profiles["ar-nabra"]

    assert nabra["repo_id"] == "marwanelamami/Nabra-82M-v0.1-ONNX"
    assert nabra["revision"] == "85065b0be8573aefb401f8f53e7edc37d6556186"
    assert nabra["model"] == {"kind": "onnx", "path": "nabra_fp32.onnx"}
    assert nabra["voices"]["items"] == {"af_msa": "voices_af_msa.pt"}
    assert nabra["onnx_contract"]["inputs"] == {
        "input_ids": "int64",
        "ref_s": "float32",
        "speed": "float32",
    }


def test_resolve_model_nabra_does_not_export_checkpoint(tmp_path: Path) -> None:
    profile = build_kokoro.load_profiles()["ar-nabra"]
    source = tmp_path / "nabra_fp32.onnx"
    source.write_bytes(b"prebuilt-onnx")
    out = tmp_path / "build" / "model.onnx"

    def fake_download(repo_id, filename, revision, cache_dir):
        assert repo_id == profile["repo_id"]
        assert revision == profile["revision"]
        assert filename == "nabra_fp32.onnx"
        return source

    with (
        patch.object(build_kokoro, "hf_download", side_effect=fake_download),
        patch.object(
            build_kokoro,
            "export_checkpoint_to_onnx",
            side_effect=AssertionError("Nabra must not export a checkpoint"),
        ),
    ):
        assert (
            build_kokoro.resolve_model(
                profile, tmp_path / "cache", out, opset=14, seq_len=64
            )
            is None
        )

    assert out.read_bytes() == b"prebuilt-onnx"


def test_resolve_auxiliary_files_copies_vocab_at_pinned_revision(
    tmp_path: Path,
) -> None:
    profile = build_kokoro.load_profiles()["ar-nabra"]
    source = tmp_path / "vocab.json"
    source.write_text('{"ʕ": 7, "ħ": 8}', encoding="utf-8")
    out_dir = tmp_path / "build"

    def fake_download(repo_id, filename, revision, cache_dir):
        assert repo_id == profile["repo_id"]
        assert filename == "vocab.json"
        assert revision == profile["revision"]
        return source

    with patch.object(build_kokoro, "hf_download", side_effect=fake_download):
        resolved = build_kokoro.resolve_auxiliary_files(
            profile, tmp_path / "cache", out_dir
        )

    assert (out_dir / "vocab.json").read_text(encoding="utf-8") == source.read_text()
    assert resolved == [{"path": "vocab.json", "role": "vocab", "format": "json"}]


def test_validate_nabra_named_onnx_contract(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    model_path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [],
        "nabra",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, None]),
            helper.make_tensor_value_info("ref_s", TensorProto.FLOAT, [1, 256]),
            helper.make_tensor_value_info("speed", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [None])],
    )
    onnx.save(helper.make_model(graph), model_path)

    assert (
        build_kokoro.validate_onnx_contract(
            model_path,
            {
                "inputs": {
                    "input_ids": "int64",
                    "ref_s": "float32",
                    "speed": "float32",
                },
                "outputs": {"audio": "float32"},
                "sample_rate": 24000,
                "max_tokens": 510,
            },
        )
        == []
    )
