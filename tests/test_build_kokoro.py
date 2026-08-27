from __future__ import annotations

import importlib.util
import json
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
        "sv-joakim",
        "de-thorsten",
        "ru-zaakirio-base",
        "ru-zaakirio-dima",
        "kk-anuarsv",
    }
    assert profiles["he-hebrew-nc"]["release"]["enabled"] is False


def test_swedish_profile_uses_stock_checkpoint_and_all_named_voices() -> None:
    profile = build_kokoro.load_profiles()["sv-joakim"]
    assert profile["repo_id"] == "Joakim/kokoro-sv-voices"
    assert profile["revision"] == "2c7968d59c2fda1667e9e3ff0dd9967150a53f74"
    assert profile["model"] == {
        "kind": "checkpoint",
        "path": "kokoro_sv.pth",
        "config": "config.json",
    }
    assert set(profile["voices"]["items"]) == {
        "Alice",
        "Anton",
        "Björn",
        "Ebba",
        "Elsa",
        "Greta",
        "Lars",
        "Nils",
        "Oskar",
        "Stina",
    }
    assert profile["postprocess"]["frequencies_hz"] == [2400, 4800, 7200, 9600]


def test_thorsten_profile_uses_default_epoch5_alias() -> None:
    profile = build_kokoro.load_profiles()["de-thorsten"]
    assert profile["repo_id"] == "Thorsten-Voice/Kokoro"
    assert profile["model"]["path"] == "model.pth"
    assert profile["voices"]["items"] == {"thorsten": "voices/thorsten.pt"}
    assert "model_ep10.pth" not in json.dumps(profile)


def test_checkpoint_profiles_resolve_through_exporter(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []
    config = tmp_path / "config.json"
    checkpoint = tmp_path / "model.pth"
    config.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")

    def fake_download(repo_id, filename, revision, cache_dir):
        calls.append((repo_id, filename, revision))
        return config if filename == "config.json" else checkpoint

    def fake_export(checkpoint_path, config_path, output_path, *, opset, seq_len):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"exported")

    with (
        patch.object(build_kokoro, "hf_download", side_effect=fake_download),
        patch.object(
            build_kokoro, "export_checkpoint_to_onnx", side_effect=fake_export
        ),
    ):
        profiles = build_kokoro.load_profiles()
        for key in ("sv-joakim", "de-thorsten", "kk-anuarsv"):
            profile = profiles[key]
            out = tmp_path / key / "model.onnx"
            build_kokoro.resolve_model(
                profile, tmp_path / key / "cache", out, opset=14, seq_len=64
            )
            assert out.read_bytes() == b"exported"
    assert calls == [
        (
            "Joakim/kokoro-sv-voices",
            "config.json",
            "2c7968d59c2fda1667e9e3ff0dd9967150a53f74",
        ),
        (
            "Joakim/kokoro-sv-voices",
            "kokoro_sv.pth",
            "2c7968d59c2fda1667e9e3ff0dd9967150a53f74",
        ),
        (
            "Thorsten-Voice/Kokoro",
            "config.json",
            "734e593d320a3d876bede7020f773dfd481a0cc7",
        ),
        (
            "Thorsten-Voice/Kokoro",
            "model.pth",
            "734e593d320a3d876bede7020f773dfd481a0cc7",
        ),
        (
            "AnuarSv/kokoro-tts-kazakh",
            "config.json",
            "90a9283ed61d76c9181a7643819ef1c48b41031d",
        ),
        (
            "AnuarSv/kokoro-tts-kazakh",
            "kokoro_kazakh.pth",
            "90a9283ed61d76c9181a7643819ef1c48b41031d",
        ),
    ]


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

def test_russian_profiles_use_prebuilt_onnx_sources(tmp_path: Path) -> None:
    profiles = build_kokoro.load_profiles()
    sources = {
        "onnx/model.onnx": b"base-onnx",
        "onnx/model_dima.onnx": b"dima-onnx",
    }
    downloaded: list[tuple[str, str, str]] = []

    def fake_download(repo_id, filename, revision, cache_dir):
        downloaded.append((repo_id, filename, revision))
        if filename == "config.json":
            path = tmp_path / "config.json"
            path.write_text("{}", encoding="utf-8")
            return path
        path = tmp_path / Path(filename).name
        path.write_bytes(sources[filename])
        return path

    with (
        patch.object(build_kokoro, "hf_download", side_effect=fake_download),
        patch.object(
            build_kokoro,
            "export_checkpoint_to_onnx",
            side_effect=AssertionError("Russian ONNX must not be re-exported"),
        ),
    ):
        for key, source_name in (
            ("ru-zaakirio-base", "onnx/model.onnx"),
            ("ru-zaakirio-dima", "onnx/model_dima.onnx"),
        ):
            out = tmp_path / key / "model.onnx"
            build_kokoro.resolve_model(
                profiles[key], tmp_path / key / "cache", out, opset=14, seq_len=64
            )
            assert out.read_bytes() == sources[source_name]

    assert downloaded == [
        (
            "zaakirio/kokoro-ru",
            "config.json",
            "d649c57b239b18c4c384378127cbf01dba039bc1",
        ),
        (
            "zaakirio/kokoro-ru",
            "onnx/model.onnx",
            "d649c57b239b18c4c384378127cbf01dba039bc1",
        ),
        (
            "zaakirio/kokoro-ru",
            "config.json",
            "d649c57b239b18c4c384378127cbf01dba039bc1",
        ),
        (
            "zaakirio/kokoro-ru",
            "onnx/model_dima.onnx",
            "d649c57b239b18c4c384378127cbf01dba039bc1",
        ),
    ]


def test_kazakh_profile_is_pinned_to_repaired_checkpoint_revision() -> None:
    profile = build_kokoro.load_profiles()["kk-anuarsv"]
    assert profile["repo_id"] == "AnuarSv/kokoro-tts-kazakh"
    assert profile["revision"] == "90a9283ed61d76c9181a7643819ef1c48b41031d"
    assert profile["model"] == {
        "kind": "checkpoint",
        "path": "kokoro_kazakh.pth",
        "config": "config.json",
    }
    assert profile["voices"]["items"] == {"km_m1": "km_m1.pt"}
    assert profile["language"] == "kk"
