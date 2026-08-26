from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_kokoro.py"
SPEC = importlib.util.spec_from_file_location("build_kokoro", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
build_kokoro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_kokoro)


def test_normalizes_rank2_voice() -> None:
    got = build_kokoro.normalize_voice(np.zeros((17, 256), dtype=np.float32), name="test")
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
    assert set(profiles) == {"vi-contextbox", "vi-anphunl", "ar-nabra", "de-crane", "he-hebrew-nc"}
    assert profiles["he-hebrew-nc"]["release"]["enabled"] is False
