from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = ROOT / "local_test" / "common.py"


def _load_common(monkeypatch):
    fake_pykokoro = types.ModuleType("pykokoro")
    fake_pykokoro.GenerationConfig = object
    fake_pykokoro.KokoroPipeline = object
    fake_pykokoro.PipelineConfig = object
    monkeypatch.setitem(sys.modules, "pykokoro", fake_pykokoro)

    fake_tokenizer = types.ModuleType("pykokoro.tokenizer")

    class FakeTokenizerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_tokenizer.TokenizerConfig = FakeTokenizerConfig
    monkeypatch.setitem(sys.modules, "pykokoro.tokenizer", fake_tokenizer)

    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.write = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    module_name = "_kokoro_local_test_common"
    spec = importlib.util.spec_from_file_location(module_name, COMMON_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_prepared_asset_path_selects_model_quality_and_voice_format(
    tmp_path: Path, monkeypatch
) -> None:
    common = _load_common(monkeypatch)
    manifest = {
        "assets": [
            {
                "name": "kokoro-v1.0.fp16.onnx",
                "role": "model",
                "quality": "fp16",
                "format": "onnx",
            },
            {
                "name": "kokoro-v1.0.onnx",
                "role": "model",
                "quality": "fp32",
                "format": "onnx",
            },
            {
                "name": "kokoro-v1.0.q8.onnx",
                "role": "model",
                "quality": "q8",
                "format": "onnx",
            },
            {
                "name": "voices-v1.0.raw.bin",
                "role": "voices",
                "format": "raw-float32-le",
            },
            {
                "name": "voices-v1.0.npz",
                "role": "voices",
                "format": "numpy-npz",
            },
            {
                "name": "vocab-v1.0.json",
                "role": "vocab",
                "format": "json",
            },
        ]
    }

    assert common._prepared_asset_path(tmp_path, manifest, "model", quality="fp32") == (
        tmp_path / "kokoro-v1.0.onnx"
    )
    assert common._prepared_asset_path(
        tmp_path, manifest, "voices", format="numpy-npz"
    ) == (tmp_path / "voices-v1.0.npz")
    assert common._prepared_asset_path(tmp_path, manifest, "vocab") == (
        tmp_path / "vocab-v1.0.json"
    )


def test_v1_0_hindi_uses_explicit_espeak(monkeypatch) -> None:
    common = _load_common(monkeypatch)

    config = common._tokenizer_for(common.SPECS["v1.0"], "hi", False)

    assert config.kwargs["backend"] == "espeak"
    assert config.kwargs["load_gold"] is False
    assert config.kwargs["load_silver"] is False


def test_v1_0_japanese_uses_native_tokenizer_config(monkeypatch) -> None:
    common = _load_common(monkeypatch)

    config = common._tokenizer_for(common.SPECS["v1.0"], "ja", False)

    assert config.kwargs == {}


def test_v1_0_unsupported_language_is_rejected(monkeypatch) -> None:
    common = _load_common(monkeypatch)

    with pytest.raises(RuntimeError, match="not a native"):
        common._tokenizer_for(common.SPECS["v1.0"], "xx", False)
