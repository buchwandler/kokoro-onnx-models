from __future__ import annotations

import importlib.util
import json
import sys
import types
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = ROOT / "local_test" / "compare_checkpoint_onnx.py"
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_checkpoint_onnx", COMPARE_PATH
)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
compare = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules["compare_checkpoint_onnx"] = compare
COMPARE_SPEC.loader.exec_module(compare)
build_kokoro = sys.modules["scripts.build_kokoro"]


def _audio(sample_count: int = 96_000) -> np.ndarray:
    t = np.arange(sample_count, dtype=np.float64) / 24_000
    envelope = 0.01 + 0.09 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t))
    phase = 2 * np.pi * (180 * t + 25 * t**2)
    return (
        envelope * (np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase))
    ).astype(np.float32)


def _profile(kind: str = "checkpoint", cases: list[dict] | None = None) -> dict:
    return {
        "repo_id": "test/repo",
        "revision": "test-revision",
        "sample_rate": 24_000,
        "model": {"kind": kind, "path": "model.pth", "config": "config.json"},
        "voices": {"kind": "fixed_pt", "items": {"voice": "voice.pt"}},
        "release": {"default_voice": "voice"},
        "export_validation": {"export_seed": 7, "cases": cases or []},
    }


def test_eligible_cases_rejects_profiles_without_native_fixtures() -> None:
    with pytest.raises(build_kokoro.BuildError, match="no native checkpoint"):
        compare.eligible_cases("prebuilt", _profile("onnx"), None)
    with pytest.raises(build_kokoro.BuildError, match="no frozen"):
        compare.eligible_cases("empty", _profile(), None)


def test_parser_exposes_comparison_options() -> None:
    args = compare.build_parser().parse_args(
        [
            "de-thorsten",
            "--case",
            "hallo",
            "--voice",
            "thorsten",
            "--runs",
            "3",
            "--seed",
            "9",
            "--provider",
            "CPUExecutionProvider",
            "--keep-build",
            "--no-wav",
        ]
    )
    assert args.profile == "de-thorsten"
    assert args.cases == ["hallo"]
    assert args.runs == 3
    assert args.seed == 9
    assert args.keep_build is True
    assert args.no_wav is True


def test_native_loader_disables_complex_decoder(monkeypatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(kwargs["disable_complex"])

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KModel=FakeModel))
    with patch.object(
        build_kokoro,
        "audit_loaded_checkpoint",
        return_value={"strict": True, "components": {}},
    ):
        build_kokoro.load_checkpoint_native(tmp_path / "model.pth", {"n_token": 178})
    assert calls == [False]


def test_run_onnx_case_preserves_native_inputs() -> None:
    tokens = np.array([[0, 1, 0]], dtype=np.int64)
    style = np.ones((1, 256), dtype=np.float32)
    speed = np.array([1.0], dtype=np.float32)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

    captured = {}

    class Session:
        def run(self, output_names, inputs):
            captured.update(inputs)
            return np.zeros(10, dtype=np.float32), np.array([1, 1, 1])

    audio, duration = build_kokoro.run_onnx_case(
        Session(),
        {"tokens": Tensor(tokens), "style": Tensor(style), "speed": Tensor(speed)},
    )
    np.testing.assert_array_equal(captured["tokens"], tokens)
    np.testing.assert_array_equal(captured["style"], style)
    np.testing.assert_array_equal(captured["speed"], speed)
    assert audio.shape == (10,)
    assert duration.tolist() == [1, 1, 1]


def test_wav_writer_preserves_relative_amplitude(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    compare._write_wav(path, np.array([0.25, -0.5], dtype=np.float32), 24_000)
    with wave.open(str(path), "rb") as source:
        assert source.getframerate() == 24_000
        assert source.getnframes() == 2
        samples = np.frombuffer(source.readframes(2), dtype="<i2")
    np.testing.assert_array_equal(samples, [8192, -16384])


def test_compare_profile_writes_repeated_report_and_wavs(
    monkeypatch, tmp_path: Path
) -> None:
    cases = [
        {
            "name": "fixture",
            "phonemes": "aa",
            "tokens": [0, 1, 0],
            "speed": 1.0,
        }
    ]
    profile = _profile(cases=cases)
    voice = np.zeros((510, 1, 256), dtype=np.float32)
    audio = _audio()
    duration = np.array([50, 50, 60], dtype=np.int64)
    order: list[str] = []
    temp_config = tmp_path / "config.json"
    temp_checkpoint = tmp_path / "model.pth"
    temp_config.write_text(json.dumps({"n_token": 178}), encoding="utf-8")
    temp_checkpoint.write_bytes(b"checkpoint")

    class Tensor:
        def __init__(self, value, shape):
            self.value = value
            self.shape = shape

        def numpy(self):
            return self.value

    native_case = {
        "name": "fixture",
        "tokens": Tensor(np.array([[0, 1, 0]], dtype=np.int64), (1, 3)),
        "style": Tensor(np.zeros((1, 256), dtype=np.float32), (1, 256)),
        "speed": Tensor(np.array([1.0], dtype=np.float32), (1,)),
        "audio": audio,
        "duration": duration,
    }

    monkeypatch.setattr(
        compare.build_kokoro, "resolve_voices", lambda profile, cache: {"voice": voice}
    )
    monkeypatch.setattr(
        compare.build_kokoro,
        "hf_download",
        lambda *args: (
            order.append("download")
            or (temp_config if args[1] == "config.json" else temp_checkpoint)
        ),
    )
    monkeypatch.setattr(
        compare.build_kokoro,
        "load_checkpoint_native",
        lambda *args: order.append("native") or object(),
    )
    monkeypatch.setattr(
        compare.build_kokoro,
        "capture_native_reference_cases",
        lambda *args, **kwargs: order.append("capture") or [native_case],
    )
    monkeypatch.setattr(
        compare.build_kokoro,
        "resolve_model",
        lambda *args, **kwargs: order.append("export"),
    )
    monkeypatch.setattr(
        compare.build_kokoro,
        "install_exact_onnx_istft",
        lambda model: ({"backend": "test"}, object()),
    )
    monkeypatch.setattr(
        compare.build_kokoro,
        "validate_patched_pytorch_against_native",
        lambda *args, **kwargs: {
            "cases": [{"waveform_structure": {"envelope_correlation": 1.0}}]
        },
    )
    monkeypatch.setattr(
        compare.build_kokoro,
        "run_patched_pytorch_outputs",
        lambda *args, **kwargs: (audio.copy(), duration.copy()),
    )

    class FakeSession:
        def __init__(self, path, providers):
            assert providers == ["CPUExecutionProvider"]

        def run(self, output_names, inputs):
            return audio.copy(), duration.copy()

    monkeypatch.setitem(
        sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=FakeSession)
    )
    monkeypatch.setitem(
        sys.modules,
        "kokoro.model",
        types.SimpleNamespace(
            KModelForONNX=lambda model: types.SimpleNamespace(eval=lambda: model)
        ),
    )

    output_dir = tmp_path / "compare" / "fixture-profile"
    report = compare.compare_profile(
        "fixture-profile",
        profile,
        build_root=tmp_path / "build",
        output_dir=output_dir,
        requested_cases=None,
        voice_name=None,
        runs=3,
        seed=7,
        provider="CPUExecutionProvider",
        write_wav=True,
    )

    assert order.index("capture") < order.index("export")
    assert report["listening"]["status"] == "not-recorded"
    assert len(report["cases"][0]["onnx"]) == 3
    assert len(report["cases"][0]["native_distribution"]) == 3
    assert "rms" in report["cases"][0]["native_envelope"]
    assert report["cases"][0]["native"]["wav"] == "fixture/torch-native.wav"
    assert report["cases"][0]["onnx"][2]["wav"] == "fixture/onnx-03.wav"
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.md").exists()
    assert len(list((output_dir / "fixture").glob("*.wav"))) == 5
    saved = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["runs"] == 3
    assert saved["reference"]["disable_complex"] is False
