from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "onnx_notch", ROOT / "scripts" / "onnx_notch.py"
)
assert SPEC and SPEC.loader
onnx_notch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = onnx_notch
SPEC.loader.exec_module(onnx_notch)


def test_iirnotch_matches_scipy() -> None:
    scipy_signal = pytest.importorskip("scipy.signal")
    expected_b, expected_a = scipy_signal.iirnotch(2400, 35, 24000)
    actual_b, actual_a = onnx_notch.iirnotch_coefficients(2400, 35, 24000)
    np.testing.assert_allclose(actual_b, expected_b, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_a, expected_a, rtol=1e-6, atol=1e-7)


def test_filtfilt_reference_matches_scipy() -> None:
    scipy_signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(123)
    values = rng.normal(size=1000).astype(np.float32)
    b, a = onnx_notch.iirnotch_coefficients(4800, 35, 24000)
    expected = scipy_signal.filtfilt(b, a, values)
    actual = onnx_notch.filtfilt_reference(values, b, a)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_embed_notch_filters_runs_in_onnxruntime(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    model_path = tmp_path / "identity.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["audio"])],
        "identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None])],
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [None])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)]),
        model_path,
    )

    metadata = onnx_notch.embed_notch_filters(
        model_path,
        frequencies_hz=[2400, 4800, 7200, 9600],
        quality=35,
        sample_rate=24000,
    )
    assert metadata["kind"] == "embedded-zero-phase-notch-v1"
    assert len(metadata["filters"]) == 4

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    assert model.graph.output[0].name == "audio"
    assert sum(node.op_type == "Scan" for node in model.graph.node) == 8

    values = np.sin(np.arange(2000, dtype=np.float32) * 0.2)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": values})[0]
    expected = onnx_notch.apply_notch_filters(
        values, [2400, 4800, 7200, 9600], 35, 24000
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-6)
