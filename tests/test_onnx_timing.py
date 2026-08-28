from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from scripts import onnx_timing


def duration_graph(*, second_chain: bool = False, exposed: bool = False):
    nodes = [
        helper.make_node("Round", ["duration_logits"], ["rounded"]),
        helper.make_node("Clip", ["rounded", "clip_min", "clip_max"], ["clipped"]),
        helper.make_node("Cast", ["clipped"], ["duration_int"], to=TensorProto.INT64),
        helper.make_node("CumSum", ["duration_int", "axis"], ["cumulative"]),
        helper.make_node("Identity", ["audio_input"], ["audio"]),
    ]
    if second_chain:
        nodes.extend(
            [
                helper.make_node("Round", ["other_logits"], ["other_rounded"]),
                helper.make_node(
                    "CumSum", ["other_rounded", "axis"], ["other_cumulative"]
                ),
            ]
        )
    outputs = [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [None])]
    if exposed:
        outputs.append(
            helper.make_tensor_value_info("duration", TensorProto.INT64, [None])
        )
    graph_inputs = [
        helper.make_tensor_value_info("tokens", TensorProto.INT64, [1, None]),
        helper.make_tensor_value_info("duration_logits", TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info("audio_input", TensorProto.FLOAT, [None]),
    ]
    if second_chain:
        graph_inputs.append(
            helper.make_tensor_value_info("other_logits", TensorProto.FLOAT, [None])
        )
    initializers = [
        helper.make_tensor("clip_min", TensorProto.FLOAT, [], [1]),
        helper.make_tensor("clip_max", TensorProto.FLOAT, [], [100]),
        helper.make_tensor("axis", TensorProto.INT64, [], [0]),
    ]
    if exposed:
        nodes.append(helper.make_node("Identity", ["clipped"], ["duration"]))
    return helper.make_model(
        helper.make_graph(nodes, "timing", graph_inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 17)],
    )


def test_semantic_detector_prefers_clip_output() -> None:
    model = duration_graph()
    assert onnx_timing.find_kokoro_duration_tensor(model) == "clipped"


def test_detector_rejects_ambiguous_duration_chains() -> None:
    with pytest.raises(onnx_timing.TimingError, match="Ambiguous"):
        onnx_timing.find_kokoro_duration_tensor(duration_graph(second_chain=True))


def test_detector_accepts_existing_duration_output() -> None:
    assert (
        onnx_timing.find_kokoro_duration_tensor(duration_graph(exposed=True))
        == "duration"
    )


def test_no_duration_graph_returns_none() -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["audio_input"], ["audio"])],
        "audio-only",
        [helper.make_tensor_value_info("audio_input", TensorProto.FLOAT, [None])],
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [None])],
    )
    assert onnx_timing.find_kokoro_duration_tensor(helper.make_model(graph)) is None


def test_expose_duration_output_adds_public_output_and_checker_passes() -> None:
    model = duration_graph()
    metadata = onnx_timing.expose_duration_output(model)
    assert metadata["duration_tensor"] == "clipped"
    assert metadata["public_output"] == "duration"
    assert {output.name for output in model.graph.output} == {"audio", "duration"}
    onnx_timing.validate_model_structure(
        model, duration_output="duration", token_input="tokens"
    )


def test_normalize_float_duration_and_validate_audio_consistency() -> None:
    result = onnx_timing.validate_runtime_outputs(
        np.zeros(1800, dtype=np.float32),
        np.asarray([1.0, 2.0], dtype=np.float32),
        token_count=2,
    )
    assert result["duration_frames"] == 3
    assert onnx_timing.normalize_durations([1.0, 2.0]).dtype == np.int64


def test_waveform_comparison_is_bit_exact() -> None:
    values = np.asarray([0.0, 1.0], dtype=np.float32)
    onnx_timing.compare_waveforms(values, values.copy())
    with pytest.raises(onnx_timing.TimingError, match="changed"):
        onnx_timing.compare_waveforms(values, np.asarray([0.0, 1.1], dtype=np.float32))
