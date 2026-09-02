"""Embed a SciPy-compatible zero-phase notch cascade in an ONNX graph."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class NotchError(ValueError):
    """Raised when an ONNX notch transform cannot be constructed."""


def iirnotch_coefficients(
    frequency_hz: float, quality: float, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized second-order coefficients used by scipy.signal.iirnotch."""
    if sample_rate <= 0.0:
        raise NotchError(f"Sample rate must be positive: {sample_rate}")
    if not 0.0 < frequency_hz < sample_rate / 2.0:
        raise NotchError(f"Notch frequency must be below Nyquist: {frequency_hz}")
    if quality <= 0.0:
        raise NotchError(f"Notch Q must be positive: {quality}")
    normalized = frequency_hz / (sample_rate / 2.0)
    angle = math.pi * normalized
    bandwidth = math.pi * normalized / quality
    beta = math.tan(bandwidth / 2.0)
    gain = 1.0 / (1.0 + beta)
    cosine = math.cos(angle)
    b = gain * np.asarray([1.0, -2.0 * cosine, 1.0], dtype=np.float64)
    a = np.asarray([1.0, -2.0 * gain * cosine, 2.0 * gain - 1.0], dtype=np.float64)
    return b.astype(np.float32), a.astype(np.float32)


def lfilter_zi(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Return the steady-state direct-form-II initial state for a biquad."""
    return np.asarray([1.0 - b[0], b[2] - a[2]], dtype=np.float32)


def _odd_pad(values: np.ndarray, padlen: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= padlen:
        raise NotchError("Audio is too short for the embedded filtfilt padding")
    left = 2.0 * values[0] - values[1 : padlen + 1][::-1]
    right = 2.0 * values[-1] - values[-padlen - 1 : -1][::-1]
    return np.concatenate((left, values, right))


def _lfilter(
    values: np.ndarray, b: np.ndarray, a: np.ndarray, zi: np.ndarray
) -> np.ndarray:
    state = np.asarray(zi, dtype=np.float64).copy()
    output = np.empty_like(np.asarray(values), dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    a64 = np.asarray(a, dtype=np.float64)
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        result = b64[0] * value + state[0]
        state[0] = b64[1] * value - a64[1] * result + state[1]
        state[1] = b64[2] * value - a64[2] * result
        output[index] = result
    return output.astype(np.float32)


def filtfilt_reference(values: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Reproduce SciPy's default odd-padded second-order filtfilt path."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    padlen = 3 * max(len(a), len(b))
    extended = _odd_pad(values, padlen)
    zi = lfilter_zi(b, a)
    forward = _lfilter(extended, b, a, zi * extended[0])
    backward = _lfilter(forward[::-1], b, a, zi * forward[-1])
    return backward[::-1][padlen:-padlen]


def apply_notch_filters(
    values: np.ndarray,
    frequencies_hz: list[float] | tuple[float, ...],
    quality: float,
    sample_rate: float,
) -> np.ndarray:
    """Apply the configured zero-phase notch filters in cascade."""
    output = np.asarray(values, dtype=np.float32).reshape(-1)
    for frequency_hz in frequencies_hz:
        b, a = iirnotch_coefficients(frequency_hz, quality, sample_rate)
        output = filtfilt_reference(output, b, a)
    return output


def _constant(name: str, values: Any) -> Any:
    from onnx import numpy_helper

    return numpy_helper.from_array(np.asarray(values), name=name)


def _scan_body(name: str, b: np.ndarray, a: np.ndarray) -> Any:
    from onnx import TensorProto, helper

    state = helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [4])
    sample = helper.make_tensor_value_info("sample_in", TensorProto.FLOAT, [])
    output_state = helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [4])
    output_sample = helper.make_tensor_value_info("sample_out", TensorProto.FLOAT, [])
    nodes = [
        helper.make_node("Gather", ["state_in", "index_0"], ["x1"], axis=0),
        helper.make_node("Gather", ["state_in", "index_1"], ["x2"], axis=0),
        helper.make_node("Gather", ["state_in", "index_2"], ["y1"], axis=0),
        helper.make_node("Gather", ["state_in", "index_3"], ["y2"], axis=0),
        helper.make_node("Mul", ["b0", "sample_in"], ["term0"]),
        helper.make_node("Mul", ["b1", "x1"], ["term1"]),
        helper.make_node("Mul", ["b2", "x2"], ["term2"]),
        helper.make_node("Mul", ["a1", "y1"], ["feedback1"]),
        helper.make_node("Mul", ["a2", "y2"], ["feedback2"]),
        helper.make_node("Add", ["term0", "term1"], ["sum01"]),
        helper.make_node("Add", ["sum01", "term2"], ["sum012"]),
        helper.make_node("Sub", ["sum012", "feedback1"], ["sum012a"]),
        helper.make_node("Sub", ["sum012a", "feedback2"], ["sample_out"]),
        helper.make_node("Unsqueeze", ["sample_in", "axis_0"], ["state_0"]),
        helper.make_node("Unsqueeze", ["x1", "axis_0"], ["state_1"]),
        helper.make_node("Unsqueeze", ["sample_out", "axis_0"], ["state_2"]),
        helper.make_node("Unsqueeze", ["y1", "axis_0"], ["state_3"]),
        helper.make_node(
            "Concat",
            ["state_0", "state_1", "state_2", "state_3"],
            ["state_out"],
            axis=0,
        ),
    ]
    initializers = [
        _constant("b0", b[0]),
        _constant("b1", b[1]),
        _constant("b2", b[2]),
        _constant("a1", a[1]),
        _constant("a2", a[2]),
        _constant("index_0", np.asarray(0, dtype=np.int64)),
        _constant("index_1", np.asarray(1, dtype=np.int64)),
        _constant("index_2", np.asarray(2, dtype=np.int64)),
        _constant("index_3", np.asarray(3, dtype=np.int64)),
        _constant("axis_0", np.asarray([0], dtype=np.int64)),
    ]
    return helper.make_graph(
        nodes,
        name,
        [state, sample],
        [output_state, output_sample],
        initializer=initializers,
    )


def _append_scan(
    graph: Any,
    source: str,
    target: str,
    b: np.ndarray,
    a: np.ndarray,
    state_name: str,
    name: str,
) -> None:
    from onnx import helper

    body = _scan_body(f"{name}_body", b, a)
    graph.node.append(
        helper.make_node(
            "Scan",
            [state_name, source],
            [f"{name}_final_state", target],
            name=name,
            num_scan_inputs=1,
            body=body,
        )
    )


def _filter_once(
    graph: Any,
    source: str,
    target: str,
    b: np.ndarray,
    a: np.ndarray,
    name: str,
) -> None:
    from onnx import helper

    padlen = 3 * max(len(a), len(b))
    zero = f"{name}_zero"
    last = f"{name}_last"
    axis = f"{name}_axis"
    step = f"{name}_step"
    graph.initializer.extend(
        [
            _constant(zero, np.asarray(0, dtype=np.int64)),
            _constant(last, np.asarray(-1, dtype=np.int64)),
            _constant(f"{name}_one", np.asarray(1, dtype=np.int64)),
            _constant(f"{name}_negative_one", np.asarray(-1, dtype=np.int64)),
            _constant(axis, np.asarray([0], dtype=np.int64)),
            _constant(step, np.asarray([1], dtype=np.int64)),
            _constant(f"{name}_two", np.asarray(2.0, dtype=np.float32)),
            _constant(
                f"{name}_reverse_pad_indices",
                np.arange(padlen - 1, -1, -1, dtype=np.int64),
            ),
            _constant(f"{name}_left_start", np.asarray([1], dtype=np.int64)),
            _constant(f"{name}_left_end", np.asarray([padlen + 1], dtype=np.int64)),
            _constant(f"{name}_right_start", np.asarray([-padlen - 1], dtype=np.int64)),
            _constant(f"{name}_right_end", np.asarray([-1], dtype=np.int64)),
            _constant(f"{name}_trim_start", np.asarray([padlen], dtype=np.int64)),
            _constant(f"{name}_negative_pad", np.asarray([padlen], dtype=np.int64)),
            _constant(f"{name}_state_shape", np.asarray([4], dtype=np.int64)),
        ]
    )
    graph.node.extend(
        [
            helper.make_node("Gather", [source, zero], [f"{name}_first"], axis=0),
            helper.make_node("Gather", [source, last], [f"{name}_last_value"], axis=0),
            helper.make_node(
                "Slice",
                [source, f"{name}_left_start", f"{name}_left_end", axis, step],
                [f"{name}_left_part"],
            ),
            helper.make_node(
                "Slice",
                [source, f"{name}_right_start", f"{name}_right_end", axis, step],
                [f"{name}_right_part"],
            ),
            helper.make_node(
                "Gather",
                [f"{name}_left_part", f"{name}_reverse_pad_indices"],
                [f"{name}_left_reverse"],
                axis=0,
            ),
            helper.make_node(
                "Gather",
                [f"{name}_right_part", f"{name}_reverse_pad_indices"],
                [f"{name}_right_reverse"],
                axis=0,
            ),
            helper.make_node(
                "Mul", [f"{name}_first", f"{name}_two"], [f"{name}_left_double"]
            ),
            helper.make_node(
                "Mul",
                [f"{name}_last_value", f"{name}_two"],
                [f"{name}_right_double"],
            ),
            helper.make_node(
                "Sub",
                [f"{name}_left_double", f"{name}_left_reverse"],
                [f"{name}_left"],
            ),
            helper.make_node(
                "Sub",
                [f"{name}_right_double", f"{name}_right_reverse"],
                [f"{name}_right"],
            ),
            helper.make_node(
                "Concat",
                [f"{name}_left", source, f"{name}_right"],
                [f"{name}_extended"],
                axis=0,
            ),
            helper.make_node(
                "Gather",
                [f"{name}_extended", zero],
                [f"{name}_extended_first"],
                axis=0,
            ),
            helper.make_node("Shape", [source], [f"{name}_shape"]),
            helper.make_node(
                "Gather", [f"{name}_shape", zero], [f"{name}_length"], axis=0
            ),
            helper.make_node("Shape", [f"{name}_extended"], [f"{name}_extended_shape"]),
            helper.make_node(
                "Gather",
                [f"{name}_extended_shape", zero],
                [f"{name}_extended_length"],
                axis=0,
            ),
            helper.make_node(
                "Sub",
                [f"{name}_extended_length", f"{name}_one"],
                [f"{name}_full_start"],
            ),
            helper.make_node(
                "Range",
                [f"{name}_full_start", f"{name}_negative_one", f"{name}_negative_one"],
                [f"{name}_reverse_indices"],
            ),
            helper.make_node(
                "Add",
                [f"{name}_length", f"{name}_negative_pad"],
                [f"{name}_trim_end"],
            ),
            helper.make_node(
                "Expand",
                [f"{name}_extended_first", f"{name}_state_shape"],
                [f"{name}_forward_state"],
            ),
        ]
    )
    _append_scan(
        graph,
        f"{name}_extended",
        f"{name}_forward",
        b,
        a,
        f"{name}_forward_state",
        f"{name}_forward_scan",
    )
    graph.node.extend(
        [
            helper.make_node(
                "Gather", [f"{name}_forward", last], [f"{name}_forward_last"], axis=0
            ),
            helper.make_node(
                "Gather",
                [f"{name}_forward", f"{name}_reverse_indices"],
                [f"{name}_backward_input"],
                axis=0,
            ),
            helper.make_node(
                "Expand",
                [f"{name}_forward_last", f"{name}_state_shape"],
                [f"{name}_backward_state"],
            ),
        ]
    )
    _append_scan(
        graph,
        f"{name}_backward_input",
        f"{name}_backward",
        b,
        a,
        f"{name}_backward_state",
        f"{name}_backward_scan",
    )
    graph.node.extend(
        [
            helper.make_node(
                "Gather",
                [f"{name}_backward", f"{name}_reverse_indices"],
                [f"{name}_untrimmed"],
                axis=0,
            ),
            helper.make_node(
                "Slice",
                [
                    f"{name}_untrimmed",
                    f"{name}_trim_start",
                    f"{name}_trim_end",
                    axis,
                    step,
                ],
                [target],
            ),
        ]
    )


def embed_notch_filters(
    path: Path,
    *,
    frequencies_hz: list[float] | tuple[float, ...],
    quality: float,
    sample_rate: float,
) -> dict[str, Any]:
    """Append the configured zero-phase notch cascade to an ONNX model."""
    try:
        import onnx
    except ImportError as exc:
        raise NotchError("onnx is required to embed notch filters") from exc
    if not frequencies_hz:
        raise NotchError("At least one notch frequency is required")

    model = onnx.load(str(path), load_external_data=True)
    outputs = [output for output in model.graph.output if output.name == "audio"]
    if len(outputs) != 1:
        raise NotchError("Expected exactly one audio output named 'audio'")
    outputs[0].name = "audio"
    for node in model.graph.node:
        node.output[:] = [
            "audio_unfiltered" if name == "audio" else name for name in node.output
        ]

    source = "audio_unfiltered"
    records: list[dict[str, Any]] = []
    for index, frequency_hz in enumerate(frequencies_hz):
        b, a = iirnotch_coefficients(float(frequency_hz), quality, sample_rate)
        target = "audio" if index == len(frequencies_hz) - 1 else f"audio_notch_{index}"
        _filter_once(model.graph, source, target, b, a, f"notch_{index}")
        records.append(
            {
                "frequency_hz": float(frequency_hz),
                "q": float(quality),
                "b": [float(value) for value in b],
                "a": [float(value) for value in a],
                "padlen": 3 * max(len(a), len(b)),
            }
        )
        source = target

    metadata = {prop.key: prop.value for prop in model.metadata_props}
    metadata["postprocess"] = "embedded-zero-phase-notch-v1"
    metadata["postprocess_frequencies_hz"] = ",".join(
        str(float(value)) for value in frequencies_hz
    )
    metadata["postprocess_q"] = str(float(quality))
    while model.metadata_props:
        model.metadata_props.pop()
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return {
        "kind": "embedded-zero-phase-notch-v1",
        "sample_rate": float(sample_rate),
        "frequencies_hz": [float(value) for value in frequencies_hz],
        "q": float(quality),
        "reference": "scipy.signal.iirnotch + scipy.signal.filtfilt",
        "filters": records,
    }
