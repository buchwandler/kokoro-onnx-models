"""Validate and expose Kokoro token-duration tensors in ONNX graphs."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


class TimingError(ValueError):
    """Raised when an ONNX graph cannot provide an unambiguous timing tensor."""


def _onnx():
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - exercised by CLI environments
        raise TimingError("onnx is required for timing graph operations") from exc
    return onnx


def load_model(path: Path):
    onnx = _onnx()
    try:
        model = onnx.load(str(path), load_external_data=True)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise TimingError(f"Cannot load valid ONNX model {path}: {exc}") from exc
    return model


def _consumers(model: Any) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            result[value].append(node)
    return result


def reaches_op(
    consumers: dict[str, list[Any]], value: str, op_type: str, *, max_hops: int = 5
) -> bool:
    """Return whether ``value`` reaches ``op_type`` within ``max_hops`` nodes."""
    queue: deque[tuple[str, int]] = deque([(value, 0)])
    visited: set[tuple[str, int]] = set()
    while queue:
        current, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for node in consumers.get(current, []):
            if node.op_type == op_type:
                return True
            state = (node.output[0], hops + 1) if node.output else ("", hops + 1)
            if node.output and state not in visited:
                visited.add(state)
                queue.append(state)
    return False


def find_exposed_duration_output(model: Any) -> str | None:
    """Find a conventional public duration output, if one already exists."""
    outputs = {value.name for value in model.graph.output}
    for name in ("duration", "pred_dur", "durations"):
        if name in outputs:
            return name
    return None


def find_kokoro_duration_tensor(model: Any, *, max_hops: int = 5) -> str | None:
    """Find the internal Kokoro duration tensor by graph topology.

    Kokoro duration prediction rounds values and then uses them in the alignment
    expansion path, which eventually reaches CumSum. Tensor names are not used
    because quantized exports rename them between quality tiers.
    """
    exposed = find_exposed_duration_output(model)
    if exposed is not None:
        return exposed

    consumers = _consumers(model)
    candidates: list[str] = []
    for node in model.graph.node:
        if node.op_type != "Round" or not node.output:
            continue
        rounded = node.output[0]
        if not reaches_op(consumers, rounded, "CumSum", max_hops=max_hops):
            continue
        clips = [item for item in consumers.get(rounded, []) if item.op_type == "Clip"]
        if len(clips) > 1:
            raise TimingError(
                f"Ambiguous duration graph: Round output {rounded!r} has multiple Clip nodes"
            )
        candidates.append(clips[0].output[0] if clips else rounded)

    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise TimingError(
            "Ambiguous duration graph: multiple Round-to-CumSum chains found: "
            + ", ".join(unique)
        )
    return unique[0] if unique else None


def _value_info(model: Any, name: str) -> Any | None:
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.name == name:
            return value
    return None


def _shape_from_info(info: Any) -> list[Any]:
    tensor = info.type.tensor_type
    dimensions: list[Any] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dimensions.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dimensions.append(dim.dim_param)
        else:
            dimensions.append(None)
    return dimensions


def _dtype_name(onnx: Any, elem_type: int) -> str:
    names = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.DOUBLE: "float64",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT32: "int32",
    }
    return names.get(elem_type, onnx.TensorProto.DataType.Name(elem_type).lower())


def _dtype_value(onnx: Any, name: str) -> int:
    values = {
        "float32": onnx.TensorProto.FLOAT,
        "float16": onnx.TensorProto.FLOAT16,
        "float64": onnx.TensorProto.DOUBLE,
        "int64": onnx.TensorProto.INT64,
        "int32": onnx.TensorProto.INT32,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise TimingError(f"Unsupported timing dtype {name!r}") from exc


def expose_duration_output(
    model: Any, *, public_name: str = "duration", public_dtype: str | None = None
) -> dict[str, Any]:
    """Expose the detected duration tensor and return transform metadata.

    A Cast is added only when a caller explicitly requests a public dtype. The
    duration predictor and all existing waveform nodes remain unchanged.
    """
    onnx = _onnx()
    tensor_name = find_kokoro_duration_tensor(model)
    if tensor_name is None:
        raise TimingError("No unambiguous Kokoro duration tensor was found")
    if any(value.name == public_name for value in model.graph.output):
        return {
            "duration_tensor": tensor_name,
            "public_output": public_name,
            "public_dtype": _dtype_name(
                onnx,
                next(
                    value for value in model.graph.output if value.name == public_name
                ).type.tensor_type.elem_type,
            ),
            "transformed": False,
        }

    inferred_model = onnx.shape_inference.infer_shapes(model)
    inferred = _value_info(inferred_model, tensor_name)
    if inferred is None:
        raise TimingError(f"Duration tensor {tensor_name!r} has no inferred type")
    source_type = inferred.type.tensor_type.elem_type
    output_name = public_name
    output_type = source_type
    if public_dtype is not None and public_dtype != _dtype_name(onnx, source_type):
        output_name = f"{public_name}_cast"
        output_type = _dtype_value(onnx, public_dtype)
        model.graph.node.append(
            onnx.helper.make_node(
                "Cast",
                [tensor_name],
                [output_name],
                name="ExposeKokoroDurationCast",
                to=output_type,
            )
        )

    output_info = onnx.helper.make_tensor_value_info(
        public_name, output_type, _shape_from_info(inferred)
    )
    if output_name != public_name:
        model.graph.node.append(
            onnx.helper.make_node(
                "Identity", [output_name], [public_name], name="ExposeKokoroDuration"
            )
        )
    else:
        model.graph.node.append(
            onnx.helper.make_node(
                "Identity", [tensor_name], [public_name], name="ExposeKokoroDuration"
            )
        )
    model.graph.output.append(output_info)
    return {
        "duration_tensor": tensor_name,
        "public_output": public_name,
        "public_dtype": _dtype_name(onnx, output_type),
        "transformed": True,
    }


def normalize_durations(values: Any) -> np.ndarray:
    """Normalize native integer or integer-valued floating duration output."""
    durations = np.asarray(values)
    if durations.ndim == 0 or durations.size == 0:
        raise TimingError("Duration output must be a non-empty tensor")
    if not np.issubdtype(durations.dtype, np.number):
        raise TimingError(f"Duration output has non-numeric dtype {durations.dtype}")
    if not np.isfinite(durations).all():
        raise TimingError("Duration output contains non-finite values")
    rounded = np.rint(durations)
    if not np.allclose(durations, rounded, atol=1e-4):
        raise TimingError("Duration output is not integer-valued")
    normalized = rounded.astype(np.int64)
    if np.any(normalized < 1):
        raise TimingError("Duration output contains values below one frame")
    return normalized.reshape(-1)


def validate_runtime_outputs(
    audio: Any,
    duration: Any,
    *,
    token_count: int,
    samples_per_frame: int = 600,
    ratio_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Validate ORT outputs and return normalized timing evidence."""
    normalized = normalize_durations(duration)
    if normalized.size != token_count:
        raise TimingError(
            f"Duration count {normalized.size} does not match token count {token_count}"
        )
    audio_values = np.asarray(audio)
    if (
        not np.issubdtype(audio_values.dtype, np.number)
        or not np.isfinite(audio_values).all()
    ):
        raise TimingError("Audio output must contain finite numeric values")
    total_frames = int(normalized.sum())
    actual_ratio = audio_values.size / total_frames
    if not np.isclose(
        actual_ratio, samples_per_frame, rtol=ratio_tolerance, atol=ratio_tolerance
    ):
        raise TimingError(
            f"Audio/duration ratio {actual_ratio:.6f} does not match "
            f"{samples_per_frame} samples per frame"
        )
    return {
        "duration_frames": total_frames,
        "duration_count": int(normalized.size),
        "audio_samples": int(audio_values.size),
        "samples_per_frame": int(samples_per_frame),
        "audio_samples_per_frame": float(actual_ratio),
    }


def compare_waveforms(original: Any, transformed: Any) -> None:
    """Require graph surgery to preserve the original waveform exactly."""
    before = np.asarray(original)
    after = np.asarray(transformed)
    if before.shape != after.shape:
        raise TimingError(
            f"Waveform shape changed from {before.shape} to {after.shape}"
        )
    if not np.array_equal(before, after):
        raise TimingError("Graph surgery changed waveform values")


def validate_model_structure(
    model: Any, *, duration_output: str = "duration", token_input: str | None = None
) -> dict[str, Any]:
    """Run checker and validate the declared timing output structure."""
    onnx = _onnx()
    onnx.checker.check_model(model)
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if token_input is not None and token_input not in inputs:
        raise TimingError(f"Missing token input {token_input!r}")
    if duration_output not in outputs:
        raise TimingError(f"Missing timing output {duration_output!r}")
    info = outputs[duration_output]
    elem_type = info.type.tensor_type.elem_type
    if elem_type == onnx.TensorProto.UNDEFINED:
        raise TimingError("Timing output has no tensor type")
    if not info.type.tensor_type.HasField("shape"):
        raise TimingError("Timing output has no tensor shape")
    return {
        "output": duration_output,
        "dtype": _dtype_name(onnx, elem_type),
        "shape": _shape_from_info(info),
    }


def transform_file(
    source: Path,
    target: Path,
    *,
    public_name: str = "duration",
    public_dtype: str | None = None,
) -> dict[str, Any]:
    """Patch one ONNX file and return reproducible transform metadata."""
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    model = load_model(source)
    metadata = expose_duration_output(
        model, public_name=public_name, public_dtype=public_dtype
    )
    validate_model_structure(model, duration_output=public_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    _onnx().save(model, str(target))
    transformed_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata.update(
        {
            "kind": "onnx-expose-kokoro-duration-v1",
            "source_sha256": source_hash,
            "transformed_sha256": transformed_hash,
            "waveform_identical": None,
        }
    )
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--public-dtype", choices=("float32", "float16", "float64", "int64", "int32")
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    metadata = transform_file(args.source, args.target, public_dtype=args.public_dtype)
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
