"""Shared runtime-contract helpers for componentized Kokoro ONNX layouts.

``split-onnx-v1`` and ``cloning-onnx-v1`` both split one Kokoro model across
several ONNX graphs. The component-set rules are identical, so candidate,
registry and profile tooling validate them through one helper instead of one
copy per layout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SINGLE_LAYOUT = "single-onnx-v1"
SPLIT_LAYOUT = "split-onnx-v1"
CLONING_LAYOUT = "cloning-onnx-v1"
LAYOUTS = (SINGLE_LAYOUT, SPLIT_LAYOUT, CLONING_LAYOUT)

SPLIT_COMPONENTS = ("prosody", "curves", "decoder")
CLONING_COMPONENTS = (
    "reference_wavlm",
    "reference_encoders",
    "reference_mapper",
    "prosody",
    "curves",
    "decoder",
)
LAYOUT_COMPONENTS: dict[str, tuple[str, ...]] = {
    SPLIT_LAYOUT: SPLIT_COMPONENTS,
    CLONING_LAYOUT: CLONING_COMPONENTS,
}

STATIC_VOICE_MODE = "static"
REFERENCE_VOICE_MODE = "reference"
VOICE_MODES = (STATIC_VOICE_MODE, REFERENCE_VOICE_MODE)

REFERENCE_FORMAT = "akinvox-cloning-reference-v1"
SUPPORT_COMPONENTS = ("source_params",)

TUNER_ENROLLER_KIND = "kokoro-voicepack-tuner"
VOICEPACK_FORMAT = "kokoro-voicepack-v1"
VOICEPACK_SHAPE = (510, 1, 256)
ENROLLER_FIELDS = (
    "id",
    "kind",
    "input",
    "transcript_required",
    "min_seconds",
    "recommended_seconds",
    "max_seconds",
    "output",
    "model_component",
    "metadata_component",
)


class ContractError(ValueError):
    """Raised when a declared runtime contract is internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def voice_mode(runtime: Mapping[str, Any] | None) -> str:
    """Return the declared voice mode; absent metadata means static voices."""
    value = (runtime or {}).get("voice_mode")
    if value is None:
        return STATIC_VOICE_MODE
    _require(value in VOICE_MODES, f"Unsupported voice_mode: {value!r}")
    return str(value)


def is_reference_mode(runtime: Mapping[str, Any] | None) -> bool:
    return voice_mode(runtime) == REFERENCE_VOICE_MODE


def is_componentized(layout: str | None) -> bool:
    return layout in LAYOUT_COMPONENTS


def required_components(layout: str | None) -> tuple[str, ...]:
    _require(is_componentized(layout), f"Layout is not componentized: {layout!r}")
    return LAYOUT_COMPONENTS[str(layout)]


def validate_component_set(
    layout: str | None,
    contract_components: Mapping[str, Any] | Iterable[str] | None,
    artifact_components: Iterable[str],
) -> None:
    """Require exactly one artifact per declared component, with no extras."""
    _require(is_componentized(layout), f"Layout is not componentized: {layout!r}")
    declared = set(contract_components or {})
    artifacts = set(artifact_components)
    _require(bool(declared), f"{layout} contract must declare components")
    _require(
        declared == artifacts,
        f"{layout} components do not match contract: "
        f"expected {sorted(declared)}, got {sorted(artifacts)}",
    )
    expected = set(required_components(layout))
    _require(
        declared == expected,
        f"{layout} contract components must be {sorted(expected)}, "
        f"got {sorted(declared)}",
    )


def validate_reference_constraints(runtime: Mapping[str, Any]) -> None:
    """Check the reference-enrollment runtime constraints for a cloning model."""
    _require(
        is_reference_mode(runtime),
        "Reference constraints require voice_mode=reference",
    )
    _require(
        "speed_supported" in runtime,
        "Reference runtime must declare speed_supported explicitly",
    )
    _require(
        isinstance(runtime["speed_supported"], bool),
        "Reference runtime speed_supported must be boolean",
    )
    _require(
        "layout" in runtime and is_componentized(runtime.get("layout")),
        "Reference runtime must use a componentized Kokoro layout",
    )

    dimensions = runtime.get("style_dimensions")
    _require(
        isinstance(dimensions, Mapping),
        "Reference runtime must declare style_dimensions",
    )
    _require(
        dimensions.get("acoustic") == 128 and dimensions.get("duration") == 128,
        "Reference runtime style_dimensions must split 128 acoustic and 128 "
        "duration values",
    )

    reference = runtime.get("reference")
    _require(isinstance(reference, Mapping), "Reference runtime must declare reference")
    for field in (
        "format",
        "sample_rate",
        "identity_sample_rate",
        "min_seconds",
        "max_seconds",
        "memory_width",
        "style_width",
    ):
        _require(field in reference, f"Reference metadata is missing {field!r}")
    _require(
        reference["format"] == REFERENCE_FORMAT,
        f"Reference format must be {REFERENCE_FORMAT!r}",
    )
    _require(reference["sample_rate"] == 24000, "Reference sample_rate must be 24000")
    _require(
        reference["identity_sample_rate"] == 16000,
        "Reference identity_sample_rate must be 16000",
    )
    _require(reference["memory_width"] == 192, "Reference memory_width must be 192")
    _require(reference["style_width"] == 256, "Reference style_width must be 256")
    minimum = reference["min_seconds"]
    maximum = reference["max_seconds"]
    _require(
        isinstance(minimum, (int, float))
        and isinstance(maximum, (int, float))
        and not isinstance(minimum, bool)
        and not isinstance(maximum, bool)
        and 0 < minimum < maximum,
        "Reference min_seconds must be positive and below max_seconds",
    )

    voices = runtime.get("voices")
    _require(
        not voices,
        "Reference runtime must not declare a static voice roster",
    )
    _require(
        runtime.get("default_voice") is None,
        "Reference runtime must not declare a default voice",
    )


def validate_voice_enrollers(
    runtime: Mapping[str, Any] | None,
    *,
    model_components: Iterable[str] | None = None,
    metadata_components: Iterable[str] | None = None,
    model_id: str = "",
    model_version: str = "",
) -> None:
    """Check optional voice_enrollers declarations and component references."""
    enrollers = (runtime or {}).get("voice_enrollers")
    if enrollers is None:
        return
    _require(
        isinstance(enrollers, list) and enrollers,
        "voice_enrollers must be a non-empty list when present",
    )
    models = None if model_components is None else set(model_components)
    metadata = None if metadata_components is None else set(metadata_components)
    seen: set[str] = set()
    for enroller in enrollers:
        _require(isinstance(enroller, Mapping), "Enroller entries must be objects")
        missing = [field for field in ENROLLER_FIELDS if field not in enroller]
        _require(not missing, f"Enroller is missing {', '.join(missing)}")
        enroller_id = str(enroller["id"])
        _require(enroller_id not in seen, f"Duplicate enroller id: {enroller_id!r}")
        seen.add(enroller_id)
        _require(
            isinstance(enroller["transcript_required"], bool),
            f"Enroller {enroller_id}: transcript_required must be boolean",
        )
        minimum = enroller["min_seconds"]
        recommended = enroller["recommended_seconds"]
        maximum = enroller["max_seconds"]
        _require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
                for value in (minimum, recommended, maximum)
            )
            and minimum <= recommended <= maximum,
            f"Enroller {enroller_id}: durations must satisfy 0 < min <= recommended <= max",
        )
        output = enroller["output"]
        _require(isinstance(output, Mapping), f"Enroller {enroller_id}: output must be an object")
        _require(
            output.get("format") == VOICEPACK_FORMAT,
            f"Enroller {enroller_id}: output format must be {VOICEPACK_FORMAT!r}",
        )
        _require(
            tuple(output.get("shape") or ()) == VOICEPACK_SHAPE
            and output.get("dtype") == "float32",
            f"Enroller {enroller_id}: output must be float32 {list(VOICEPACK_SHAPE)}",
        )
        if models is not None:
            _require(
                str(enroller["model_component"]) in models,
                f"Enroller {enroller_id}: model component "
                f"{enroller['model_component']!r} is missing from the artifacts",
            )
        if metadata is not None:
            _require(
                str(enroller["metadata_component"]) in metadata,
                f"Enroller {enroller_id}: metadata component "
                f"{enroller['metadata_component']!r} is missing from the artifacts",
            )
        if str(enroller["kind"]) == TUNER_ENROLLER_KIND or enroller_id.startswith("inno-"):
            _require(
                model_id == "v1.0" and model_version == "1.0",
                f"Enroller {enroller_id}: the Inno voicepack tuner is only compatible "
                "with the Kokoro v1.0 model",
            )
            _require(
                str(enroller["input"]) == "reference-audio",
                f"Enroller {enroller_id}: tuner input must be reference audio",
            )
