#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.export_validation import RANDOM_SOURCE_OPS
except ModuleNotFoundError:
    try:
        from export_validation import RANDOM_SOURCE_OPS
    except ModuleNotFoundError:
        import importlib.util

        validation_spec = importlib.util.spec_from_file_location(
            "_verify_candidate_export_validation",
            Path(__file__).with_name("export_validation.py"),
        )
        if validation_spec is None or validation_spec.loader is None:
            raise
        validation_module = importlib.util.module_from_spec(validation_spec)
        validation_spec.loader.exec_module(validation_module)
        RANDOM_SOURCE_OPS = validation_module.RANDOM_SOURCE_OPS

TARGET_REPOSITORY = "buchwandler/kokoro-onnx-models"
ALLOWED_FILES = {"release-manifest.json", "SHA256SUMS", "release-notes.md"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BINARY_SUFFIXES = {".bin", ".onnx", ".npz", ".pt", ".zip", ".safetensors"}


class CandidateError(ValueError):
    """Raised when a release candidate is not publishable."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateError(message)


def _validate_json_asset(path: Path, role: str) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"Invalid JSON {role} asset {path.name}: {exc}") from exc
    if role == "vocab":
        _require(
            isinstance(value, dict), f"Vocabulary asset {path.name} must be an object"
        )
        _require(
            all(
                isinstance(key, str)
                and isinstance(token, int)
                and not isinstance(token, bool)
                for key, token in value.items()
            ),
            f"Vocabulary asset {path.name} must map string keys to integer token IDs",
        )


def _onnx_type_name(onnx: Any, elem_type: int) -> str:
    names = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.DOUBLE: "float64",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT32: "int32",
    }
    return names.get(elem_type, onnx.TensorProto.DataType.Name(elem_type).lower())


def _validate_tensor_contract(
    path: Path, label: str, expected: Any, actual: dict[str, str]
) -> None:
    if isinstance(expected, dict):
        for name, expected_type in expected.items():
            _require(
                name in actual, f"ONNX model {path.name} is missing {label} {name!r}"
            )
            _require(
                actual[name] == expected_type,
                f"ONNX model {path.name} {label} {name!r} is {actual[name]}, "
                f"expected {expected_type}",
            )
    elif isinstance(expected, list):
        missing = sorted(set(expected) - actual.keys())
        _require(
            not missing,
            f"ONNX model {path.name} is missing {label}: {', '.join(missing)}",
        )
    else:
        raise CandidateError(f"Invalid ONNX {label} contract in {path.name}")


def _validate_timing_contract(
    contract: dict[str, Any],
    *,
    actual_outputs: dict[str, str] | None = None,
    label: str = "ONNX contract",
) -> None:
    timing = contract.get("timing")
    if timing is None:
        return
    _require(isinstance(timing, dict), f"{label} timing must be an object")
    for field in (
        "kind",
        "output",
        "unit",
        "samples_per_frame",
        "includes_boundary_tokens",
    ):
        _require(field in timing, f"{label} timing is missing {field!r}")
    _require(
        timing["kind"] == "token-duration-v1",
        f"{label} timing kind is unsupported",
    )
    _require(timing["unit"] == "frame", f"{label} timing unit must be frame")
    _require(
        isinstance(timing["samples_per_frame"], int)
        and timing["samples_per_frame"] > 0,
        f"{label} samples_per_frame must be positive",
    )
    _require(
        isinstance(timing["includes_boundary_tokens"], bool),
        f"{label} boundary-token semantics must be boolean",
    )
    if actual_outputs is not None:
        output = timing["output"]
        _require(
            output in actual_outputs,
            f"ONNX model timing output {output!r} is missing",
        )
        _require(
            actual_outputs[output]
            in {"int64", "int32", "float16", "float32", "float64"},
            f"ONNX timing output {output!r} is not numeric",
        )


def _validate_exporter_contract(manifest: dict[str, Any]) -> None:
    contract = manifest["onnx_contract"]
    timing = contract.get("timing")
    if timing is None:
        return
    effective = contract
    component = timing.get("component") if isinstance(timing, dict) else None
    if component:
        effective = (contract.get("components") or {}).get(component) or {}
    _validate_timing_contract(effective, label="Manifest ONNX contract")
    output = timing.get("output")
    _require(
        output in (effective.get("outputs") or {}),
        f"Timing output {output!r} is not declared in its output contract",
    )
    exporter = (manifest.get("provenance") or {}).get("exporter") or {}
    if exporter:
        exported = exporter.get("outputs") or []
        _require(
            output in exported,
            f"Exporter provenance is missing timing output {output!r}",
        )


def _validate_transform_provenance(manifest: dict[str, Any]) -> None:
    transform = manifest.get("transform") or {}
    if transform.get("type") != "onnx-expose-kokoro-duration-v1":
        return
    _require(transform.get("version") == 1, "Unsupported ONNX timing transform version")
    records = transform.get("assets") or []
    _require(records, "ONNX timing transform has no asset provenance")
    by_hash = {record.get("transformed_sha256"): record for record in records}
    model_assets = [
        asset for asset in manifest["assets"] if asset.get("role") == "model"
    ]
    for asset in model_assets:
        record = by_hash.get(asset.get("sha256"))
        if record is None:
            continue
        _require(
            record.get("waveform_identical") in {True, None},
            "Invalid waveform provenance",
        )
        _require(
            bool(record.get("source_sha256")), "Transform is missing source SHA-256"
        )
        _require(
            record.get("public_output") == "duration", "Transform must expose duration"
        )
        return
    raise CandidateError("No model asset matches ONNX timing transform provenance")


def _validate_onnx_asset(
    path: Path, contract: dict[str, Any], component: str | None = None
) -> None:
    try:
        import onnx
    except ImportError:
        return
    try:
        model = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise CandidateError(f"Invalid ONNX model {path.name}: {exc}") from exc
    component_contracts = contract.get("components") or {}
    effective = component_contracts.get(component, contract) if component else contract
    actual_inputs = {
        value.name: _onnx_type_name(onnx, value.type.tensor_type.elem_type)
        for value in model.graph.input
    }
    _validate_tensor_contract(path, "input", effective.get("inputs", {}), actual_inputs)
    actual_outputs = {
        value.name: _onnx_type_name(onnx, value.type.tensor_type.elem_type)
        for value in model.graph.output
    }
    _validate_tensor_contract(
        path, "output", effective.get("outputs", {}), actual_outputs
    )
    _require(bool(actual_outputs), f"ONNX model {path.name} has no outputs")
    timing_contract = contract
    if component and (contract.get("timing") or {}).get("component") == component:
        timing_contract = contract
    elif component:
        timing_contract = effective
    _validate_timing_contract(
        timing_contract,
        actual_outputs=actual_outputs,
        label=f"{path.name} ONNX contract",
    )


def _validate_voice_asset(
    path: Path, asset: dict[str, Any], runtime: dict[str, Any]
) -> None:
    asset_format = asset["format"]
    if asset_format == "numpy-npz":
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as archive:
                _require(bool(archive.files), f"Voice archive {path.name} is empty")
                missing = sorted(set(runtime["voices"]) - set(archive.files))
                _require(
                    not missing,
                    f"Voice archive {path.name} is missing voices: {', '.join(missing)}",
                )
                for name in archive.files:
                    values = archive[name]
                    _require(
                        bool(np.isfinite(values).all()),
                        f"Voice archive {path.name} contains non-finite values",
                    )
                    _require(
                        values.dtype == np.float32,
                        f"Voice archive {path.name} voice {name} must use float32",
                    )
                    handling = asset.get("handling") or {}
                    if "style_width" in handling:
                        width = handling["style_width"]
                        _require(
                            values.ndim in {2, 3} and values.shape[-1] == width,
                            f"Voice archive {path.name} voice {name} must have style width {width}",
                        )
        except ImportError:
            import zipfile

            _require(
                zipfile.is_zipfile(path),
                f"Voice archive {path.name} is not a NumPy archive",
            )
        except (OSError, ValueError) as exc:
            raise CandidateError(f"Invalid voice archive {path.name}: {exc}") from exc
    elif asset_format == "raw-float32-le":
        _require(
            path.stat().st_size % 4 == 0,
            f"Raw voice asset {path.name} is not float32-aligned",
        )


def _validate_asset_format(
    path: Path, asset: dict[str, Any], manifest: dict[str, Any]
) -> None:
    role = asset["role"]
    if asset["format"] == "json" or role in {"bundle", "config", "vocab"}:
        _validate_json_asset(path, role)
    elif role == "model" and asset["format"] == "onnx":
        _validate_onnx_asset(path, manifest["onnx_contract"], asset.get("component"))
    elif role == "voices":
        _validate_voice_asset(path, asset, manifest["runtime"])


def _validate_checksums(candidate: Path, assets: list[dict[str, Any]]) -> None:
    sums_path = candidate / "SHA256SUMS"
    _require(sums_path.is_file(), "Candidate is missing SHA256SUMS")
    expected = {asset["name"]: asset["sha256"] for asset in assets}
    actual: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CandidateError(f"Cannot read SHA256SUMS: {exc}") from exc
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        _require(len(parts) == 2, f"Invalid SHA256SUMS line: {line!r}")
        digest, name = parts[0], parts[1].strip()
        _require(name not in actual, f"Duplicate SHA256SUMS entry: {name}")
        _require(
            SHA256_RE.fullmatch(digest) is not None,
            f"Invalid SHA-256 in SHA256SUMS: {name}",
        )
        actual[name] = digest
    _require(actual == expected, "SHA256SUMS does not exactly match manifest assets")


def _validate_manifest_identity(
    manifest: dict[str, Any],
    *,
    expected_tag: str | None,
    expected_profile: str | None,
    expected_builder_commit: str | None,
    allow_restricted: bool,
) -> None:
    required = {
        "schema",
        "runtime_contract",
        "repository",
        "tag",
        "profile",
        "model_version",
        "generated_at",
        "source",
        "license",
        "publication",
        "runtime",
        "onnx_contract",
        "assets",
    }
    _require(
        required <= manifest.keys(), "Manifest is missing required schema-2 fields"
    )
    _require(manifest["schema"] == 2, "Manifest schema must be 2")
    _require(manifest["runtime_contract"] == 1, "Unsupported runtime contract")
    _require(
        manifest["repository"] == TARGET_REPOSITORY, "Manifest repository is incorrect"
    )
    _require(bool(manifest["tag"]), "Manifest tag must not be empty")
    if expected_tag is not None:
        _require(
            manifest["tag"] == expected_tag,
            f"Manifest tag is {manifest['tag']!r}, expected {expected_tag!r}",
        )
    if expected_profile is not None:
        _require(
            manifest["profile"] == expected_profile,
            f"Manifest profile is {manifest['profile']!r}, expected {expected_profile!r}",
        )
    _require(
        bool(manifest["publication"].get("enabled")) or allow_restricted,
        "Publication is disabled",
    )
    if expected_builder_commit is not None:
        builder = manifest.get("builder") or {}
        _require(
            builder.get("commit") == expected_builder_commit,
            f"Builder commit is {builder.get('commit')!r}, expected {expected_builder_commit!r}",
        )


def _validate_checkpoint_provenance(manifest: dict[str, Any]) -> None:
    if manifest.get("profile") != "de-thorsten":
        return
    provenance = manifest.get("provenance") or {}
    _require(isinstance(provenance, dict), "Checkpoint provenance must be an object")
    source = provenance.get("source_artifacts") or {}
    model = source.get("model") or {}
    voice = (source.get("voices") or {}).get("thorsten") or {}
    _require(
        model.get("path") == "model_ep5.pth", "Thorsten checkpoint path is not epoch 5"
    )
    _require(
        voice.get("path") == "voices/thorsten_ep5.pt",
        "Thorsten voice path is not epoch 5",
    )
    for label, value in (
        ("model", model.get("sha256")),
        ("model config", model.get("config_sha256")),
        ("Thorsten voice", voice.get("sha256")),
    ):
        _require(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"Thorsten provenance is missing a valid {label} SHA-256",
        )
    exporter = provenance.get("exporter") or {}
    for field in (
        "kokoro_version",
        "torch_version",
        "onnx_version",
        "onnxruntime_version",
        "python_version",
        "opset",
        "outputs",
        "random_source_ops",
        "waveform_validation",
    ):
        _require(
            field in exporter,
            f"Thorsten provenance is missing exporter field {field!r}",
        )
    _require(exporter["opset"] == 17, "Thorsten exporter opset must be 17")
    _require(
        exporter["outputs"] == ["audio", "duration"],
        "Thorsten exporter outputs are incomplete",
    )
    random_ops = exporter.get("random_source_ops")
    _require(
        isinstance(random_ops, list) and bool(random_ops),
        "Thorsten exporter provenance has no stochastic source operators",
    )
    _require(
        set(random_ops) <= RANDOM_SOURCE_OPS,
        "Thorsten exporter provenance contains unsupported random operators",
    )
    decoder_reconstruction = exporter.get("decoder_reconstruction")
    _require(
        isinstance(decoder_reconstruction, dict),
        "Thorsten exporter is missing decoder reconstruction provenance",
    )
    _require(
        decoder_reconstruction.get("reference_backend") == "torch.istft",
        "Thorsten decoder reference backend must be torch.istft",
    )
    _require(
        decoder_reconstruction.get("backend") == "exact-convtranspose-istft-v1",
        "Thorsten decoder export backend must be exact-convtranspose-istft-v1",
    )
    for field in ("one_sided_bin_scaling", "window_envelope_normalization"):
        _require(
            decoder_reconstruction.get(field) is True,
            f"Thorsten decoder reconstruction is missing {field}",
        )
    native_patched = decoder_reconstruction.get("native_patched_validation")
    _require(
        isinstance(native_patched, dict),
        "Thorsten decoder is missing native/patched validation",
    )
    max_abs_error = native_patched.get("max_abs_error")
    _require(
        isinstance(max_abs_error, (int, float))
        and not isinstance(max_abs_error, bool)
        and max_abs_error <= 1.0e-4,
        "Thorsten native/patched reconstruction error exceeds 1e-4",
    )
    waveform_validation = exporter.get("waveform_validation")
    _require(
        isinstance(waveform_validation, dict)
        and isinstance(waveform_validation.get("cases"), list)
        and bool(waveform_validation["cases"]),
        "Thorsten exporter has no waveform validation cases",
    )
    for case in waveform_validation["cases"]:
        _require(isinstance(case, dict), "Thorsten waveform case must be an object")
        for metric_name in ("native", "patched", "onnx"):
            _require(
                isinstance(case.get(metric_name), dict),
                f"Thorsten waveform case is missing {metric_name} metrics",
            )


def verify_candidate(
    candidate: Path,
    *,
    expected_tag: str | None = None,
    expected_profile: str | None = None,
    expected_builder_commit: str | None = None,
    allow_restricted: bool = False,
) -> dict[str, Any]:
    candidate = candidate.resolve()
    manifest_path = candidate / "release-manifest.json"
    _require(candidate.is_dir(), f"Candidate directory does not exist: {candidate}")
    _require(manifest_path.is_file(), "Candidate is missing release-manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"Invalid release manifest: {exc}") from exc
    _require(isinstance(manifest, dict), "Release manifest must be an object")
    _validate_manifest_identity(
        manifest,
        expected_tag=expected_tag,
        expected_profile=expected_profile,
        expected_builder_commit=expected_builder_commit,
        allow_restricted=allow_restricted,
    )

    _validate_checkpoint_provenance(manifest)
    assets = manifest["assets"]
    _require(
        isinstance(assets, list) and assets, "Manifest assets must be a non-empty list"
    )
    _validate_exporter_contract(manifest)
    _validate_transform_provenance(manifest)
    names: set[str] = set()
    slots: set[tuple[str, str, str | None, str | None]] = set()
    model_components: set[str] = set()
    model_count = voice_count = 0
    for asset in assets:
        _require(isinstance(asset, dict), "Manifest asset must be an object")
        for field in ("name", "role", "format", "size", "sha256"):
            _require(field in asset, f"Asset is missing {field}: {asset!r}")
        name = asset["name"]
        _require(
            isinstance(name, str) and name and name not in names,
            f"Duplicate asset name: {name!r}",
        )
        names.add(name)
        asset_path = (candidate / name).resolve()
        _require(
            candidate == asset_path or candidate in asset_path.parents,
            f"Asset escapes candidate: {name!r}",
        )
        _require(asset_path.is_file(), f"Manifest asset is missing: {name}")
        _require(
            isinstance(asset["size"], int) and asset["size"] > 0,
            f"Invalid asset size: {name}",
        )
        _require(
            SHA256_RE.fullmatch(asset["sha256"]) is not None,
            f"Invalid asset SHA-256: {name}",
        )
        _require(
            asset_path.stat().st_size == asset["size"], f"Size mismatch for {name}"
        )
        _require(sha256(asset_path) == asset["sha256"], f"SHA-256 mismatch for {name}")
        role = asset["role"]
        slot = (
            role,
            asset["format"],
            asset.get("quality"),
            asset.get("component"),
        )
        if role == "model":
            _require(slot not in slots, f"Duplicate asset role/format slot: {slot}")
        slots.add(slot)
        if role == "model":
            model_count += 1
            _require(
                bool(asset.get("quality")), f"Model asset {name} is missing quality"
            )
            component = asset.get("component")
            if component is not None:
                _require(
                    isinstance(component, str) and bool(component),
                    f"Model asset {name} has an invalid component",
                )
                model_components.add(component)
        elif role == "voices":
            voice_count += 1
        _validate_asset_format(asset_path, asset, manifest)
    runtime = manifest["runtime"]
    if runtime.get("layout") == "split-onnx-v1":
        expected_components = set(
            (manifest["onnx_contract"].get("components") or {}).keys()
        )
        _require(expected_components, "Split ONNX contract must declare components")
        _require(
            expected_components == model_components,
            "Split ONNX components do not match contract: "
            f"expected {sorted(expected_components)}, got {sorted(model_components)}",
        )

    _require(model_count > 0, "Candidate must contain a model asset")
    _require(voice_count > 0, "Candidate must contain a voices asset")
    _validate_checksums(candidate, assets)
    unexpected = []
    for path in candidate.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(candidate).as_posix()
        if relative in names or (
            path.parent == candidate and path.name in ALLOWED_FILES
        ):
            continue
        unexpected.append(relative)
    _require(
        not unexpected, "Unexpected candidate files: " + ", ".join(sorted(unexpected))
    )
    return {
        "tag": manifest["tag"],
        "profile": manifest["profile"],
        "asset_count": len(assets),
        "manifest": manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a Kokoro release candidate")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-tag")
    parser.add_argument("--expected-profile")
    parser.add_argument("--expected-builder-commit")
    parser.add_argument("--allow-restricted", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify_candidate(
            args.candidate,
            expected_tag=args.expected_tag,
            expected_profile=args.expected_profile,
            expected_builder_commit=args.expected_builder_commit,
            allow_restricted=args.allow_restricted,
        )
    except CandidateError as exc:
        parser.error(str(exc))
    print(
        f"Verified candidate {result['profile']} ({result['tag']}) with {result['asset_count']} assets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
