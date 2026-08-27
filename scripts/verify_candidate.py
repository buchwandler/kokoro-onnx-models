#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

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

    assets = manifest["assets"]
    _require(
        isinstance(assets, list) and assets, "Manifest assets must be a non-empty list"
    )
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
