from __future__ import annotations

import json
import math
import platform
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.build_kokoro import (
        BuildError,
        _validate_onnx_case,
        capture_native_reference_cases,
        hf_download,
        install_exact_onnx_istft,
        load_checkpoint_native,
        resolve_fixed_pt_voices,
        sha256,
        validate_native_reference_cases,
        validate_patched_pytorch_against_native,
        validate_random_source_graph,
        verify_source_hash,
        write_numpy_voice_archive,
    )
except ModuleNotFoundError:
    from build_kokoro import (  # type: ignore[no-redef]
        BuildError,
        _validate_onnx_case,
        capture_native_reference_cases,
        hf_download,
        install_exact_onnx_istft,
        load_checkpoint_native,
        resolve_fixed_pt_voices,
        sha256,
        validate_native_reference_cases,
        validate_patched_pytorch_against_native,
        validate_random_source_graph,
        verify_source_hash,
        write_numpy_voice_archive,
    )

MODEL_ID = "akinvox-multispeaker-v1"
MODEL_VERSION = "2026.10.07"
SOURCE_REPO = "AKinvox/kokoro-multispeaker"
SOURCE_REVISION = "2b3fa45b39256382872823562da36d371a808e5a"
BASE_REPO = "hexgrad/Kokoro-82M"
BASE_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
CATALOG_FORMAT = "akinvox-multispeaker-catalog/v1"
LANGUAGE_BUNDLE_FORMAT = "akinvox-stock-language-bundle/v1"
ADAPTER_FORMAT = "akinvox-stock-multispeaker-adapter/v1"
LANGUAGES = ("en", "de")
BASE_COMPONENTS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")
ADAPTER_ROOT_KEYS = {
    "contract",
    "task",
    "base_sha256",
    "update",
    "targets",
    "state",
    "source_checkpoint_sha256",
    "speaker_names",
    "voicebank",
    "voice_conditioning",
}
TARGET_KEYS = {"root", "module", "parameter", "rank", "alpha", "prefix", "source_index"}
EXPECTED_TARGET_COUNT = 201
EXPECTED_ADAPTER_TENSOR_COUNT = 403
EXPECTED_DEFAULT_INITIALIZED_COUNTS = {
    "bert": 0,
    "bert_encoder": 0,
    "predictor": 24,
    "decoder": 116,
    "text_encoder": 0,
}
DEFAULT_INITIALIZED_SUFFIXES = (
    ".norm1.norm.bias",
    ".norm1.norm.weight",
    ".norm2.norm.bias",
    ".norm2.norm.weight",
)
STYLE_SHAPE = (510, 1, 256)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"Could not read {label} at {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _digest(value: Any, label: str) -> str:
    text = str(value)
    _require(
        len(text) == 64 and all(char in "0123456789abcdef" for char in text),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return text


def _pinned_download(
    repo_id: str,
    revision: str,
    filename: str,
    cache_dir: Path,
    expected_sha256: str,
    label: str,
) -> Path:
    expected = _digest(expected_sha256, f"{label} SHA-256 pin")
    local = hf_download(repo_id, filename, revision, cache_dir)
    verify_source_hash(local, expected, label=label)
    return local


def _parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        _require(len(fields) == 2, f"Malformed SHA256SUMS line {number}")
        digest, filename = fields
        filename = filename.lstrip("* ")
        digest = _digest(digest, f"SHA256SUMS line {number}")
        _require(filename not in entries, f"Duplicate SHA256SUMS entry for {filename}")
        entries[filename] = digest
    return entries


def _download_sources(profile: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    model = profile["model"]
    _require(
        profile.get("repo_id") == SOURCE_REPO, "Unexpected AkinVox source repository"
    )
    _require(
        profile.get("revision") == SOURCE_REVISION,
        "AkinVox revision is not the immutable pinned commit",
    )
    _require(
        model.get("kind") == "akinvox-multispeaker", "Unexpected model builder kind"
    )
    expected_legal_files = {
        "LICENSE": "LICENSE",
        "NOTICE": "NOTICE",
        "THIRD_PARTY.md": "THIRD_PARTY.md",
        "StyleTTS2-MIT.txt": "licenses/StyleTTS2-MIT.txt",
    }
    _require(
        {name: spec["path"] for name, spec in model["legal_files"].items()}
        == expected_legal_files,
        "Pinned legal file set changed",
    )

    catalog_spec = model["source_catalog"]
    sums_spec = model["checksum_manifest"]
    catalog_path = str(catalog_spec["path"])
    sums_path = str(sums_spec["path"])
    catalog_file = _pinned_download(
        SOURCE_REPO,
        SOURCE_REVISION,
        catalog_path,
        cache_dir,
        str(catalog_spec["sha256"]),
        "root source catalog",
    )
    sums_file = _pinned_download(
        SOURCE_REPO,
        SOURCE_REVISION,
        sums_path,
        cache_dir,
        str(sums_spec["sha256"]),
        "source checksum manifest",
    )

    base = model["base"]
    _require(base.get("repo_id") == BASE_REPO, "Unexpected Kokoro base repository")
    _require(
        base.get("revision") == BASE_REVISION, "Kokoro base revision is not pinned"
    )
    base_config = _pinned_download(
        BASE_REPO,
        BASE_REVISION,
        str(base["config_path"]),
        cache_dir,
        str(base["config_sha256"]),
        "Kokoro base config",
    )
    base_checkpoint = _pinned_download(
        BASE_REPO,
        BASE_REVISION,
        str(base["checkpoint_path"]),
        cache_dir,
        str(base["checkpoint_sha256"]),
        "Kokoro base checkpoint",
    )

    language_files: dict[str, dict[str, Path]] = {}
    for language in LANGUAGES:
        spec = model["languages"][language]
        language_files[language] = {
            "config": _pinned_download(
                SOURCE_REPO,
                SOURCE_REVISION,
                str(spec["config_path"]),
                cache_dir,
                str(spec["config_sha256"]),
                f"{language} adapter config",
            ),
            "adapter": _pinned_download(
                SOURCE_REPO,
                SOURCE_REVISION,
                str(spec["adapter_path"]),
                cache_dir,
                str(spec["adapter_sha256"]),
                f"{language} adapter",
            ),
        }

    legal_files: dict[str, Path] = {}
    for output_name, spec in model["legal_files"].items():
        legal_files[output_name] = _pinned_download(
            SOURCE_REPO,
            SOURCE_REVISION,
            str(spec["path"]),
            cache_dir,
            str(spec["sha256"]),
            f"upstream legal file {spec['path']}",
        )

    checksums = _parse_sha256sums(sums_file)
    upstream_expectations: dict[str, str] = {
        catalog_path: str(catalog_spec["sha256"]),
    }
    for language in LANGUAGES:
        spec = model["languages"][language]
        upstream_expectations[str(spec["config_path"])] = str(spec["config_sha256"])
        upstream_expectations[str(spec["adapter_path"])] = str(spec["adapter_sha256"])
    for name, item in profile["voices"]["items"].items():
        upstream_expectations[str(item["path"])] = str(item["sha256"])
    for spec in model["legal_files"].values():
        upstream_expectations[str(spec["path"])] = str(spec["sha256"])
    for filename, expected in upstream_expectations.items():
        _require(
            filename in checksums, f"Pinned source checksum manifest omits {filename}"
        )
        _require(
            checksums[filename] == _digest(expected, f"{filename} SHA-256"),
            f"Source checksum manifest disagrees for {filename}",
        )

    return {
        "catalog": catalog_file,
        "checksum_manifest": sums_file,
        "base_config": base_config,
        "base_checkpoint": base_checkpoint,
        "languages": language_files,
        "legal_files": legal_files,
        "checksum_entries": checksums,
    }


def _expected_architecture(base_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hidden_dim": base_config["hidden_dim"],
        "style_dim": base_config["style_dim"],
        "n_layer": base_config["n_layer"],
        "n_token": base_config["n_token"],
        "max_dur": base_config["max_dur"],
        "dropout": base_config["dropout"],
        "n_mels": base_config["n_mels"],
        "decoder": {"type": "istftnet", **base_config["istftnet"]},
    }


def _validate_source_configs(
    profile: Mapping[str, Any], sources: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    model = profile["model"]
    catalog = _read_json(sources["catalog"], "source catalog")
    base_config = _read_json(sources["base_config"], "base config")
    _require(
        catalog.get("format") == CATALOG_FORMAT, "Unexpected source catalog format"
    )
    _require(
        set(catalog.get("languages", {})) == set(LANGUAGES),
        "Catalog language set changed",
    )
    _require(
        catalog.get("base_sha256") == model["base"]["checkpoint_sha256"],
        "Catalog base checkpoint hash does not match the pinned base",
    )
    _require(
        base_config.get("n_token") == 178,
        "Base tokenizer vocabulary is not Kokoro v1.0",
    )
    vocab = base_config.get("vocab")
    _require(isinstance(vocab, dict), "Base tokenizer vocabulary must be a mapping")
    vocab_ids = list(vocab.values())
    _require(
        all(type(token_id) is int and 0 < token_id < 178 for token_id in vocab_ids),
        "Base tokenizer IDs must fit the 178-entry Kokoro vocabulary",
    )
    _require(len(vocab_ids) == len(set(vocab_ids)), "Base tokenizer IDs must be unique")

    configs: dict[str, dict[str, Any]] = {}
    expected_arch = _expected_architecture(base_config)
    expected_plbert = {"vocab_size": base_config["n_token"], **base_config["plbert"]}
    for language in LANGUAGES:
        spec = model["languages"][language]
        config = _read_json(
            sources["languages"][language]["config"], f"{language} adapter config"
        )
        catalog_entry = catalog["languages"][language]
        _require(
            config.get("format") == LANGUAGE_BUNDLE_FORMAT,
            f"Invalid {language} language format",
        )
        _require(
            config.get("language") == language, f"{language} config language mismatch"
        )
        _require(
            config.get("base_repo") == BASE_REPO,
            f"{language} config base repository mismatch",
        )
        _require(
            config.get("base_revision") == BASE_REVISION,
            f"{language} config base revision mismatch",
        )
        _require(
            config.get("base_sha256") == model["base"]["checkpoint_sha256"],
            f"{language} config base checkpoint hash mismatch",
        )
        _require(
            config.get("architecture") == expected_arch,
            f"{language} architecture differs from stock Kokoro",
        )
        _require(
            config.get("plbert_config") == expected_plbert,
            f"{language} PLBERT config differs from stock Kokoro",
        )
        _require(
            config.get("sample_rate") == 24000,
            f"{language} sample rate must be 24000 Hz",
        )
        _require(
            config.get("release_date") == "2026-10-07",
            f"{language} model release date changed",
        )
        _require(
            config.get("speaker_names") == spec["speaker_names"],
            f"{language} speaker order changed",
        )
        _require(
            catalog_entry.get("path") == str(spec["config_path"]).rsplit("/", 1)[0],
            f"{language} catalog path mismatch",
        )
        _require(
            catalog_entry.get("adapter_sha256") == spec["adapter_sha256"],
            f"{language} catalog adapter hash mismatch",
        )
        _require(
            catalog_entry.get("config_sha256") == spec["config_sha256"],
            f"{language} catalog config hash mismatch",
        )
        _require(
            catalog_entry.get("speaker_names") == spec["speaker_names"],
            f"{language} catalog speaker order mismatch",
        )

        conditioning = config.get("voice_conditioning")
        _require(
            isinstance(conditioning, dict),
            f"{language} voice conditioning must be an object",
        )
        _require(
            conditioning.get("format") == "materialized-fixed-speakers-v1",
            f"{language} voice conditioning format changed",
        )
        _require(
            conditioning.get("all510_rows_identical_per_voice") is True,
            f"{language} row invariant is not declared",
        )
        _require(
            conditioning.get("decoder_and_predictor_halves_finalized") is True,
            f"{language} voice halves are not finalized",
        )
        voicepacks = conditioning.get("voicepacks")
        _require(
            isinstance(voicepacks, dict), f"{language} voicepack mapping is missing"
        )
        _require(
            set(voicepacks) == set(spec["speaker_names"]),
            f"{language} voicepack roster differs from config",
        )
        for name in spec["speaker_names"]:
            voicepack = voicepacks[name]
            _require(
                voicepack.get("shape") == list(STYLE_SHAPE),
                f"{name} has an unexpected declared shape",
            )
            item = profile["voices"]["items"].get(name)
            _require(
                isinstance(item, Mapping),
                f"{name} is missing from the profile voice mapping",
            )
            _require(
                item.get("path") == f"{catalog_entry['path']}/{voicepack['file']}",
                f"{name} source path does not match the language catalog",
            )
            _require(
                item.get("sha256") == voicepack.get("sha256"),
                f"{name} profile/config hash mismatch",
            )
        _require(
            config.get("source_checkpoint_sha256") == spec["source_checkpoint_sha256"],
            f"{language} source checkpoint hash mismatch",
        )
        _require(
            config.get("update") == spec["update"],
            f"{language} adapter update metadata mismatch",
        )
        configs[language] = config

    ordered_names = [
        name
        for language in LANGUAGES
        for name in model["languages"][language]["speaker_names"]
    ]
    _require(
        list(profile["voices"]["items"]) == ordered_names,
        "Profile voice order is not the published order",
    )
    return catalog, configs, base_config


def _tensor_inventory(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    import torch

    records: list[dict[str, Any]] = []

    def visit(value: Any, name: str) -> None:
        if isinstance(value, torch.Tensor):
            records.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
            )
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                child = f"{name}.{key}" if name else str(key)
                visit(value[key], child)
        elif isinstance(value, (list, tuple)):
            for index, child_value in enumerate(value):
                visit(child_value, f"{name}[{index}]")

    visit(payload, "")
    return sorted(records, key=lambda record: record["name"])


def _json_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_metadata(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise BuildError(
        f"Adapter metadata contains unsupported value {type(value).__name__}"
    )


def _resolve_target(model: Any, target: Mapping[str, Any]) -> tuple[Any, Any]:
    root_name = str(target["root"])
    _require(
        root_name in BASE_COMPONENTS,
        f"Adapter target uses unknown base component {root_name!r}",
    )
    root = getattr(model, root_name, None)
    _require(root is not None, f"Base model has no component {root_name!r}")
    module_path = str(target["module"])
    try:
        module = root.get_submodule(module_path) if module_path else root
    except AttributeError as exc:
        raise BuildError(
            f"Adapter module {root_name}.{module_path} is absent from the base model"
        ) from exc
    parameter_name = str(target["parameter"])
    parametrizations = getattr(module, "parametrizations", None)
    if parameter_name not in getattr(module, "_parameters", {}) and not (
        parametrizations is not None and parameter_name in parametrizations
    ):
        raise BuildError(
            f"Adapter target {target['prefix']} does not name a model parameter"
        )
    weight = getattr(module, parameter_name)
    _require(
        hasattr(weight, "shape"), f"Base target {target['prefix']} is not tensor-valued"
    )
    return module, weight


def _audit_and_apply_adapter(
    language: str,
    adapter_path: Path,
    config: Mapping[str, Any],
    source_spec: Mapping[str, Any],
    model: Any,
    voices: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    import torch
    from torch.nn.utils import parametrize

    payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping), f"{language} adapter root must be a mapping")
    _require(set(payload) == ADAPTER_ROOT_KEYS, f"{language} adapter root keys changed")
    _require(
        payload.get("contract") == ADAPTER_FORMAT,
        f"{language} adapter contract changed",
    )
    _require(
        payload.get("task") == source_spec["task"],
        f"{language} adapter task metadata mismatch",
    )
    _require(
        payload.get("base_sha256") == config["base_sha256"],
        f"{language} adapter base hash mismatch",
    )
    _require(
        payload.get("source_checkpoint_sha256") == config["source_checkpoint_sha256"],
        f"{language} source checkpoint metadata mismatch",
    )
    _require(
        payload.get("update") == config["update"],
        f"{language} adapter update metadata mismatch",
    )
    _require(
        payload.get("speaker_names") == config["speaker_names"],
        f"{language} adapter speaker roster mismatch",
    )
    _require(
        payload.get("voice_conditioning") == config["voice_conditioning"],
        f"{language} adapter voice metadata mismatch",
    )

    state = payload.get("state")
    targets = payload.get("targets")
    _require(isinstance(state, Mapping), f"{language} adapter state must be a mapping")
    _require(
        isinstance(targets, list) and targets,
        f"{language} adapter targets must be a non-empty list",
    )
    base_audit = getattr(model, "_checkpoint_load_audit", {})
    component_audits = base_audit.get("components", {})
    _require(
        set(component_audits) == set(BASE_COMPONENTS),
        "Base checkpoint components were not all audited",
    )
    _require(
        base_audit.get("strict") is True,
        "Base checkpoint did not pass exact-load audit",
    )

    for component_name in BASE_COMPONENTS:
        audit = component_audits[component_name]
        for audit_field in (
            "missing_keys",
            "unexpected_keys",
            "shape_mismatches",
            "loaded_tensor_mismatches",
        ):
            _require(
                audit.get(audit_field) == [],
                f"Base component {component_name} has {audit_field}",
            )
        default_keys = audit.get("default_initialized_keys")
        _require(
            isinstance(default_keys, list)
            and len(default_keys)
            == EXPECTED_DEFAULT_INITIALIZED_COUNTS[component_name],
            f"Base component {component_name} default-initialized key inventory changed",
        )
        _require(
            all(
                isinstance(key, str)
                and (
                    key.endswith(DEFAULT_INITIALIZED_SUFFIXES)
                    or (
                        (".adain1." in key or ".adain2." in key)
                        and key.endswith((".norm.bias", ".norm.weight"))
                    )
                )
                for key in default_keys
            ),
            f"Base component {component_name} has unexpected default-initialized keys",
        )
    target_records: list[dict[str, Any]] = []
    expected_factor_names: set[str] = set()
    seen_prefixes: set[str] = set()
    seen_parameters: set[tuple[str, str, str]] = set()
    for target in targets:
        _require(
            isinstance(target, Mapping), f"{language} adapter target must be an object"
        )
        _require(
            set(target) == TARGET_KEYS, f"{language} adapter target fields changed"
        )
        prefix = str(target["prefix"])
        root_name = str(target["root"])
        module_path = str(target["module"])
        parameter_name = str(target["parameter"])
        _require(
            prefix
            == ".".join(
                part for part in (root_name, module_path, parameter_name) if part
            ),
            f"Adapter prefix mismatch for {prefix}",
        )
        _require(
            prefix not in seen_prefixes, f"Duplicate adapter target prefix {prefix}"
        )
        seen_prefixes.add(prefix)
        identity = (root_name, module_path, parameter_name)
        _require(
            identity not in seen_parameters,
            f"Duplicate adapter model target {identity}",
        )
        seen_parameters.add(identity)

        rank = target["rank"]
        alpha = target["alpha"]
        source_index = target["source_index"]
        _require(type(rank) is int and rank > 0, f"Invalid adapter rank for {prefix}")
        _require(
            isinstance(alpha, (int, float)) and not isinstance(alpha, bool),
            f"Invalid adapter alpha for {prefix}",
        )
        _require(math.isfinite(float(alpha)), f"Non-finite adapter alpha for {prefix}")
        _require(
            type(source_index) is int and source_index in (0, 1),
            f"Invalid source_index for {prefix}",
        )

        a_name = f"{prefix}.A"
        b_name = f"{prefix}.B"
        _require(
            a_name in state and b_name in state,
            f"Adapter factors are missing for {prefix}",
        )
        factor_a = state[a_name]
        factor_b = state[b_name]
        _require(
            isinstance(factor_a, torch.Tensor) and isinstance(factor_b, torch.Tensor),
            f"Adapter factors for {prefix} must be tensors",
        )
        _require(
            factor_a.dtype == torch.float32 and factor_b.dtype == torch.float32,
            f"Adapter factors for {prefix} must be float32",
        )
        _require(
            factor_a.ndim == 2 and factor_b.ndim == 2,
            f"Adapter factors for {prefix} must be matrices",
        )
        _require(
            factor_a.shape[0] == rank and factor_b.shape[1] == rank,
            f"Adapter rank metadata disagrees for {prefix}",
        )
        _require(
            bool(torch.isfinite(factor_a).all() and torch.isfinite(factor_b).all()),
            f"Adapter factors for {prefix} contain non-finite values",
        )

        module, weight = _resolve_target(model, target)
        _require(weight.dtype == torch.float32, f"Base target {prefix} is not float32")
        _require(
            weight.ndim >= 2 and weight.shape[0] == factor_b.shape[0],
            f"Adapter output dimension mismatch for {prefix}",
        )
        _require(
            factor_a.shape[1] * factor_b.shape[0] == weight.numel(),
            f"Adapter factors cannot materialize base shape for {prefix}",
        )
        base_key = ".".join(
            part for part in (root_name, module_path, parameter_name) if part
        )
        target_records.append(
            {
                **_json_metadata(target),
                "base_component": root_name,
                "base_model_key": base_key,
                "base_weight_shape": list(weight.shape),
                "base_weight_dtype": str(weight.dtype).removeprefix("torch."),
                "base_component_load_audit": _json_metadata(
                    component_audits[root_name]
                ),
                "factor_a": {
                    "name": f"state.{a_name}",
                    "shape": list(factor_a.shape),
                    "dtype": str(factor_a.dtype).removeprefix("torch."),
                },
                "factor_b": {
                    "name": f"state.{b_name}",
                    "shape": list(factor_b.shape),
                    "dtype": str(factor_b.dtype).removeprefix("torch."),
                },
            }
        )
        expected_factor_names.update((a_name, b_name))

    _require(
        len(targets) == len(seen_prefixes),
        f"{language} adapter target inventory is incomplete",
    )
    _require(
        set(state) == expected_factor_names,
        f"{language} adapter has missing or unconsumed factor tensors",
    )
    _require(
        len(state) == 2 * len(targets),
        f"{language} adapter factor count is inconsistent",
    )

    voicebank = payload.get("voicebank")
    _require(
        isinstance(voicebank, torch.Tensor),
        f"{language} adapter voicebank must be a tensor",
    )
    names = list(config["speaker_names"])
    expected_voicebank = np.stack([np.asarray(voices[name])[:, 0, :] for name in names])
    _require(
        voicebank.dtype == torch.float32,
        f"{language} adapter voicebank must be float32",
    )
    _require(
        tuple(voicebank.shape) == expected_voicebank.shape,
        f"{language} adapter voicebank shape mismatch",
    )
    _require(
        np.array_equal(voicebank.cpu().numpy(), expected_voicebank),
        f"{language} adapter voicebank differs from published voice files",
    )

    all_tensors = _tensor_inventory(payload)
    tensor_names = {record["name"] for record in all_tensors}
    expected_names = {f"state.{name}" for name in expected_factor_names} | {"voicebank"}
    _require(
        tensor_names == expected_names,
        f"{language} adapter contains an unaccounted tensor",
    )
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/akinvox_multispeaker_adapter_inventory.json"
    )
    fixture = _read_json(fixture_path, "adapter structural inventory fixture")
    _require(
        fixture.get("format") == "akinvox-adapter-structural-inventory/v1",
        "Adapter inventory fixture format changed",
    )
    fixture_language = fixture.get("languages", {}).get(language)
    actual_metadata = {
        "base_sha256": payload["base_sha256"],
        "contract": payload["contract"],
        "root_keys": sorted(str(key) for key in payload),
        "root_type": type(payload).__name__,
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "speaker_names": payload["speaker_names"],
        "state_tensor_count": len(state),
        "target_count": len(targets),
        "task": payload["task"],
        "update": payload["update"],
    }
    _require(
        actual_metadata == fixture_language,
        f"{language} adapter structure differs from the audited fixture",
    )
    actual_inventory = [
        [record["name"], record["shape"], record["dtype"]] for record in all_tensors
    ]
    _require(
        len(fixture.get("tensors", [])) == fixture.get("tensor_count"),
        "Adapter tensor fixture count is inconsistent",
    )
    _require(
        actual_inventory == fixture["tensors"],
        f"{language} adapter tensor inventory differs from the audited fixture",
    )
    _require(len(targets) == EXPECTED_TARGET_COUNT, f"{language} target count changed")
    _require(
        len(all_tensors) == EXPECTED_ADAPTER_TENSOR_COUNT,
        f"{language} adapter tensor count changed",
    )
    target_records.sort(key=lambda record: record["prefix"])

    materialized = 0
    removed_parametrizations = 0
    for target in sorted(targets, key=lambda item: str(item["prefix"])):
        module, weight = _resolve_target(model, target)
        parameter_name = str(target["parameter"])
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None and parameter_name in parametrizations:
            parametrize.remove_parametrizations(
                module, parameter_name, leave_parametrized=True
            )
            removed_parametrizations += 1
            weight = getattr(module, parameter_name)
        prefix = str(target["prefix"])
        factor_a = state[f"{prefix}.A"]
        factor_b = state[f"{prefix}.B"]
        update = torch.matmul(factor_b, factor_a).reshape(weight.shape)
        scale = float(target["alpha"]) / int(target["rank"])
        with torch.no_grad():
            weight.add_(update.to(dtype=weight.dtype) * scale)
        _require(
            not (
                getattr(module, "parametrizations", None)
                and parameter_name in module.parametrizations
            ),
            f"Adapter parametrization survived materialization for {prefix}",
        )
        materialized += 1
    _require(
        materialized == len(targets),
        f"{language} adapter materialization count mismatch",
    )
    _require(
        2 * materialized + 1 == len(all_tensors),
        f"{language} has unconsumed adapter tensors",
    )

    model.to("cpu").eval()
    model.requires_grad_(False)
    _require(
        all(not parameter.requires_grad for parameter in model.parameters()),
        f"{language} model parameters are not frozen",
    )
    _require(
        not any(
            "adapter" in type(module).__name__.lower() for module in model.modules()
        ),
        f"{language} runtime model contains an adapter module",
    )

    return {
        "language": language,
        "source_adapter_sha256": sha256(adapter_path),
        "root_object_type": type(payload).__name__,
        "root_keys": sorted(str(key) for key in payload),
        "key_hierarchy": {
            "state": sorted(str(key) for key in state),
            "target_fields": sorted(TARGET_KEYS),
            "voicebank": {
                "shape": list(voicebank.shape),
                "dtype": str(voicebank.dtype).removeprefix("torch."),
            },
        },
        "scalar_metadata": {
            key: _json_metadata(payload[key])
            for key in (
                "contract",
                "task",
                "base_sha256",
                "update",
                "source_checkpoint_sha256",
                "speaker_names",
                "voice_conditioning",
            )
        },
        "target_count": len(targets),
        "tensor_count": len(all_tensors),
        "tensor_inventory": all_tensors,
        "target_mappings": target_records,
        "tensor_accounting": {
            "state_tensors_total": len(state),
            "state_tensors_consumed": 2 * materialized,
            "voicebank_tensor_consumed": True,
            "total_tensors": len(all_tensors),
            "consumed_tensors": 2 * materialized + 1,
            "unconsumed_tensor_names": [],
            "all_adapter_tensors_consumed": len(all_tensors) == 2 * materialized + 1,
        },
        "materialization": {
            "operation": "weight += reshape(B @ A, weight.shape) * alpha / rank",
            "scale_source": "adapter target alpha and rank metadata",
            "targets_materialized": materialized,
            "parametrizations_materialized": removed_parametrizations,
            "runtime_adapter_modules": 0,
            "model_device": "cpu",
            "model_mode": "eval",
            "parameters_frozen": True,
        },
    }


def _language_cases(
    language: str, cases: list[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    prefixes = {"en": "english-", "de": "german-"}
    _require(language in prefixes, f"Unsupported parity language {language!r}")
    selected = [
        case
        for case in cases
        if str(case.get("name", "")).startswith(prefixes[language])
    ]
    _require(bool(selected), f"{language} has no language-specific parity fixtures")
    return selected


def _selected_parity_pairs(
    language: str,
    cases: list[Mapping[str, Any]],
    voice_names: list[str],
) -> list[tuple[Mapping[str, Any], str]]:
    _require(len(voice_names) >= 2, f"{language} needs at least two voices for parity")
    standard_speed = [case for case in cases if float(case.get("speed", 1.0)) == 1.0]
    _require(len(standard_speed) >= 2, f"{language} needs two speed-1.0 token fixtures")
    pairs = [(case, voice) for case in standard_speed[:2] for voice in voice_names[:2]]
    nondefault = next(
        (case for case in cases if float(case.get("speed", 1.0)) != 1.0), None
    )
    _require(
        nondefault is not None, f"{language} needs one non-default speed parity case"
    )
    pairs.append((nondefault, voice_names[0]))
    return pairs


def _export_language_model(
    language: str,
    model: Any,
    profile: Mapping[str, Any],
    voices: Mapping[str, np.ndarray],
    out_path: Path,
    *,
    opset: int,
) -> dict[str, Any]:
    import kokoro
    import onnx
    import onnxruntime as ort
    import torch
    from kokoro.model import KModelForONNX

    validation = profile["export_validation"]
    cases = _language_cases(language, list(validation.get("cases", [])))
    voice_names = list(profile["model"]["languages"][language]["speaker_names"])
    pairs = _selected_parity_pairs(language, cases, voice_names)
    seed = int(validation["export_seed"])
    native_wrapper = KModelForONNX(model).eval()
    native_cases: list[dict[str, Any]] = []
    for case, voice_name in pairs:
        native_cases.extend(
            capture_native_reference_cases(
                native_wrapper,
                [case],
                voices[voice_name],
                max_phonemes=510,
                seed=seed,
            )
        )
    native_reference_validation = validate_native_reference_cases(
        native_cases,
        sample_rate=int(profile["sample_rate"]),
        validation=validation,
    )

    istft_metadata, export_stft = install_exact_onnx_istft(model)
    export_model = KModelForONNX(model).eval()
    _require(
        model.decoder.generator.stft is export_stft
        and export_stft.onnx_transform_enabled is False,
        "Exact ONNX iSTFT installation did not preserve the native transform",
    )
    delegate_validation = validate_patched_pytorch_against_native(
        export_model,
        native_cases,
        sample_rate=int(profile["sample_rate"]),
        validation=validation,
        seed=seed,
    )
    _require(
        model.decoder.generator.stft is export_stft,
        "Export iSTFT replacement was lost before enabling ONNX mode",
    )
    export_stft.use_onnx_transform()
    _require(
        export_stft.onnx_transform_enabled,
        "Exact ONNX iSTFT did not enter ONNX-safe mode",
    )
    patched_validation = validate_patched_pytorch_against_native(
        export_model,
        native_cases,
        sample_rate=int(profile["sample_rate"]),
        validation=validation,
        seed=seed,
    )

    first = native_cases[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (first["tokens"], first["style"], first["speed"]),
            str(out_path),
            input_names=["tokens", "style", "speed"],
            output_names=["audio", "duration"],
            dynamic_axes={
                "tokens": {1: "sequence_length"},
                "audio": {0: "audio_samples"},
                "duration": {0: "duration_tokens"},
            },
            opset_version=opset,
            dynamo=False,
        )

    graph = onnx.load(str(out_path), load_external_data=True)
    onnx.checker.check_model(graph)
    random_ops = validate_random_source_graph(
        graph,
        requires_random_source_ops=bool(
            validation.get("requires_random_source_ops", True)
        ),
    )
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    parity_cases: list[dict[str, Any]] = []
    for index, ((case, voice_name), native_case, patched_case) in enumerate(
        zip(pairs, native_cases, patched_validation["cases"], strict=True)
    ):
        result = _validate_onnx_case(
            session,
            native_case,
            patched_case,
            sample_rate=int(profile["sample_rate"]),
            validation=validation,
        )
        result.update(
            {
                "fixture": str(case["name"]),
                "voice": voice_name,
                "speed": float(case.get("speed", 1.0)),
                "seed": seed,
                "case_index": index,
            }
        )
        parity_cases.append(result)

    return {
        "language": language,
        "file": out_path.name,
        "sha256": sha256(out_path),
        "exporter": {
            "kokoro_version": str(getattr(kokoro, "__version__", "unknown")),
            "torch_version": str(torch.__version__),
            "onnx_version": str(onnx.__version__),
            "onnxruntime_version": str(ort.__version__),
            "python_version": platform.python_version(),
            "opset": opset,
            "inputs": ["tokens", "style", "speed"],
            "outputs": ["audio", "duration"],
            "random_source_ops": random_ops,
            "checkpoint_load_audit": _json_metadata(model._checkpoint_load_audit),
            "istft": istft_metadata,
        },
        "onnx_checker": "passed",
        "contract_validation": _validate_graph_contract(out_path),
        "native_reference_validation": native_reference_validation,
        "delegate_validation": delegate_validation,
        "patched_pytorch_validation": patched_validation,
        "parity": {"status": "pass", "cases": parity_cases},
    }


def _onnx_dtype(elem_type: int) -> str:
    import onnx

    names = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.INT64: "int64",
    }
    return names.get(elem_type, onnx.TensorProto.DataType.Name(elem_type).lower())


def _validate_graph_contract(path: Path) -> dict[str, Any]:
    import onnx

    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)
    actual_inputs = {
        value.name: _onnx_dtype(value.type.tensor_type.elem_type)
        for value in graph.graph.input
    }
    aliases = {
        "tokens": ("tokens", "input_ids"),
        "style": ("style", "ref_s"),
        "speed": ("speed",),
    }
    normalized: dict[str, dict[str, str]] = {}
    used: set[str] = set()
    for semantic, names in aliases.items():
        matches = [name for name in names if name in actual_inputs]
        _require(
            len(matches) == 1, f"{path.name} must expose one {semantic} input alias"
        )
        actual_name = matches[0]
        expected_type = "int64" if semantic == "tokens" else "float32"
        _require(
            actual_inputs[actual_name] == expected_type,
            f"{path.name} {semantic} dtype mismatch",
        )
        normalized[semantic] = {
            "name": actual_name,
            "dtype": actual_inputs[actual_name],
        }
        used.add(actual_name)
    _require(set(actual_inputs) == used, f"{path.name} exposes unexpected ONNX inputs")

    actual_outputs = {
        value.name: _onnx_dtype(value.type.tensor_type.elem_type)
        for value in graph.graph.output
    }
    _require(
        actual_outputs == {"audio": "float32", "duration": "int64"},
        f"{path.name} output contract mismatch: {actual_outputs}",
    )
    return {
        "onnx_checker": "passed",
        "inputs": normalized,
        "outputs": actual_outputs,
        "timing": {
            "kind": "token-duration-v1",
            "output": "duration",
            "unit": "frame",
            "samples_per_frame": 600,
            "includes_boundary_tokens": True,
        },
        "max_tokens": 510,
    }


def _validate_voice_archive(
    voices: Mapping[str, np.ndarray], profile: Mapping[str, Any], out_dir: Path
) -> dict[str, Any]:
    names = list(profile["voices"]["items"])
    _require(
        list(voices) == names, "Resolved voice order differs from the published roster"
    )
    voice_hashes: dict[str, str] = {}
    for name, voice in voices.items():
        _require(
            tuple(voice.shape) == STYLE_SHAPE, f"{name} shape must be {STYLE_SHAPE}"
        )
        _require(voice.dtype == np.dtype("float32"), f"{name} must be float32")
        _require(voice.flags.c_contiguous, f"{name} must be contiguous")
        _require(bool(np.isfinite(voice).all()), f"{name} contains non-finite values")
        first_row = voice[0].tobytes(order="C")
        _require(
            all(
                voice[index].tobytes(order="C") == first_row for index in range(1, 510)
            ),
            f"{name} rows are not byte-identical",
        )
        voice_hashes[name] = str(profile["voices"]["items"][name]["sha256"])

    write_numpy_voice_archive(voices, out_dir / "voices.npz")
    with np.load(out_dir / "voices.npz", allow_pickle=False) as archive:
        _require(archive.files == names, "Voice NPZ roster or order changed")
        for name in names:
            _require(
                tuple(archive[name].shape) == STYLE_SHAPE,
                f"Archived {name} shape changed",
            )
            _require(
                archive[name].dtype == np.dtype("float32"),
                f"Archived {name} dtype changed",
            )
            _require(
                np.array_equal(archive[name], voices[name]),
                f"Archived {name} values changed",
            )
    return {
        "file": "voices.npz",
        "sha256": sha256(out_dir / "voices.npz"),
        "format": "numpy-npz",
        "names": names,
        "shape_per_voice": list(STYLE_SHAPE),
        "dtype": "float32",
        "all_rows_identical_per_voice": True,
        "voice_sha256": voice_hashes,
    }


def _runtime_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    runtime = profile["release"]["runtime"]
    expected_mapping = {"en": "model_en", "de": "model_de"}
    voice_names = list(profile["voices"]["items"])
    expected_languages = {
        name: language
        for language in LANGUAGES
        for name in profile["model"]["languages"][language]["speaker_names"]
    }
    expected_labels = {"en": "English", "de": "German"}
    _require(
        runtime.get("layout") == "language-router-onnx-v1", "Unsupported runtime layout"
    )
    _require(
        runtime.get("voice_mode") == "static", "AkinVox target requires static voices"
    )
    _require(
        runtime.get("language_models") == expected_mapping,
        "Language model mapping changed",
    )
    _require(
        profile.get("language_codes") == list(LANGUAGES), "Language code order changed"
    )
    _require(
        runtime.get("language_codes") == list(LANGUAGES),
        "Runtime language codes changed",
    )
    _require(runtime.get("max_tokens") == 510, "AkinVox max_tokens must be 510")
    _require(
        runtime.get("default_voice") == "AV_ariala", "AkinVox default voice changed"
    )
    _require(
        profile["release"].get("default_voice") == "AV_ariala",
        "Release default voice changed",
    )
    _require(len(voice_names) == 16, "AkinVox profile must contain exactly 16 voices")
    _require(runtime.get("voices") == voice_names, "Runtime voice order changed")
    metadata = runtime.get("voice_metadata")
    _require(isinstance(metadata, Mapping), "Runtime voice metadata is missing")
    _require(
        list(metadata) == voice_names, "Runtime voice metadata roster or order changed"
    )
    for name in voice_names:
        language = expected_languages[name]
        _require(
            metadata[name]
            == {
                "gender": "unknown",
                "language": language,
                "locale": language,
                "language_label": expected_labels[language],
            },
            f"{name} runtime metadata is not conservative or language-correct",
        )
    return _json_metadata(runtime)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_akinvox_multispeaker_profile(
    profile_key: str,
    profile: dict[str, Any],
    out_root: Path,
    *,
    opset: int,
    cache_dir: Path,
    run_checker: bool,
) -> Path:
    _require(profile_key == MODEL_ID, f"Unexpected profile key {profile_key!r}")
    _require(
        profile.get("model_version") == MODEL_VERSION, "AkinVox model version changed"
    )
    _require(
        profile.get("sample_rate") == 24000, "AkinVox sample rate must be 24000 Hz"
    )
    _require(
        profile.get("language_codes") == list(LANGUAGES),
        "AkinVox language list changed",
    )
    runtime = _runtime_contract(profile)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / profile_key
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = _download_sources(profile, cache_dir)
    catalog, configs, base_config = _validate_source_configs(profile, sources)
    voices = resolve_fixed_pt_voices(
        SOURCE_REPO,
        SOURCE_REVISION,
        cache_dir,
        profile["voices"]["items"],
    )
    voice_archive = _validate_voice_archive(voices, profile, out_dir)
    source_hashes: dict[str, Any] = {
        "repository": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "source_catalog": {
            "path": str(profile["model"]["source_catalog"]["path"]),
            "sha256": sha256(sources["catalog"]),
        },
        "checksum_manifest": {
            "path": str(profile["model"]["checksum_manifest"]["path"]),
            "sha256": sha256(sources["checksum_manifest"]),
        },
        "base": {
            "repository": BASE_REPO,
            "revision": BASE_REVISION,
            "config_sha256": sha256(sources["base_config"]),
            "checkpoint_sha256": sha256(sources["base_checkpoint"]),
        },
        "languages": {},
        "legal_files": {},
    }
    component_provenance: dict[str, Any] = {}
    adapter_summaries: dict[str, Any] = {}
    for language in LANGUAGES:
        spec = profile["model"]["languages"][language]
        language_sources = sources["languages"][language]
        _require(
            sha256(language_sources["config"]) == spec["config_sha256"],
            f"{language} config hash changed",
        )
        _require(
            sha256(language_sources["adapter"]) == spec["adapter_sha256"],
            f"{language} adapter hash changed",
        )
        source_hashes["languages"][language] = {
            "config_sha256": sha256(language_sources["config"]),
            "adapter_sha256": sha256(language_sources["adapter"]),
            "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
            "voice_sha256": {
                name: str(profile["voices"]["items"][name]["sha256"])
                for name in spec["speaker_names"]
            },
        }

        model = load_checkpoint_native(sources["base_checkpoint"], base_config)
        adapter_audit = _audit_and_apply_adapter(
            language,
            language_sources["adapter"],
            configs[language],
            spec,
            model,
            voices,
        )
        audit_path = out_dir / f"adapter-audit-{language}.json"
        _write_json(audit_path, adapter_audit)
        adapter_summaries[language] = {
            "path": audit_path.name,
            "sha256": sha256(audit_path),
            "source_adapter_sha256": adapter_audit["source_adapter_sha256"],
            "target_count": adapter_audit["target_count"],
            "tensor_count": adapter_audit["tensor_count"],
            "consumed_tensors": adapter_audit["tensor_accounting"]["consumed_tensors"],
            "all_adapter_tensors_consumed": adapter_audit["tensor_accounting"][
                "all_adapter_tensors_consumed"
            ],
            "materialization": adapter_audit["materialization"],
        }

        component = f"model_{language}"
        exporter = _export_language_model(
            language,
            model,
            profile,
            voices,
            out_dir / f"{component}.onnx",
            opset=opset,
        )
        component_provenance[component] = {
            **exporter,
            "adapter_load_audit": adapter_summaries[language],
        }
        del model

    for output_name, source_path in sources["legal_files"].items():
        shutil.copyfile(source_path, out_dir / output_name)
        source_hashes["legal_files"][output_name] = sha256(source_path)

    shutil.copyfile(sources["catalog"], out_dir / "source-catalog.json")
    onnx_contract = profile["onnx_contract"]
    for language in LANGUAGES:
        component = f"model_{language}"
        actual_contract = component_provenance[component]["contract_validation"]
        _require(
            actual_contract["outputs"] == onnx_contract["outputs"],
            f"{component} does not match the declared output contract",
        )
        _require(
            actual_contract["max_tokens"] == onnx_contract["max_tokens"],
            f"{component} max_tokens mismatch",
        )
        expected_inputs = onnx_contract["inputs"]
        input_aliases = {
            "tokens": ("tokens", "input_ids"),
            "style": ("style", "ref_s"),
            "speed": ("speed",),
        }
        declared_names: set[str] = set()
        for semantic, aliases in input_aliases.items():
            matches = [name for name in aliases if name in expected_inputs]
            _require(
                len(matches) == 1,
                f"Declared ONNX contract must specify one {semantic} alias",
            )
            declared_name = matches[0]
            declared_names.add(declared_name)
            expected_type = "int64" if semantic == "tokens" else "float32"
            _require(
                expected_inputs[declared_name] == expected_type,
                f"Declared {semantic} dtype mismatch",
            )
            _require(
                actual_contract["inputs"][semantic]["dtype"] == expected_type,
                f"{component} {semantic} dtype differs from the declared contract",
            )
        _require(
            set(expected_inputs) == declared_names,
            "Declared ONNX contract has unexpected inputs",
        )
        _require(
            actual_contract["timing"] == onnx_contract["timing"],
            f"{component} timing contract mismatch",
        )
    _require(
        set(onnx_contract["components"]) == {"model_en", "model_de"},
        "ONNX component contract must contain both language models",
    )
    for component_name in ("model_en", "model_de"):
        component_contract = onnx_contract["components"][component_name]
        _require(
            component_contract["inputs"] == onnx_contract["inputs"],
            f"{component_name} input contract differs",
        )
        _require(
            component_contract["outputs"] == onnx_contract["outputs"],
            f"{component_name} output contract differs",
        )

    runtime_config = {
        "format": "akinvox-multispeaker-runtime-config/v1",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "sample_rate": 24000,
        "language_codes": list(LANGUAGES),
        "voices": list(profile["voices"]["items"]),
        "default_voice": "AV_ariala",
        "runtime": runtime,
        "onnx_contract": _json_metadata(onnx_contract),
    }
    _write_json(out_dir / "config.json", runtime_config)

    files = {
        path.name: sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != "bundle.json"
    }
    forbidden_suffixes = {".pt", ".pth", ".py", ".whl"}
    forbidden = [
        path.name
        for path in out_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    _require(
        not forbidden, f"Runtime bundle contains forbidden build inputs: {forbidden}"
    )
    _require("adapter.pt" not in files, "Runtime bundle contains an adapter checkpoint")

    bundle = {
        "profile": profile_key,
        "model_version": MODEL_VERSION,
        "release_version": 1,
        "source": source_hashes,
        "runtime": runtime,
        "components": component_provenance,
        "onnx_contract": _json_metadata(onnx_contract),
        "voice_archive": voice_archive,
        "adapter_load_audit": adapter_summaries,
        "legal_assets": {
            output_name: {"path": output_name, "sha256": digest}
            for output_name, digest in source_hashes["legal_files"].items()
        },
        "files": files,
        "onnx_checker": {
            "requested_by_cli": bool(run_checker),
            "mandatory_for_target": True,
            "status": "passed",
        },
    }
    _write_json(out_dir / "bundle.json", bundle)
    shutil.rmtree(cache_dir, ignore_errors=True)
    return out_dir
