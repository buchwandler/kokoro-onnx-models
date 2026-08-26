#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub",
#   "numpy",
#   "onnx",
#   "torch",
#   "kokoro>=0.9.4",
# ]
# ///
"""Build Kokoro-family ONNX + sherpa-onnx raw voice bundles from HF profiles.

This intentionally separates *acoustic-model compatibility* from *text frontend
compatibility*. A model can be converted successfully while still requiring its
training-time G2P (for example vig2p or phonikud) before token IDs are passed to
ONNX.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

PROFILE_FILE = Path(__file__).with_name("kokoro_profiles.json")
STYLE_WIDTH = 256


class BuildError(RuntimeError):
    pass


def load_profiles(path: Path = PROFILE_FILE) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise BuildError(f"Expected an object in {path}")
    return data


def hf_download(repo_id: str, filename: str, revision: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(cache_dir),
        )
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_voice(value: Any, *, name: str) -> np.ndarray:
    """Normalize a Kokoro style table to [rows, 1, 256] float32."""
    arr = _as_numpy(value)
    if arr.ndim == 2 and arr.shape[1] == STYLE_WIDTH:
        arr = arr[:, None, :]
    if arr.ndim != 3 or arr.shape[1:] != (1, STYLE_WIDTH):
        raise BuildError(
            f"Voice {name!r} has shape {tuple(arr.shape)}; expected "
            f"[rows, {STYLE_WIDTH}] or [rows, 1, {STYLE_WIDTH}]"
        )
    return np.ascontiguousarray(arr, dtype="<f4")


def load_pt_voice(path: Path, *, name: str) -> np.ndarray:
    import torch

    value = torch.load(path, weights_only=True, map_location="cpu")
    return normalize_voice(value, name=name)


def _validate_same_shape(voices: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    if not voices:
        raise BuildError("No voices were resolved")
    shapes = {name: tuple(arr.shape) for name, arr in voices.items()}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise BuildError(f"All speakers must share one style shape; got {shapes}")
    shape = next(iter(unique))
    return int(shape[0]), int(shape[1]), int(shape[2])


def resolve_json_pt_voices(
    repo_id: str, revision: str, cache_dir: Path, index_path: str
) -> dict[str, np.ndarray]:
    local_index = hf_download(repo_id, index_path, revision, cache_dir)
    with local_index.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        raise BuildError(f"Expected an object in {index_path}")

    voices: dict[str, np.ndarray] = {}
    # Preserve publisher order so sid mapping is deterministic and inspectable.
    for name, item in index.items():
        if not isinstance(item, dict) or not item.get("filename"):
            raise BuildError(f"Invalid voice entry {name!r} in {index_path}")
        local = hf_download(repo_id, str(item["filename"]), revision, cache_dir)
        voices[name] = load_pt_voice(local, name=name)
    return voices


def resolve_fixed_pt_voices(
    repo_id: str, revision: str, cache_dir: Path, items: Mapping[str, str]
) -> dict[str, np.ndarray]:
    voices: dict[str, np.ndarray] = {}
    for name, filename in items.items():
        local = hf_download(repo_id, filename, revision, cache_dir)
        voices[name] = load_pt_voice(local, name=name)
    return voices


def _read_raw_voice_file(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    if not names:
        raise BuildError("Raw voice source requires speaker names")
    raw = np.fromfile(path, dtype="<f4")
    per_row = len(names) * STYLE_WIDTH
    if raw.size == 0 or raw.size % per_row:
        raise BuildError(
            f"{path} contains {raw.size} float32 values, which cannot be split "
            f"into {len(names)} speaker(s) with {STYLE_WIDTH} columns"
        )
    rows = raw.size // per_row
    packed = raw.reshape(len(names), rows, 1, STYLE_WIDTH)
    return {name: np.ascontiguousarray(packed[i]) for i, name in enumerate(names)}


def _read_npz_voice_archive(
    path: Path, expected_names: list[str]
) -> dict[str, np.ndarray]:
    voices: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        for name in expected_names:
            if name in archive:
                voices[name] = normalize_voice(archive[name], name=name)
        if not voices and len(expected_names) == 1 and len(keys) == 1:
            name = expected_names[0]
            voices[name] = normalize_voice(archive[keys[0]], name=name)
    missing = [name for name in expected_names if name not in voices]
    if missing:
        raise BuildError(
            f"Voice archive {path} does not contain expected speaker(s): {missing}"
        )
    return voices


def resolve_prepacked_voices(
    repo_id: str,
    revision: str,
    cache_dir: Path,
    filename: str,
    names: list[str],
    *,
    archive_or_raw: bool,
) -> dict[str, np.ndarray]:
    local = hf_download(repo_id, filename, revision, cache_dir)
    if archive_or_raw and zipfile.is_zipfile(local):
        return _read_npz_voice_archive(local, names)
    return _read_raw_voice_file(local, names)


def resolve_voices(profile: dict[str, Any], cache_dir: Path) -> dict[str, np.ndarray]:
    repo_id = profile["repo_id"]
    revision = profile.get("revision", "main")
    spec = profile["voices"]
    kind = spec["kind"]
    if kind == "json_pt":
        return resolve_json_pt_voices(repo_id, revision, cache_dir, spec["index"])
    if kind == "fixed_pt":
        return resolve_fixed_pt_voices(repo_id, revision, cache_dir, spec["items"])
    if kind == "raw":
        return resolve_prepacked_voices(
            repo_id,
            revision,
            cache_dir,
            spec["path"],
            list(spec["names"]),
            archive_or_raw=False,
        )
    if kind == "archive_or_raw":
        return resolve_prepacked_voices(
            repo_id,
            revision,
            cache_dir,
            spec["path"],
            list(spec["names"]),
            archive_or_raw=True,
        )
    raise BuildError(f"Unsupported voices.kind={kind!r}")


def write_sherpa_voices_bin(
    voices: Mapping[str, np.ndarray], out_path: Path
) -> tuple[int, int, int]:
    shape = _validate_same_shape(voices)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for arr in voices.values():
            f.write(np.asarray(arr, dtype="<f4").tobytes(order="C"))

    expected = len(voices) * int(np.prod(shape)) * 4
    actual = out_path.stat().st_size
    if expected != actual:
        raise BuildError(f"Voice pack size mismatch: expected {expected}, got {actual}")
    return shape


def write_numpy_voice_archive(
    voices: Mapping[str, np.ndarray], out_path: Path
) -> tuple[int, int, int]:
    """Write a named NumPy archive for pykokoro consumers."""
    shape = _validate_same_shape(voices)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{name: arr for name, arr in voices.items()})
    if not zipfile.is_zipfile(out_path):
        raise BuildError(f"Voice archive {out_path} is not a NumPy zip archive")
    return shape


def _waveform_only_wrapper(model_for_onnx: Any) -> Any:
    import torch

    class WaveformOnly(torch.nn.Module):
        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner

        def forward(self, tokens: Any, style: Any, speed: Any) -> Any:
            result = self.inner(tokens, style, speed)
            if isinstance(result, (tuple, list)):
                return result[0]
            return result

    return WaveformOnly(model_for_onnx)


def export_checkpoint_to_onnx(
    checkpoint: Path,
    config_path: Path,
    out_path: Path,
    *,
    opset: int,
    seq_len: int,
) -> None:
    import torch
    from kokoro import KModel
    from kokoro.model import KModelForONNX

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    n_token = int(config.get("n_token", len(config.get("vocab", {})) or 178))
    model = (
        KModel(
            repo_id="not-used-local-checkpoint",
            model=str(checkpoint),
            config=config,
            disable_complex=True,
        )
        .to("cpu")
        .eval()
    )
    wrapper = _waveform_only_wrapper(KModelForONNX(model).eval()).eval()

    inner_len = max(8, int(seq_len))
    middle = torch.randint(1, max(2, n_token), (inner_len,), dtype=torch.long)
    tokens = torch.cat(
        [torch.zeros(1, dtype=torch.long), middle, torch.zeros(1, dtype=torch.long)]
    ).unsqueeze(0)
    style = torch.rand(1, STYLE_WIDTH, dtype=torch.float32)
    # sherpa-onnx supplies speed as a 1-element float tensor.
    speed = torch.tensor([1.0], dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (tokens, style, speed),
            str(out_path),
            input_names=["tokens", "style", "speed"],
            output_names=["audio"],
            dynamic_axes={
                "tokens": {1: "sequence_length"},
                "audio": {0: "audio_length"},
            },
            opset_version=opset,
            dynamo=False,
        )


def resolve_model(
    profile: dict[str, Any],
    cache_dir: Path,
    out_path: Path,
    *,
    opset: int,
    seq_len: int,
) -> Path | None:
    repo_id = profile["repo_id"]
    revision = profile.get("revision", "main")
    spec = profile["model"]
    config_local: Path | None = None
    if spec.get("config"):
        config_local = hf_download(repo_id, spec["config"], revision, cache_dir)

    if spec["kind"] == "onnx":
        source = hf_download(repo_id, spec["path"], revision, cache_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, out_path)
        return config_local

    if spec["kind"] == "checkpoint":
        if config_local is None:
            raise BuildError("Checkpoint export requires model.config")
        checkpoint = hf_download(repo_id, spec["path"], revision, cache_dir)
        export_checkpoint_to_onnx(
            checkpoint,
            config_local,
            out_path,
            opset=opset,
            seq_len=seq_len,
        )
        return config_local

    raise BuildError(f"Unsupported model.kind={spec['kind']!r}")

def resolve_auxiliary_files(
    profile: dict[str, Any], cache_dir: Path, out_dir: Path) -> list[dict[str, str]]:
    repo_id = profile["repo_id"]
    revision = profile.get("revision", "main")
    resolved: list[dict[str, str]] = []
    for item in profile.get("auxiliary_files", []):
        source = str(item["path"])
        output = str(item.get("output") or Path(source).name)
        local = hf_download(repo_id, source, revision, cache_dir)
        target = out_dir / output
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        resolved.append(
            {
                "path": output,
                "role": str(item.get("role", "auxiliary")),
                "format": str(item.get("format", "unknown")),
            }
        )
    return resolved


def _onnx_type_name(elem_type: int) -> str:
    import onnx

    names = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.DOUBLE: "float64",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT32: "int32",
    }
    return names.get(elem_type, onnx.TensorProto.DataType.Name(elem_type).lower())


def validate_onnx_contract(
    path: Path, expected: dict[str, Any] | None = None
) -> list[str]:
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    issues: list[str] = []
    actual_inputs = {
        value.name: _onnx_type_name(value.type.tensor_type.elem_type)
        for value in model.graph.input
    }
    expected_inputs = (expected or {}).get("inputs")
    if expected_inputs:
        for name, expected_type in expected_inputs.items():
            actual_type = actual_inputs.get(name)
            if actual_type is None:
                issues.append(f"missing input {name!r}")
            elif actual_type != expected_type:
                issues.append(
                    f"input {name!r} is {actual_type}, expected {expected_type}"
                )
    else:
        if len(model.graph.input) < 3:
            raise BuildError(
                f"{path} exposes only {len(model.graph.input)} inputs; expected 3"
            )
        positional = list(model.graph.input[:3])
        expected_types = ["int64", "float32", "float32"]
        labels = ["token", "style", "speed"]
        for value, label, expected_type in zip(positional, labels, expected_types):
            actual_type = _onnx_type_name(value.type.tensor_type.elem_type)
            if actual_type != expected_type:
                issues.append(
                    f"{label} input {value.name!r} is {actual_type}, "
                    f"expected {expected_type}"
                )

    actual_outputs = {
        value.name: _onnx_type_name(value.type.tensor_type.elem_type)
        for value in model.graph.output
    }
    expected_outputs = (expected or {}).get("outputs")
    if expected_outputs:
        for name, expected_type in expected_outputs.items():
            actual_type = actual_outputs.get(name)
            if actual_type is None:
                issues.append(f"missing output {name!r}")
            elif actual_type != expected_type:
                issues.append(
                    f"output {name!r} is {actual_type}, expected {expected_type}"
                )
    elif not actual_outputs:
        issues.append("model has no outputs")

    if "sample_rate" in (expected or {}) and expected["sample_rate"] != 24000:
        issues.append("sample_rate metadata must be 24000 Hz")
    max_tokens = (expected or {}).get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        issues.append("max_tokens metadata must be a positive integer")
    return issues


def add_sherpa_metadata(
    onnx_path: Path,
    profile: dict[str, Any],
    voices: Mapping[str, np.ndarray],
    style_shape: tuple[int, int, int],
) -> None:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    names = list(voices.keys())
    speaker2id = {name: i for i, name in enumerate(names)}
    frontend = profile["frontend"]

    # These profiles are single-language custom fine-tunes. sherpa's version>=2
    # path means "multi-lingual Kokoro frontend", so version 1 is intentional.
    meta = {
        "model_type": "kokoro",
        "language": profile["language"],
        "has_espeak": int(frontend["kind"] in {"espeak", "espeak_plus_preprocess"}),
        "sample_rate": int(profile.get("sample_rate", 24000)),
        "version": 1,
        "voice": profile["language"],
        "style_dim": ",".join(map(str, style_shape)),
        "n_speakers": len(names),
        "id2speaker": ",".join(f"{i}->{name}" for i, name in enumerate(names)),
        "speaker2id": ",".join(f"{name}->{i}" for name, i in speaker2id.items()),
        "speaker_names": ",".join(names),
        "model_url": f"https://huggingface.co/{profile['repo_id']}",
        "source_repo": profile["repo_id"],
        "license": profile["license"],
        "frontend": frontend["name"],
        "sherpa_text_compatible": str(bool(frontend["sherpa_text_compatible"])).lower(),
        "comment": frontend["note"],
    }

    # Preserve source-specific metadata (for example an embedded vocab) and
    # overwrite only the keys this builder owns.
    merged = {prop.key: prop.value for prop in model.metadata_props}
    merged.update({key: str(value) for key, value in meta.items()})
    while model.metadata_props:
        model.metadata_props.pop()
    for key, value in merged.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, str(onnx_path))


def write_bundle_manifest(
    out_dir: Path,
    profile_key: str,
    profile: dict[str, Any],
    voices: Mapping[str, np.ndarray],
    style_shape: tuple[int, int, int],
    contract_issues: list[str],
    config_local: Path | None,
    auxiliary_files: list[dict[str, str]],
) -> None:
    manifest = {
        "profile": profile_key,
        "source_repo": profile["repo_id"],
        "revision": profile.get("revision", "main"),
        "license": profile["license"],
        "language": profile["language"],
        "sample_rate": profile.get("sample_rate", 24000),
        "speakers": [{"sid": i, "name": name} for i, name in enumerate(voices.keys())],
        "voice_artifacts": {
            "pykokoro": {"path": "voices.npz", "format": "numpy-npz"},
            "sherpa": {"path": "voices.raw.bin", "format": "raw-float32-le"},
        },
        "auxiliary_artifacts": auxiliary_files,
        "style_dim": list(style_shape),
        "frontend": profile["frontend"],
        "onnx_contract": profile.get("onnx_contract", {}),
        "onnx_contract_issues": contract_issues,
        "publication": profile.get("release", {}),
    }
    with (out_dir / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if config_local is not None:
        shutil.copyfile(config_local, out_dir / "config.json")


def build_profile(
    profile_key: str,
    profile: dict[str, Any],
    out_root: Path,
    *,
    opset: int,
    seq_len: int,
    run_checker: bool,
) -> Path:
    out_dir = out_root / profile_key
    cache_dir = out_dir / ".hf"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{profile_key}] resolving voices", file=sys.stderr)
    voices = resolve_voices(profile, cache_dir)
    style_shape = write_sherpa_voices_bin(voices, out_dir / "voices.raw.bin")
    write_numpy_voice_archive(voices, out_dir / "voices.npz")

    print(f"[{profile_key}] resolving model", file=sys.stderr)
    config_local = resolve_model(
        profile,
        cache_dir,
        out_dir / "model.onnx",
        opset=opset,
        seq_len=seq_len,
    )


    auxiliary_files = resolve_auxiliary_files(
        profile, cache_dir, out_dir
    )
    contract = dict(profile.get("onnx_contract", {}))
    contract["sample_rate"] = profile.get("sample_rate", 24000)
    contract_issues = validate_onnx_contract(
        out_dir / "model.onnx", contract
    )
    if contract_issues:
        for issue in contract_issues:
            print(f"warning: {issue}", file=sys.stderr)

    add_sherpa_metadata(out_dir / "model.onnx", profile, voices, style_shape)

    if run_checker:
        import onnx

        onnx.checker.check_model(str(out_dir / "model.onnx"))

    write_bundle_manifest(
        out_dir,
        profile_key,
        profile,
        voices,
        style_shape,
        contract_issues,
        config_local,
        auxiliary_files,
    )
    shutil.rmtree(cache_dir, ignore_errors=True)
    return out_dir


def list_profiles(profiles: Mapping[str, dict[str, Any]]) -> None:
    print(f"{'PROFILE':16} {'LANG':5} {'MODEL':10} {'TEXT FRONTEND':16} SOURCE")
    for key, profile in profiles.items():
        front = profile["frontend"]
        status = "stock-sherpa" if front["sherpa_text_compatible"] else "external"
        print(
            f"{key:16} {profile['language']:5} {profile['model']['kind']:10} "
            f"{status:16} {profile['repo_id']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Kokoro-family ONNX/sherpa voice bundles from Hugging Face profiles"
    )
    parser.add_argument("--profiles", type=Path, default=PROFILE_FILE)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available model profiles")

    build = sub.add_parser("build", help="Build one profile or all profiles")
    build.add_argument("profile", help="Profile key or 'all'")
    build.add_argument("--out", type=Path, default=Path("build"))
    build.add_argument("--opset", type=int, default=14)
    build.add_argument("--seq-len", type=int, default=64)
    build.add_argument("--skip-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles)
    if args.command == "list":
        list_profiles(profiles)
        return 0

    keys = list(profiles) if args.profile == "all" else [args.profile]
    unknown = [key for key in keys if key not in profiles]
    if unknown:
        raise BuildError(f"Unknown profile(s): {', '.join(unknown)}")

    for key in keys:
        out = build_profile(
            key,
            profiles[key],
            args.out,
            opset=args.opset,
            seq_len=args.seq_len,
            run_checker=not args.skip_check,
        )
        print(out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
