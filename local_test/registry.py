"""Download and validate one complete runtime registry distribution."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "catalog" / "models.json"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_distribution(
    profile: str, *, provider: str | None = None, path: Path = REGISTRY_PATH
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = load_registry(path)
    try:
        model = registry["models"][profile]
    except KeyError as exc:
        raise ValueError(f"Unknown registry profile: {profile}") from exc
    distributions = [
        item
        for item in model.get("distributions", [])
        if provider is None or item["provider"] == provider
    ]
    if not distributions:
        raise ValueError(f"No matching runtime distribution for {profile}")
    return model, distributions[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_format(path: Path, artifact: dict[str, Any]) -> None:
    fmt = artifact["format"]
    if fmt == "json":
        json.loads(path.read_text(encoding="utf-8"))
    elif fmt == "numpy-npz":
        if not zipfile.is_zipfile(path):
            raise ValueError(f"{artifact['id']}: expected NumPy archive")
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError(f"{artifact['id']}: archive is empty")
    elif fmt == "raw-float32-le":
        if path.stat().st_size % 4:
            raise ValueError(f"{artifact['id']}: raw voice is not float32-aligned")
        handling = artifact.get("handling") or {}
        shape = handling.get("shape")
        count = int(handling.get("voice_count", 1))
        if shape and path.stat().st_size != shape[0] * shape[1] * 4 * count:
            raise ValueError(
                f"{artifact['id']}: raw voice size does not match its shape"
            )


def _download(artifact: dict[str, Any], target: Path) -> None:
    request = urllib.request.Request(
        artifact["url"], headers={"User-Agent": "kokoro-onnx-models-local-test"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        target.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if target.stat().st_size != artifact["size"]:
        raise ValueError(f"{artifact['id']}: downloaded size does not match registry")
    if _sha256(target) != artifact["sha256"]:
        raise ValueError(
            f"{artifact['id']}: downloaded SHA-256 does not match registry"
        )
    _validate_format(target, artifact)


def _target_name(artifact: dict[str, Any], layout: str) -> str:
    role = artifact["role"]
    if role == "model":
        return (
            f"{artifact['component']}.onnx"
            if layout == "split-onnx-v1"
            else "model.onnx"
        )
    if role == "voice":
        return artifact["local_name"]
    if role == "voices":
        return "voices.npz" if artifact["format"] == "numpy-npz" else "voices.bin"
    if role == "config":
        return "onnx_manifest.json" if layout == "split-onnx-v1" else "config.json"
    if role == "vocab":
        return "vocab.json"
    if role == "bundle":
        return "bundle.json"
    return artifact["local_name"]


def download_distribution(
    profile: str,
    target: Path,
    *,
    provider: str | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    model, distribution = select_distribution(
        profile, provider=provider, path=registry_path
    )
    target.mkdir(parents=True, exist_ok=True)
    layout = model["runtime"]["layout"]
    with tempfile.TemporaryDirectory(prefix="kokoro-registry-download-") as temporary:
        temporary_dir = Path(temporary)
        for artifact in distribution["artifacts"]:
            source = temporary_dir / artifact["local_name"]
            _download(artifact, source)
            destination = target / _target_name(artifact, layout)
            if destination.exists():
                destination.unlink()
            shutil.copy2(source, destination)
        voice_artifacts = [
            artifact
            for artifact in distribution["artifacts"]
            if artifact["role"] == "voice" and artifact["format"] == "raw-float32-le"
        ]
        if voice_artifacts and not any(
            artifact["role"] == "voices" for artifact in distribution["artifacts"]
        ):
            arrays = {}
            for artifact in voice_artifacts:
                values = np.fromfile(target / artifact["local_name"], dtype="<f4")
                rows, width = (artifact.get("handling") or {}).get("shape", [0, 256])
                if not rows or values.size != rows * width:
                    raise ValueError(f"{artifact['id']}: raw voice shape is invalid")
                arrays[artifact["voice"]] = values.reshape(rows, 1, width)
            np.savez(target / "voices.npz", **arrays)
    return {"model": model, "distribution": distribution}
