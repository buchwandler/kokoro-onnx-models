#!/usr/bin/env python3
"""Validate the client-visible Kokoro runtime model registry."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "catalog" / "models.json"
SCHEMA = ROOT / "schemas" / "model-registry.schema.json"
RELEASES = ROOT / "catalog" / "releases.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FRONTEND_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2,4})?$")
HF_URL_RE = re.compile(r"^https://huggingface\.co/[^/]+/[^/]+/resolve/([^/]+)/.+$")


class RegistryError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _validate_schema(registry: dict[str, Any], schema_path: Path) -> None:
    schema = _load(schema_path)
    try:
        import jsonschema
    except ImportError:
        _require(registry.get("schema") == 1, "Registry schema must be 1")
        return
    try:
        jsonschema.Draft202012Validator(schema).validate(registry)
    except jsonschema.ValidationError as exc:
        raise RegistryError(f"Schema validation failed: {exc.message}") from exc


def _validate_artifact(artifact: dict[str, Any], *, model_id: str, distribution_id: str) -> None:
    required = ("id", "role", "url", "local_name", "format", "size", "sha256")
    _require(all(field in artifact for field in required), f"{model_id}/{distribution_id}: artifact missing required field")
    _require(isinstance(artifact["id"], str) and artifact["id"], f"{model_id}: artifact id is empty")
    _require(urlparse(artifact["url"]).scheme == "https", f"{model_id}: artifact URL is not HTTPS: {artifact['url']}")
    _require(isinstance(artifact["size"], int) and not isinstance(artifact["size"], bool) and artifact["size"] > 0, f"{model_id}: invalid size for {artifact['id']}")
    _require(isinstance(artifact["sha256"], str) and SHA256_RE.fullmatch(artifact["sha256"]) is not None, f"{model_id}: invalid SHA-256 for {artifact['id']}")
    role = artifact["role"]
    if role == "model":
        _require(bool(artifact.get("quality")), f"{model_id}: model {artifact['id']} is missing quality")
    if role in {"voice", "voices"} and artifact["format"] == "raw-float32-le":
        handling = artifact.get("handling") or {}
        _require(handling.get("dtype") == "float32", f"{model_id}: raw voice {artifact['id']} needs float32 dtype")
        _require(handling.get("endianness") == "little", f"{model_id}: raw voice {artifact['id']} needs little endianness")
        shape = handling.get("shape")
        _require(isinstance(shape, list) and len(shape) == 2 and all(isinstance(x, int) and x > 0 for x in shape), f"{model_id}: raw voice {artifact['id']} needs a positive 2D shape")
    _require("?" not in artifact["url"] or "resolve/" in artifact["url"], f"{model_id}: provider URL must identify its revision")


def _validate_distribution(model_id: str, model: dict[str, Any], distribution: dict[str, Any], releases: dict[str, Any]) -> None:
    distribution_id = distribution["id"]
    _require(distribution_id not in _distribution_ids, f"Duplicate distribution id: {distribution_id}")
    _distribution_ids.add(distribution_id)
    _require(distribution.get("runtime_ready") is True, f"{model_id}/{distribution_id}: distribution is not runtime-ready")
    artifacts = distribution["artifacts"]
    artifact_ids: set[str] = set()
    model_count = 0
    voice_count = 0
    components: set[str] = set()
    voice_names: set[str] = set()
    for artifact in artifacts:
        _require(isinstance(artifact, dict), f"{model_id}/{distribution_id}: artifact is not an object")
        _require(artifact["id"] not in artifact_ids, f"{model_id}/{distribution_id}: duplicate artifact id {artifact['id']}")
        artifact_ids.add(artifact["id"])
        _validate_artifact(artifact, model_id=model_id, distribution_id=distribution_id)
        if artifact["role"] == "model":
            model_count += 1
            if artifact.get("component"):
                components.add(artifact["component"])
        if artifact["role"] in {"voice", "voices"}:
            voice_count += 1
            if artifact["role"] == "voice":
                _require(bool(artifact.get("voice")), f"{model_id}: voice artifact needs voice name")
                voice_names.add(artifact["voice"])
        if distribution["provider"] == "huggingface":
            match = HF_URL_RE.match(artifact["url"])
            _require(match is not None, f"{model_id}: Hugging Face artifact URL is not pinned: {artifact['url']}")
            _require(match.group(1) != "main", f"{model_id}: Hugging Face runtime URL uses main")
    _require(model_count > 0, f"{model_id}/{distribution_id}: no model artifact")
    _require(voice_count > 0, f"{model_id}/{distribution_id}: no voice artifact")
    if voice_names:
        _require(voice_names == set(model["runtime"]["voices"]), f"{model_id}/{distribution_id}: voice artifacts do not match voice roster")
    if model["runtime"]["layout"] == "split-onnx-v1":
        expected = set(model["onnx_contract"].get("components") or {})
        _require(expected and expected == components, f"{model_id}/{distribution_id}: split model components do not match contract")
    if distribution["provider"] == "huggingface":
        _require(distribution.get("repository") and distribution.get("revision"), f"{model_id}/{distribution_id}: Hugging Face provenance is incomplete")
    if distribution["provider"] == "github-release":
        release_key = distribution.get("release_key")
        _require(release_key in releases, f"{model_id}/{distribution_id}: unknown release key {release_key!r}")
        release = releases[release_key]
        _require(release.get("tag") == distribution.get("release_tag"), f"{model_id}/{distribution_id}: release tag mismatch")


def verify_registry(registry_path: Path = REGISTRY, schema_path: Path = SCHEMA, releases_path: Path = RELEASES) -> dict[str, Any]:
    global _distribution_ids
    _distribution_ids = set()
    registry = _load(registry_path)
    releases = _load(releases_path)
    _validate_schema(registry, schema_path)
    _require(registry.get("schema") == 1, "Registry schema must be 1")
    _require(registry.get("runtime_contract") == 1, "Registry runtime contract must be 1")
    release_entries = releases.get("releases")
    _require(isinstance(release_entries, dict), "Release catalog has no releases object")
    models = registry.get("models")
    _require(isinstance(models, dict) and models, "Registry has no models")
    _require("ru-zaakirio-base" in models and "ru-zaakirio-dima" in models, "Russian profiles are missing from registry")
    _require("ru-zaakirio-base" not in release_entries and "ru-zaakirio-dima" not in release_entries, "Russian profiles must not be release jobs")
    for model_id, model in models.items():
        _require(FRONTEND_RE.fullmatch(model["frontend"]) is not None, f"{model_id}: frontend must be a machine-readable ID")
        _require(all(LANGUAGE_RE.fullmatch(code) for code in model["language_codes"]), f"{model_id}: invalid language code")
        runtime = model["runtime"]
        _require(runtime["default_voice"] in runtime["voices"], f"{model_id}: default voice is not in voice roster")
        _require(model["mirror_policy"] in {"required", "preferred", "optional", "forbidden"}, f"{model_id}: invalid mirror policy")
        distributions = model["distributions"]
        _require(bool(distributions) == bool(model.get("runtime_available", True)), f"{model_id}: runtime_available does not match distributions")
        distribution_ids = [item["id"] for item in distributions]
        _require(len(distribution_ids) == len(set(distribution_ids)), f"{model_id}: duplicate distribution ID")
        for distribution in distributions:
            _validate_distribution(model_id, model, distribution, release_entries)
            if model["mirror_policy"] == "forbidden":
                _require(distribution["provider"] != "github-release", f"{model_id}: forbidden mirror has GitHub distribution")
    for release_key in release_entries:
        _require(release_key in models, f"Release catalog key {release_key!r} is missing from model registry")
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the Kokoro runtime model registry")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--schema", type=Path, default=SCHEMA)
    parser.add_argument("--releases", type=Path, default=RELEASES)
    args = parser.parse_args(argv)
    try:
        registry = verify_registry(args.registry, args.schema, args.releases)
    except RegistryError as exc:
        print(f"registry verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {len(registry['models'])} runtime models and {len(_distribution_ids)} distributions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
