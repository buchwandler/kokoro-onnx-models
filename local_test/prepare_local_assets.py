#!/usr/bin/env python3
"""Populate ignored local test assets using the repository's own build/mirror tools.

Nothing is downloaded into git-tracked paths. Output goes below ``.local-test/``.

Examples:
    python local_test/prepare_local_assets.py v1.2-de-martin
    python local_test/prepare_local_assets.py vi-contextbox
    python local_test/prepare_local_assets.py all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local-test"
ASSETS = LOCAL / "assets"
DOWNLOADS = LOCAL / "downloads"
BUILDS = LOCAL / "build"
CATALOG_PATH = ROOT / "catalog" / "releases.json"


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"staged {target.relative_to(ROOT)}")


def _catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _asset_name(item: str | dict[str, Any]) -> str:
    return item if isinstance(item, str) else str(item.get("name") or item["source"])


def _select_model_asset(items: list[str | dict[str, Any]]) -> str:
    names = [_asset_name(item) for item in items]
    fp32 = [
        name
        for name in names
        if name.endswith(".onnx")
        and all(marker not in name for marker in (".fp16", ".int8", ".q", "uint8"))
    ]
    if len(fp32) != 1:
        raise RuntimeError(f"Cannot uniquely choose fp32 ONNX asset from {names}")
    return fp32[0]


def _select_voice_asset(items: list[str | dict[str, Any]]) -> str:
    names = [_asset_name(item) for item in items]
    voices = [
        name
        for name in names
        if "voice" in name.lower() and name.endswith((".bin", ".npz"))
    ]
    npz = [name for name in voices if name.endswith(".npz")]
    if len(npz) == 1:
        return npz[0]
    if len(voices) != 1:
        raise RuntimeError(f"Cannot uniquely choose pykokoro voice asset from {names}")
    return voices[0]


def prepare_mirror(key: str, spec: dict[str, Any]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "mirror_release.py"),
            key,
            "--dist",
            str(DOWNLOADS),
        ],
        cwd=ROOT,
        check=True,
    )
    source_dir = DOWNLOADS / spec["tag"]
    target = ASSETS / key
    target.mkdir(parents=True, exist_ok=True)
    layout = str(spec.get("runtime_layout", "single-onnx-v1"))
    if layout == "single-onnx-v1":
        model_name = _select_model_asset(spec["assets"])
        voice_name = _select_voice_asset(spec["assets"])
        _copy(source_dir / model_name, target / "model.onnx")
        _copy(source_dir / voice_name, target / "voices.bin")
    elif layout == "split-onnx-v1":
        for item in spec["assets"]:
            name = _asset_name(item)
            role = item.get("role") if isinstance(item, dict) else None
            if role == "model":
                component = str(item.get("component") or "")
                if not component:
                    raise RuntimeError(f"Split model asset has no component: {name}")
                target_name = f"{component}.onnx"
            elif role == "config":
                target_name = "onnx_manifest.json"
            elif role == "metadata":
                source_name = Path(str(item.get("source") or name)).name
                target_name = {
                    "source_params.npz": "source_params.npz",
                    "styles.npz": "styles.npz",
                }.get(source_name, Path(name).name)
            elif role == "voices":
                target_name = "voices.npz"
            else:
                continue
            _copy(source_dir / name, target / target_name)
    else:
        raise RuntimeError(f"Unsupported runtime layout: {layout}")
    manifest = source_dir / "release-manifest.json"
    if manifest.is_file():
        _copy(manifest, target / "release-manifest.json")


def prepare_build(key: str, spec: dict[str, Any]) -> None:
    profile = str(spec["profile"])
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_kokoro.py"),
            "build",
            profile,
            "--out",
            str(BUILDS),
        ],
        cwd=ROOT,
        check=True,
    )
    source_dir = BUILDS / profile
    target = ASSETS / key
    target.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("model.onnx", "model.onnx"),
        ("voices.npz", "voices.npz"),
        ("voices.raw.bin", "voices.raw.bin"),
        ("bundle.json", "bundle.json"),
        ("config.json", "config.json"),
        ("vocab.json", "vocab.json"),
    ):
        source = source_dir / source_name
        if source.is_file():
            _copy(source, target / target_name)


def main() -> int:
    catalog = _catalog()
    releases: dict[str, dict[str, Any]] = catalog["releases"]

    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=[*releases, "all"])
    args = parser.parse_args()

    keys = list(releases) if args.profile == "all" else [args.profile]
    for key in keys:
        spec = releases[key]
        print(f"\n=== preparing {key} ===")
        if spec.get("kind") == "mirror":
            prepare_mirror(key, spec)
        elif spec.get("kind") == "build":
            prepare_build(key, spec)
        else:
            raise RuntimeError(
                f"Unsupported catalog kind for {key}: {spec.get('kind')!r}"
            )
    print(f"\nLocal assets are under {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
