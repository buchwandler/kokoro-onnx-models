#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "scripts" / "kokoro_profiles.json"
TARGET_REPOSITORY = "buchwandler/kokoro-onnx-models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_profiles() -> dict[str, dict[str, Any]]:
    return json.loads(PROFILES.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Rename a built profile into GitHub Release-ready assets")
    ap.add_argument("profile")
    ap.add_argument("--build-root", type=Path, default=Path("build"))
    ap.add_argument("--dist", type=Path, default=Path("dist"))
    ap.add_argument("--allow-restricted", action="store_true")
    args = ap.parse_args()

    profiles = load_profiles()
    if args.profile not in profiles:
        raise SystemExit(f"Unknown profile: {args.profile}")
    profile = profiles[args.profile]
    release = profile.get("release") or {}
    if not release.get("enabled", False) and not args.allow_restricted:
        raise SystemExit(
            f"Release disabled for {args.profile}. Review MODEL_LICENSES.md and pass --allow-restricted only if redistribution is permitted."
        )

    src = args.build_root / args.profile
    required = [src / "model.onnx", src / "voices.bin", src / "bundle.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("Missing build artifacts: " + ", ".join(missing))

    tag = release["tag"]
    out = args.dist / tag
    out.mkdir(parents=True, exist_ok=True)

    mapping = {
        src / "model.onnx": out / release["model_filename"],
        src / "voices.bin": out / release["voices_filename"],
        src / "bundle.json": out / "bundle.json",
    }
    if (src / "config.json").is_file() and release.get("config_filename"):
        mapping[src / "config.json"] = out / release["config_filename"]

    for source, target in mapping.items():
        shutil.copy2(source, target)

    assets = []
    for path in sorted(out.iterdir()):
        if path.name == "release-manifest.json" or not path.is_file():
            continue
        assets.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)})

    manifest = {
        "schema": 1,
        "repository": TARGET_REPOSITORY,
        "tag": tag,
        "profile": args.profile,
        "source": {"type": "huggingface", "repository": profile["repo_id"], "revision": profile.get("revision", "main")},
        "license": profile["license"],
        "language": profile.get("language"),
        "frontend": profile.get("frontend"),
        "assets": assets,
    }
    (out / "release-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
