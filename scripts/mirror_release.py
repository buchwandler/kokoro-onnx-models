#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "releases.json"


def request_json(url: str) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "kokoro-onnx-models-mirror"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path) -> None:
    headers = {"User-Agent": "kokoro-onnx-models-mirror"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, path.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description="Mirror selected assets from an upstream public GitHub release")
    ap.add_argument("release_key")
    ap.add_argument("--dist", type=Path, default=Path("dist"))
    args = ap.parse_args()

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    releases = catalog["releases"]
    if args.release_key not in releases:
        raise SystemExit(f"Unknown release key: {args.release_key}")
    spec = releases[args.release_key]
    if spec.get("kind") != "mirror":
        raise SystemExit(f"{args.release_key} is a build profile, not a mirrored release")

    repo = spec["source_repository"]
    source_tag = spec["source_tag"]
    api = f"https://api.github.com/repos/{repo}/releases/tags/{source_tag}"
    release = request_json(api)
    by_name = {item["name"]: item for item in release.get("assets", [])}
    missing = [name for name in spec["assets"] if name not in by_name]
    if missing:
        raise SystemExit("Upstream release is missing expected assets: " + ", ".join(missing))

    out = args.dist / spec["tag"]
    out.mkdir(parents=True, exist_ok=True)
    assets = []
    for name in spec["assets"]:
        target = out / name
        if not target.is_file():
            print(f"Downloading {name}")
            download(by_name[name]["browser_download_url"], target)
        assets.append({"name": name, "size": target.stat().st_size, "sha256": sha256(target)})

    manifest = {
        "schema": 1,
        "repository": catalog["target_repository"],
        "tag": spec["tag"],
        "profile": args.release_key,
        "source": {"type": "github-release", "repository": repo, "tag": source_tag, "url": release.get("html_url")},
        "license": spec["license"],
        "language": None,
        "frontend": None,
        "assets": assets,
    }
    (out / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
