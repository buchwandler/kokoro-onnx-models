#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "releases.json"
TARGET_REPOSITORY = "buchwandler/kokoro-onnx-models"


@dataclass(frozen=True)
class MirrorAsset:
    source: str
    name: str
    role: str = "metadata"
    format: str = "unknown"
    quality: str | None = None
    sha256: str | None = None
    size: int | None = None


def request_json(url: str) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "kokoro-onnx-models-mirror",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "kokoro-onnx-models-mirror"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        path.open("wb") as file,
    ):
        while chunk := response.read(1024 * 1024):
            file.write(chunk)


def normalize_assets(items: list[str | dict[str, Any]]) -> list[MirrorAsset]:
    assets = []
    for item in items:
        if isinstance(item, str):
            assets.append(MirrorAsset(source=item, name=item))
            continue
        if isinstance(item, dict):
            assets.append(
                MirrorAsset(
                    source=str(item["source"]),
                    name=str(item.get("name", item["source"])),
                    role=str(item.get("role", "metadata")),
                    format=str(item.get("format", "unknown")),
                    quality=item.get("quality"),
                    sha256=item.get("sha256"),
                    size=item.get("size"),
                )
            )
            continue
        raise SystemExit(f"Invalid mirror asset entry: {item!r}")
    return assets


def verify_asset(path: Path, asset: MirrorAsset) -> None:
    actual_size = path.stat().st_size
    if asset.size is not None and actual_size != asset.size:
        raise SystemExit(
            f"Size mismatch for {asset.name}: expected {asset.size}, got {actual_size}"
        )
    actual_sha256 = sha256(path)
    if asset.sha256 is not None and actual_sha256 != asset.sha256:
        raise SystemExit(
            f"SHA-256 mismatch for {asset.name}: expected {asset.sha256}, got {actual_sha256}"
        )


def asset_matches(path: Path, asset: MirrorAsset) -> bool:
    if not path.is_file():
        return False
    try:
        verify_asset(path, asset)
    except (OSError, SystemExit):
        return False
    return True


def stage_asset(url: str, target: Path, asset: MirrorAsset) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{target.name}.",
        suffix=".part",
        dir=target.parent,
        delete=False,
    ) as file:
        temporary = Path(file.name)
    try:
        download(url, temporary)
        verify_asset(temporary, asset)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def github_sources(
    spec: dict[str, Any], assets: list[MirrorAsset]
) -> tuple[dict[str, str], dict[str, Any]]:
    repository = spec["source_repository"]
    source_tag = spec["source_tag"]
    release = request_json(
        f"https://api.github.com/repos/{repository}/releases/tags/{source_tag}"
    )
    by_name = {item["name"]: item for item in release.get("assets", [])}
    missing = [asset.source for asset in assets if asset.source not in by_name]
    if missing:
        raise SystemExit(
            "Upstream release is missing expected assets: " + ", ".join(missing)
        )
    urls = {
        asset.name: by_name[asset.source]["browser_download_url"] for asset in assets
    }
    source = {
        "type": "github-release",
        "repository": repository,
        "revision": source_tag,
        "tag": source_tag,
        "url": release.get("html_url"),
    }
    return urls, source


def huggingface_sources(
    spec: dict[str, Any], assets: list[MirrorAsset]
) -> tuple[dict[str, str], dict[str, Any]]:
    repository = spec["source_repository"]
    revision = spec["source_revision"]
    urls = {
        asset.name: (
            f"https://huggingface.co/{repository}/resolve/"
            f"{urllib.parse.quote(revision, safe='')}/"
            f"{urllib.parse.quote(asset.source, safe='/')}?download=true"
        )
        for asset in assets
    }
    return urls, {"type": "huggingface", "repository": repository, "revision": revision}


def _runtime(spec: dict[str, Any]) -> dict[str, Any]:
    configured = dict(spec.get("runtime") or {})
    voices = list(configured.get("voices") or ["default"])
    return {
        "language_codes": list(
            spec.get("language_codes") or [spec.get("language", "und")]
        ),
        "sample_rate": int(spec.get("sample_rate", 24000)),
        "frontend": str(spec.get("frontend", "pykokoro-native-v1")),
        "frontend_experimental": bool(spec.get("frontend_experimental", False)),
        "tokenizer_vocab_version": str(spec.get("tokenizer_vocab_version", "1.0")),
        "vocabulary_source": str(spec.get("vocabulary_source", "downloaded-config")),
        "max_tokens": int(spec.get("max_tokens", 510)),
        "default_voice": str(configured.get("default_voice", voices[0])),
        "voices": voices,
    }


def _write_checksums(out: Path, assets: list[dict[str, Any]]) -> None:
    (out / "SHA256SUMS").write_text(
        "\n".join(f"{asset['sha256']}  {asset['name']}" for asset in assets) + "\n",
        encoding="utf-8",
    )


def _write_release_notes(out: Path, manifest: dict[str, Any]) -> None:
    runtime = manifest["runtime"]
    qualities = [
        asset["quality"]
        for asset in manifest["assets"]
        if asset["role"] == "model" and "quality" in asset
    ]
    lines = [
        f"# {manifest['profile']} {manifest['model_version']}",
        "",
        f"- Language(s): {', '.join(runtime['language_codes'])}",
        f"- Frontend: {runtime['frontend']}",
        f"- Model qualities: {', '.join(qualities) or 'unspecified'}",
        f"- Voices: {', '.join(runtime['voices'])}",
        f"- Source: {manifest['source']['repository']} @ {manifest['source']['revision']}",
        f"- License: {manifest['license']}",
        f"- SHA-256: recorded for {len(manifest['assets'])} assets",
    ]
    (out / "release-notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror selected upstream model release assets"
    )
    parser.add_argument("release_key")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    releases = catalog["releases"]
    if args.release_key not in releases:
        raise SystemExit(f"Unknown release key: {args.release_key}")
    spec = releases[args.release_key]
    if spec.get("kind") != "mirror":
        raise SystemExit(
            f"{args.release_key} is a build profile, not a mirrored release"
        )

    assets = normalize_assets(spec["assets"])
    source_type = spec.get("source_type", "github-release")
    if source_type == "github-release":
        urls, source = github_sources(spec, assets)
    elif source_type == "huggingface":
        urls, source = huggingface_sources(spec, assets)
    else:
        raise SystemExit(f"Unsupported mirror source type: {source_type}")

    out = args.dist / spec["tag"]
    out.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for asset in assets:
            target = out / asset.name
            if asset_matches(target, asset):
                print(f"Using existing {asset.name}")
                continue
            print(f"Downloading {asset.name}")
            staged.append((stage_asset(urls[asset.name], target, asset), target))
        for temporary, target in staged:
            temporary.replace(target)
    except BaseException:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise

    manifest_assets = [
        {
            "name": asset.name,
            "role": asset.role,
            "format": asset.format,
            **({"quality": asset.quality} if asset.quality is not None else {}),
            "size": (out / asset.name).stat().st_size,
            "sha256": sha256(out / asset.name),
        }
        for asset in assets
    ]
    contract = dict(spec.get("onnx_contract") or {})
    contract.setdefault(
        "inputs", {"tokens": "int64", "style": "float32", "speed": "float32"}
    )
    contract.setdefault("outputs", {"audio": "float32"})
    contract.setdefault("max_tokens", int(spec.get("max_tokens", 510)))
    manifest: dict[str, Any] = {
        "schema": 2,
        "runtime_contract": 1,
        "repository": catalog.get("target_repository", TARGET_REPOSITORY),
        "tag": spec["tag"],
        "profile": str(spec.get("profile", args.release_key)),
        "model_version": str(spec.get("model_version", spec["tag"])),
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "license": spec["license"],
        "publication": {"enabled": bool(spec.get("publish", True))},
        "runtime": _runtime(spec),
        "onnx_contract": contract,
        "assets": manifest_assets,
    }
    builder_commit = os.environ.get("GITHUB_SHA")
    if builder_commit:
        manifest["builder"] = {
            "repository": os.environ.get("GITHUB_REPOSITORY", TARGET_REPOSITORY),
            "commit": builder_commit,
        }
    (out / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_checksums(out, manifest_assets)
    _write_release_notes(out, manifest)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
