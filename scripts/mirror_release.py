#!/usr/bin/env python3
"""Mirror pinned Hugging Face release assets and build deterministic voice packs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
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
    component: str | None = None
    sha256: str | None = None
    size: int | None = None
    transform: str | None = None
    source_size: int | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class VoiceSource:
    name: str
    path: str
    size: int | None = None
    sha256: str | None = None


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
                    component=item.get("component"),
                    sha256=item.get("sha256"),
                    size=item.get("size"),
                    transform=item.get("transform"),
                    source_size=item.get("source_size"),
                    source_sha256=item.get("source_sha256"),
                )
            )
            continue
        raise SystemExit(f"Invalid mirror asset entry: {item!r}")
    return assets


def verify_asset(path: Path, asset: MirrorAsset) -> None:
    actual_size = path.stat().st_size
    expected_size = asset.source_size if asset.transform else asset.size
    if expected_size is not None and actual_size != expected_size:
        raise SystemExit(
            f"Size mismatch for {asset.name}: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256(path)
    expected_sha256 = asset.source_sha256 if asset.transform else asset.sha256
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise SystemExit(
            f"SHA-256 mismatch for {asset.name}: expected {expected_sha256}, got {actual_sha256}"
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
    source_tag = spec["source_revision"]
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


def huggingface_url(repository: str, revision: str, path: str) -> str:
    return (
        f"https://huggingface.co/{repository}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/"
        f"{urllib.parse.quote(path, safe='/')}?download=true"
    )


def huggingface_sources(
    spec: dict[str, Any], assets: list[MirrorAsset]
) -> tuple[dict[str, str], dict[str, Any]]:
    repository = spec["source_repository"]
    revision = spec["source_revision"]
    urls = {
        asset.name: huggingface_url(repository, revision, asset.source)
        for asset in assets
    }
    return urls, {"type": "huggingface", "repository": repository, "revision": revision}


def _voice_sources(spec: dict[str, Any], pack: dict[str, Any]) -> list[VoiceSource]:
    entries = pack.get("source_assets")
    if entries is not None:
        sources = [
            VoiceSource(
                str(item["name"]),
                str(item.get("path", f"voices/{item['name']}.bin")),
                int(item["size"]) if item.get("size") is not None else None,
                str(item["sha256"]) if item.get("sha256") is not None else None,
            )
            for item in entries
        ]
    else:
        names = [
            str(name)
            for name in pack.get("voices", spec.get("runtime", {}).get("voices", []))
        ]
        sources = [
            VoiceSource(
                name,
                f"{pack.get('source_prefix', 'voices/')}{name}{pack.get('source_suffix', '.bin')}",
            )
            for name in names
        ]
    if len(sources) != len({source.name for source in sources}):
        raise SystemExit("Voice pack contains duplicate voice names")
    expected_count = int(pack.get("expected_count", len(sources)))
    if len(sources) != expected_count:
        raise SystemExit(
            f"Voice pack declares {expected_count} voices but lists {len(sources)}"
        )
    return sources


def _huggingface_tree(
    repository: str, revision: str, path: str = ""
) -> list[dict[str, Any]]:
    url = (
        f"https://huggingface.co/api/models/{repository}/tree/"
        f"{urllib.parse.quote(revision, safe='')}/{path}?recursive=true&expand=true"
    )
    entries: list[dict[str, Any]] = []
    while url:
        request = urllib.request.Request(
            url, headers={"User-Agent": "kokoro-onnx-models-mirror"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)
            if not isinstance(page, list):
                raise SystemExit("Hugging Face tree response is not a list")
            entries.extend(item for item in page if isinstance(item, dict))
            link = response.headers.get("Link", "")
        url = next(
            (
                part.split(";", 1)[0].strip("<>")
                for part in link.split(",")
                if 'rel="next"' in part
            ),
            "",
        )
    return entries


def _discover_voice_sources(
    spec: dict[str, Any], pack: dict[str, Any]
) -> list[VoiceSource]:
    repository = spec["source_repository"]
    revision = spec["source_revision"]
    entries = _huggingface_tree(repository, revision, "voices")
    prefix = str(pack.get("source_prefix", "voices/"))
    suffix = str(pack.get("source_suffix", ".bin"))
    found = {
        item["path"]: item
        for item in entries
        if isinstance(item, dict)
        and str(item.get("path", "")).startswith(prefix)
        and str(item.get("path", "")).endswith(suffix)
    }
    declared = _voice_sources(spec, pack)
    expected_paths = {item.path for item in declared}
    unexpected = sorted(set(found) - expected_paths)
    missing = sorted(expected_paths - set(found))
    if missing:
        raise SystemExit("Upstream is missing expected voices: " + ", ".join(missing))
    if unexpected:
        raise SystemExit(
            "Upstream contains unexpected voices: " + ", ".join(unexpected)
        )
    return [
        VoiceSource(item.name, item.path, int(found[item.path].get("size", 0)) or None)
        for item in declared
    ]


def _voice_source_list(spec: dict[str, Any], pack: dict[str, Any]) -> list[VoiceSource]:
    if pack.get("discover"):
        return _discover_voice_sources(spec, pack)
    return _voice_sources(spec, pack)


def _voice_array(path: Path, source: VoiceSource, style_width: int) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required to build a voice archive") from exc
    raw = path.read_bytes()
    if len(raw) % 4:
        raise SystemExit(f"Voice {source.name} is not float32-aligned")
    values = np.frombuffer(raw, dtype="<f4")
    if values.size == 0 or values.size % style_width:
        raise SystemExit(
            f"Voice {source.name} has {values.size} values, incompatible with style width {style_width}"
        )
    values = values.reshape((-1, style_width))
    if not np.isfinite(values).all():
        raise SystemExit(f"Voice {source.name} contains non-finite values")
    return values.copy()


def _npy_bytes(array: Any) -> bytes:
    import numpy as np

    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def pack_voice_archive(
    sources: list[tuple[VoiceSource, Path]], target: Path, *, style_width: int = 256
) -> list[dict[str, Any]]:
    """Pack validated raw voices into a reproducible, named NumPy archive."""
    if len({source.name for source, _ in sources}) != len(sources):
        raise SystemExit("Voice pack contains duplicate voice names")
    members: list[tuple[str, bytes, VoiceSource, int]] = []
    for source, path in sources:
        array = _voice_array(path, source, style_width)
        payload = _npy_bytes(array)
        members.append((f"{source.name}.npy", payload, source, array.shape[0]))

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for member_name, payload, _, _ in members:
            info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, payload)

    return [
        {
            "path": source.path,
            "size": path.stat().st_size,
            "sha256": sha256(path),
            "target_member": source.name,
            "shape": [rows, style_width],
        }
        for _, _, source, rows in members
        for path in [next(path for item, path in sources if item == source)]
    ]


def _derive_vocab(source_path: Path, target: Path, spec: dict[str, Any]) -> None:
    try:
        data = json.loads(source_path.read_text(encoding="utf-8"))
        vocabulary = data["model"]["vocab"]
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as exc:
        raise SystemExit(
            f"Cannot derive vocabulary from {source_path.name}: {exc}"
        ) from exc
    if not isinstance(vocabulary, dict) or not all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        for key, value in vocabulary.items()
    ):
        raise SystemExit(
            f"Tokenizer vocabulary in {source_path.name} is not a string-to-integer map"
        )
    target.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


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
        "layout": str(spec.get("runtime_layout", "single-onnx-v1")),
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
        f"- Voices: {len(runtime['voices'])} ({', '.join(runtime['voices'])})",
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

    assets = normalize_assets(spec.get("assets", []))
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
            if asset.transform is None and asset_matches(target, asset):
                print(f"Using existing {asset.name}")
                continue
            print(f"Downloading {asset.source}")
            staged.append((stage_asset(urls[asset.name], target, asset), target))
        for temporary, target in staged:
            temporary.replace(target)
    except BaseException:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise

    provenance: list[dict[str, Any]] = []
    voice_pack = spec.get("voice_pack")
    if voice_pack:
        voice_sources = _voice_source_list(spec, voice_pack)
        voice_staged: list[tuple[VoiceSource, Path]] = []
        try:
            for voice in voice_sources:
                source_path = out / ".sources" / voice.path.replace("/", "_")
                source_path.parent.mkdir(parents=True, exist_ok=True)
                voice_asset = MirrorAsset(
                    voice.path, voice.path, size=voice.size, sha256=voice.sha256
                )
                temporary = stage_asset(
                    huggingface_url(
                        spec["source_repository"], spec["source_revision"], voice.path
                    ),
                    source_path,
                    voice_asset,
                )
                temporary.replace(source_path)
                voice_staged.append((voice, source_path))
            target = out / str(voice_pack["target"])
            provenance = pack_voice_archive(
                voice_staged,
                target,
                style_width=int(voice_pack.get("style_width", 256)),
            )
        finally:
            shutil.rmtree(out / ".sources", ignore_errors=True)
        provenance_path = out / "source-assets.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "source": source,
                    "transform": "raw-float32-le-to-numpy-npz-v1",
                    "style_width": int(voice_pack.get("style_width", 256)),
                    "assets": provenance,
                    "output": {
                        "name": str(voice_pack["target"]),
                        "sha256": sha256(target),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    for asset in assets:
        if asset.transform == "vocabulary":
            source_path = out / f".{asset.name}.source"
            source_path.write_bytes((out / asset.name).read_bytes())
            _derive_vocab(source_path, out / asset.name, spec)
            source_path.unlink()

    manifest_assets: list[dict[str, Any]] = []
    for asset in assets:
        path = out / asset.name
        manifest_assets.append(
            {
                "name": asset.name,
                "role": asset.role,
                "format": asset.format,
                **({"quality": asset.quality} if asset.quality is not None else {}),
                **(
                    {"component": asset.component}
                    if asset.component is not None
                    else {}
                ),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if voice_pack:
        target = out / str(voice_pack["target"])
        manifest_assets.append(
            {
                "name": target.name,
                "role": "voices",
                "format": "numpy-npz",
                "size": target.stat().st_size,
                "sha256": sha256(target),
                "handling": {
                    "dtype": "float32",
                    "style_width": int(voice_pack.get("style_width", 256)),
                    "voice_count": len(provenance),
                    "members": [item["target_member"] for item in provenance],
                },
                "provenance": "source-assets.json",
            }
        )
        source_assets = out / "source-assets.json"
        manifest_assets.append(
            {
                "name": source_assets.name,
                "role": "metadata",
                "format": "json",
                "size": source_assets.stat().st_size,
                "sha256": sha256(source_assets),
            }
        )
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
    if voice_pack:
        manifest["transform"] = {
            "type": "voice-pack",
            "version": 1,
            "source_manifest": "source-assets.json",
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
