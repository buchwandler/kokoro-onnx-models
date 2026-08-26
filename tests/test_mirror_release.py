from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mirror_release.py"
spec = importlib.util.spec_from_file_location("mirror_release", MODULE_PATH)
assert spec and spec.loader
mirror_release = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mirror_release
spec.loader.exec_module(mirror_release)


def asset_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_normalize_assets_supports_legacy_and_mapped_entries() -> None:
    assets = mirror_release.normalize_assets(
        [
            "foo.bin",
            {"source": "upstream.npz", "name": "release.bin", "size": 3, "sha256": "abc"},
        ]
    )

    assert assets == [
        mirror_release.MirrorAsset(source="foo.bin", name="foo.bin"),
        mirror_release.MirrorAsset(
            source="upstream.npz", name="release.bin", size=3, sha256="abc"
        ),
    ]


def test_huggingface_sources_use_pinned_revision_and_source_paths() -> None:
    assets = [mirror_release.MirrorAsset("voices/test.npz", "test.bin")]
    urls, source = mirror_release.huggingface_sources(
        {
            "source_repository": "source/repo",
            "source_revision": "a1b2c3",
        },
        assets,
    )

    assert urls["test.bin"] == (
        "https://huggingface.co/source/repo/resolve/"
        "a1b2c3/voices/test.npz?download=true"
    )
    assert source == {
        "type": "huggingface",
        "repository": "source/repo",
        "revision": "a1b2c3",
    }


def test_main_preserves_bytes_when_mapping_source_to_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = b"model bytes"
    voices = b"npz bytes"
    catalog = {
        "target_repository": "buchwandler/kokoro-onnx-models",
        "releases": {
            "martin": {
                "kind": "mirror",
                "source_type": "huggingface",
                "source_repository": "Godelaune/model",
                "source_revision": "revision",
                "tag": "model-files-martin",
                "license": "Apache-2.0",
                "assets": [
                    {
                        "source": "model.onnx",
                        "name": "release.onnx",
                        "size": len(model),
                        "sha256": asset_hash(model),
                    },
                    {
                        "source": "voices.npz",
                        "name": "release.bin",
                        "size": len(voices),
                        "sha256": asset_hash(voices),
                    },
                ],
            }
        },
    }
    catalog_path = tmp_path / "releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(mirror_release, "CATALOG", catalog_path)
    payloads = {"model.onnx": model, "voices.npz": voices}

    def fake_download(url: str, path: Path) -> None:
        source_name = url.rsplit("/", 1)[-1].split("?", 1)[0]
        path.write_bytes(payloads[source_name])

    monkeypatch.setattr(mirror_release, "download", fake_download)
    monkeypatch.setattr(
        "sys.argv", ["mirror_release.py", "martin", "--dist", str(tmp_path / "dist")]
    )

    assert mirror_release.main() == 0
    output = tmp_path / "dist" / "model-files-martin"
    assert (output / "release.onnx").read_bytes() == model
    assert (output / "release.bin").read_bytes() == voices
    manifest = json.loads((output / "release-manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == {
        "type": "huggingface",
        "repository": "Godelaune/model",
        "revision": "revision",
    }


def test_mismatch_is_rejected_before_any_asset_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = b"good"
    bad = b"bad"
    catalog = {
        "target_repository": "target/repo",
        "releases": {
            "test": {
                "kind": "mirror",
                "source_type": "huggingface",
                "source_repository": "source/repo",
                "source_revision": "revision",
                "tag": "tag",
                "license": "Apache-2.0",
                "assets": [
                    {"source": "good", "name": "good.bin", "size": len(good), "sha256": asset_hash(good)},
                    {"source": "bad", "name": "bad.bin", "size": 99, "sha256": asset_hash(bad)},
                ],
            }
        },
    }
    catalog_path = tmp_path / "releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(mirror_release, "CATALOG", catalog_path)

    def fake_download(url: str, path: Path) -> None:
        path.write_bytes(good if url.endswith("/good?download=true") else bad)

    monkeypatch.setattr(mirror_release, "download", fake_download)
    monkeypatch.setattr("sys.argv", ["mirror_release.py", "test", "--dist", str(tmp_path / "dist")])

    with pytest.raises(SystemExit, match="Size mismatch for bad.bin"):
        mirror_release.main()

    output = tmp_path / "dist" / "tag"
    assert not (output / "good.bin").exists()
    assert not (output / "bad.bin").exists()
    assert not list(output.glob("*.part"))
