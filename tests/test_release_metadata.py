from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = ROOT / "scripts" / "prepare_release.py"
PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_release", PREPARE_PATH)
assert PREPARE_SPEC is not None and PREPARE_SPEC.loader is not None
prepare_release = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(prepare_release)


def test_catalog_target_repo() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    assert data["target_repository"] == "buchwandler/kokoro-onnx-models"
    assert data["releases"]["v1.0"]["tag"] == "model-files-v1.0"
    assert data["releases"]["v1.1-zh"]["tag"] == "model-files-v1.1"


def test_v1_0_voice_asset_is_numpy_archive() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    voices = next(
        asset
        for asset in data["releases"]["v1.0"]["assets"]
        if asset["name"] == "voices-v1.0.bin"
    )

    assert voices["format"] == "numpy-npz"


def test_swedish_and_thorsten_release_metadata() -> None:
    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    catalog = json.loads((ROOT / "catalog" / "releases.json").read_text())
    swedish = profiles["sv-joakim"]
    thorsten = profiles["de-thorsten"]
    assert swedish["license"] == "Apache-2.0"
    assert swedish["voices"]["items"]["Björn"] == "voices/Björn.pt"
    assert swedish["release"]["default_voice"] == "Alice"
    assert swedish["postprocess"]["q"] == 35
    assert thorsten["license"] == "Apache-2.0"
    assert thorsten["model"]["path"] == "model.pth"
    assert thorsten["voices"]["items"] == {"thorsten": "voices/thorsten.pt"}
    assert thorsten["release"]["default_voice"] == "thorsten"
    assert catalog["releases"]["sv-joakim"]["tag"] == "model-files-swedish-v1.0"
    assert (
        catalog["releases"]["de-thorsten"]["tag"] == "model-files-german-thorsten-v1.0"
    )


def test_thai_wayu_is_pinned_split_mirror() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["th-wayu"]
    assert spec["kind"] == "mirror"
    assert spec["source_type"] == "huggingface"
    assert spec["source_repository"] == "kunato/wayu-kokoro-thai-v1"
    assert spec["source_revision"] == "50d7f60e41ac118e5bb92b0ba52c30bb7830103c"
    assert spec["runtime_layout"] == "split-onnx-v1"
    assert len(spec["runtime"]["voices"]) == 12
    components = spec["onnx_contract"]["components"]
    assert set(components) == {"prosody", "curves", "decoder"}
    model_assets = [asset for asset in spec["assets"] if asset["role"] == "model"]
    assert {asset["component"] for asset in model_assets} == set(components)
    assert all(
        asset["size"] > 0 and len(asset["sha256"]) == 64 for asset in spec["assets"]
    )


def test_runtime_metadata_preserves_default_and_postprocess(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps({"speakers": [{"name": "Alice"}, {"name": "Björn"}]}),
        encoding="utf-8",
    )
    profile = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )["sv-joakim"]
    runtime = prepare_release._runtime_metadata(profile, bundle, profile["release"])
    assert runtime["default_voice"] == "Alice"
    assert runtime["voices"] == ["Alice", "Björn"]
    assert runtime["layout"] == "single-onnx-v1"
    assert runtime["postprocess"]["kind"] == "notch_filters"


def test_nabra_build_uses_prebuilt_onnx_source() -> None:
    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    profile = profiles["ar-nabra"]

    assert profile["repo_id"] == "marwanelamami/Nabra-82M-v0.1-ONNX"
    assert profile["model"]["kind"] == "onnx"
    assert profile["model"]["path"] == "nabra_fp32.onnx"
    assert "vocab.json" in profile["release"]["auxiliary_assets"][0]["source"]


def test_hebrew_not_published_by_default() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    assert data["releases"]["he-hebrew-nc"]["publish"] is False


def test_martin_is_mirrored_from_godelaune() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["v1.2-de-martin"]

    assert spec["kind"] == "mirror"
    assert spec["source_type"] == "huggingface"
    assert spec["source_repository"] == "Godelaune/Kokoro-82M-ONNX-German-Martin"
    assert spec["source_revision"] == "main"
    assert spec["tag"] == "model-files-german-martin-v1.2"

    by_name = {item["name"]: item for item in spec["assets"]}
    assert by_name["kokoro-german-martin-v1.2.onnx"]["source"] == "kokoro-martin.onnx"
    assert by_name["voices-german-martin-v1.2.bin"]["source"] == "voices-martin.npz"


def test_nabra_release_includes_vocabulary_metadata(tmp_path, monkeypatch) -> None:
    build_dir = tmp_path / "build" / "ar-nabra"
    build_dir.mkdir(parents=True)
    (build_dir / "model.onnx").write_bytes(b"model")
    np.savez(build_dir / "voices.npz", af_msa=np.zeros((510, 1, 256), dtype="<f4"))
    (build_dir / "voices.raw.bin").write_bytes(b"\x00" * (510 * 256 * 4))
    (build_dir / "bundle.json").write_text("{}\n", encoding="utf-8")
    (build_dir / "vocab.json").write_text('{"ʕ": 7, "ħ": 8}\n', encoding="utf-8")
    dist = tmp_path / "dist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_release.py",
            "ar-nabra",
            "--build-root",
            str(tmp_path / "build"),
            "--dist",
            str(dist),
        ],
    )

    assert prepare_release.main() == 0

    manifest_path = dist / "model-files-arabic-nabra-v0.1" / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocab = next(
        asset
        for asset in manifest["assets"]
        if asset["name"] == "vocab-arabic-nabra-v0.1.json"
    )
    assert vocab["role"] == "vocab"
    assert vocab["format"] == "json"
    assert (
        vocab["size"]
        == (dist / "model-files-arabic-nabra-v0.1" / vocab["name"]).stat().st_size
    )


def test_german_v1_1_and_holgern_are_retired() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "MODEL_LICENSES.md",
        ROOT / "catalog" / "releases.json",
        ROOT / "docs" / "LOCAL_PYKOKORO_PRE_RELEASE_TESTING.md",
        ROOT / "docs" / "PYKOKORO_MIGRATION.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "v1.1-de" not in text
    assert "holgern/kokoro-onnx-model" not in text

def test_release_catalog_excludes_upstream_only_russian_profiles() -> None:
    catalog = json.loads((ROOT / "catalog" / "releases.json").read_text(encoding="utf-8"))
    assert "ru-zaakirio-base" not in catalog["releases"]
    assert "ru-zaakirio-dima" not in catalog["releases"]
    assert catalog["releases"]["kk-anuarsv"]["publish"] is True
    assert catalog["releases"]["kk-anuarsv"]["language_codes"] == ["kk"]
