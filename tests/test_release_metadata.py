from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_target_repo() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    assert data["target_repository"] == "buchwandler/kokoro-onnx-models"
    assert data["releases"]["v1.0"]["tag"] == "model-files-v1.0"
    assert data["releases"]["v1.1-zh"]["tag"] == "model-files-v1.1"


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
