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
