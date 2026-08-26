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


def test_martin_uses_godelaune_huggingface_source() -> None:
    data = json.loads((ROOT / "catalog" / "releases.json").read_text())
    spec = data["releases"]["v1.2-de-martin"]

    assert spec["kind"] == "mirror"
    assert spec["source_type"] == "huggingface"
    assert spec["source_repository"] == "Godelaune/Kokoro-82M-ONNX-German-Martin"
    assert spec["source_revision"] == "a1cba7fbf0e72fbae38f0a3a48ce0dc8e6077804"
    assert spec["tag"] == "model-files-german-martin-v1.2"

    by_name = {item["name"]: item for item in spec["assets"]}
    assert by_name["kokoro-german-martin-v1.2.onnx"]["source"] == "kokoro-martin.onnx"
    assert by_name["kokoro-german-martin-v1.2.onnx"]["size"] == 325512630
    assert by_name["kokoro-german-martin-v1.2.onnx"]["sha256"] == (
        "c302f1d8bc7adf40a842cb550e18c39a5026bdb1afdd29dbb700b501cb49276b"
    )
    assert by_name["voices-german-martin-v1.2.bin"]["source"] == "voices-martin.npz"
    assert by_name["voices-german-martin-v1.2.bin"]["size"] == 522506
    assert by_name["voices-german-martin-v1.2.bin"]["sha256"] == (
        "5b9c8553398d7abf67498ce500c186cefaa7b68fed3e3d415da5380670105acd"
    )
