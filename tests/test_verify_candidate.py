from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_candidate", ROOT / "scripts" / "verify_candidate.py"
)
assert SPEC and SPEC.loader
verify_candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_candidate)


def _write_candidate(tmp_path: Path, *, enabled: bool = True) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.onnx").write_bytes(b"model")
    np.savez(candidate / "voices.npz", af=np.zeros((1, 1), dtype=np.float32))
    (candidate / "bundle.json").write_text(
        '{"speakers": [{"name": "af"}]}\n', encoding="utf-8"
    )
    assets = []
    for name, role, fmt, quality in (
        ("model.onnx", "model", "onnx", "fp32"),
        ("voices.npz", "voices", "numpy-npz", None),
        ("bundle.json", "bundle", "json", None),
    ):
        path = candidate / name
        asset = {
            "name": name,
            "role": role,
            "format": fmt,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if quality:
            asset["quality"] = quality
        assets.append(asset)
    manifest = {
        "schema": 2,
        "runtime_contract": 1,
        "repository": "buchwandler/kokoro-onnx-models",
        "tag": "model-files-test",
        "profile": "test",
        "model_version": "1.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "source": {"type": "test", "repository": "source/repo", "revision": "rev"},
        "license": "Apache-2.0",
        "publication": {"enabled": enabled},
        "runtime": {
            "language_codes": ["en"],
            "sample_rate": 24000,
            "frontend": "pykokoro-native-v1",
            "frontend_experimental": False,
            "max_tokens": 510,
            "default_voice": "af",
            "voices": ["af"],
        },
        "onnx_contract": {
            "inputs": {"tokens": "int64"},
            "outputs": {"audio": "float32"},
            "max_tokens": 510,
        },
        "assets": assets,
    }
    (candidate / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (candidate / "SHA256SUMS").write_text(
        "\n".join(f"{asset['sha256']}  {asset['name']}" for asset in assets) + "\n",
        encoding="utf-8",
    )
    return candidate


def test_verify_candidate_checks_manifest_assets_and_checksums(tmp_path: Path) -> None:
    result = verify_candidate.verify_candidate(_write_candidate(tmp_path))
    assert result["asset_count"] == 3


def test_verify_candidate_rejects_size_mismatch(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"][0]["size"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(verify_candidate.CandidateError, match="Size mismatch"):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_unmanifested_file(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    (candidate / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(
        verify_candidate.CandidateError, match="Unexpected candidate files"
    ):
        verify_candidate.verify_candidate(candidate)


def test_verify_candidate_rejects_disabled_publication_by_default(
    tmp_path: Path,
) -> None:
    candidate = _write_candidate(tmp_path, enabled=False)
    with pytest.raises(
        verify_candidate.CandidateError, match="Publication is disabled"
    ):
        verify_candidate.verify_candidate(candidate)
    verify_candidate.verify_candidate(candidate, allow_restricted=True)


def test_verify_candidate_rejects_duplicate_quality(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path)
    extra = candidate / "model2.onnx"
    extra.write_bytes(b"second")
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"].append(
        {
            "name": extra.name,
            "role": "model",
            "quality": "fp32",
            "format": "onnx",
            "size": extra.stat().st_size,
            "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        verify_candidate.CandidateError, match="Duplicate asset role/format slot"
    ):
        verify_candidate.verify_candidate(candidate)
