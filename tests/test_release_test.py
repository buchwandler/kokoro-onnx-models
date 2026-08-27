from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "test_release.py"
SPEC = importlib.util.spec_from_file_location("test_release", MODULE_PATH)
assert SPEC and SPEC.loader
release_test = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_test
SPEC.loader.exec_module(release_test)


def test_main_mirrors_and_verifies_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "release"
    commands: list[list[str]] = []
    monkeypatch.setattr(release_test, "run_command", commands.append)

    assert release_test.main(["v1.0", "--dist", str(dist)]) == 0

    assert commands == [
        [
            sys.executable,
            "scripts/mirror_release.py",
            "v1.0",
            "--dist",
            str(dist),
        ],
        [
            sys.executable,
            "scripts/verify_candidate.py",
            str(dist / "model-files-v1.0"),
            "--expected-tag",
            "model-files-v1.0",
            "--expected-profile",
            "v1.0",
        ],
    ]
