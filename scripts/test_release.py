#!/usr/bin/env python3
"""Build or mirror a release and verify it locally without publishing."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "releases.json"
DEFAULT_DIST = Path(".local-test") / "release"


def load_release(release_key: str) -> dict[str, Any]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    try:
        return catalog["releases"][release_key]
    except KeyError as exc:
        raise SystemExit(f"Unknown release key: {release_key}") from exc


def run_command(command: list[str]) -> None:
    print(f"$ {shlex.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or mirror a release and verify it locally"
    )
    parser.add_argument("release_key", help="Entry from catalog/releases.json")
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--allow-restricted", action="store_true")
    args = parser.parse_args(argv)

    spec = load_release(args.release_key)
    publish = bool(spec.get("publish", True))
    if not publish and not args.allow_restricted:
        parser.error("Release is disabled; pass --allow-restricted after reviewing it")

    profile = str(spec.get("profile", args.release_key))
    tag = str(spec["tag"])
    dist = args.dist
    if spec["kind"] == "mirror":
        run_command(
            [
                sys.executable,
                "scripts/mirror_release.py",
                args.release_key,
                "--dist",
                str(dist),
            ]
        )
    elif spec["kind"] == "build":
        build_root = dist.parent / f".{dist.name}.build"
        run_command(
            [
                sys.executable,
                "scripts/build_kokoro.py",
                "build",
                profile,
                "--out",
                str(build_root),
            ]
        )
        prepare = [
            sys.executable,
            "scripts/prepare_release.py",
            profile,
            "--build-root",
            str(build_root),
            "--dist",
            str(dist),
        ]
        if args.allow_restricted:
            prepare.append("--allow-restricted")
        run_command(prepare)
    else:
        parser.error(f"Unsupported release kind: {spec['kind']}")

    verify = [
        sys.executable,
        "scripts/verify_candidate.py",
        str(dist / tag),
        "--expected-tag",
        tag,
        "--expected-profile",
        profile,
    ]
    if args.allow_restricted:
        verify.append("--allow-restricted")
    run_command(verify)
    print(f"Verified local release candidate: {dist / tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
