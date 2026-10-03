#!/usr/bin/env python3
"""A/B gate: compare the AkinVox PyTorch reference path with its ONNX export.

Builds the `en-akinvox-cloning-v1` bundle from its pinned public sources and
writes a machine-readable parity report covering every component graph plus
full enrollment and target synthesis. Only public or repository-owned synthetic
references are used; no private recordings are read or written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import akinvox_cloning as akinvox

PROFILE_KEY = "en-akinvox-cloning-v1"
DEFAULT_BUILD_ROOT = ROOT / ".local-test" / "akinvox-build"
DEFAULT_OUTPUT_ROOT = ROOT / ".local-test" / "compare"


def _load_profile(profiles_path: Path, profile_key: str) -> dict:
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    try:
        return profiles[profile_key]
    except KeyError as exc:
        raise SystemExit(f"Unknown profile: {profile_key}") from exc


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=PROFILE_KEY)
    parser.add_argument(
        "--profiles", type=Path, default=ROOT / "scripts" / "kokoro_profiles.json"
    )
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--keep-build", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args(argv)

    profile = dict(_load_profile(args.profiles, args.profile))
    profile["export_validation"] = {
        **dict(profile.get("export_validation") or {}),
        "export_seed": args.seed,
    }
    out_dir = args.build_root / args.profile
    cache_dir = args.build_root / ".cache" / args.profile
    print(f"[{args.profile}] building ONNX-only cloning bundle", file=sys.stderr)
    akinvox.build_akinvox_profile(
        args.profile,
        profile,
        args.build_root,
        opset=args.opset,
        cache_dir=cache_dir,
        run_checker=not args.skip_check,
    )

    report_path = out_dir / akinvox.PARITY_REPORT_FILENAME
    if not report_path.is_file():
        raise SystemExit(f"Build produced no parity report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise SystemExit(f"PyTorch/ONNX parity failed; see {report_path}")

    output_dir = args.output_root / args.profile
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "profile": args.profile,
        "seed": args.seed,
        "status": report["status"],
        "components": sorted(report["components"]),
        "component_count": len(report["components"]),
        "seeded_cases": report["seeded_cases"],
        "reference_durations": [
            case["reference_seconds"] for case in report["end_to_end"]
        ],
        "cases": [
            {
                "name": case["name"],
                "reference_mask_exact": case["reference_mask_exact"],
                "synthesis": [
                    {
                        "name": item["name"],
                        "durations_exact": item["durations_exact"],
                        "output_length_exact": item["output_length_exact"],
                        "waveform_rms_ratio": item["waveform"]["rms_ratio"],
                    }
                    for item in case["synthesis"]
                ],
            }
            for case in report["end_to_end"]
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "parity-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.keep_build:
        shutil.rmtree(out_dir, ignore_errors=True)
    print(output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
