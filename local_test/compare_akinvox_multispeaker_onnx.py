#!/usr/bin/env python3
"""Build or inspect AKinVox multispeaker ONNX parity evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_kokoro

PROFILE_KEY = "akinvox-multispeaker-v1"
DEFAULT_BUILD_ROOT = ROOT / ".local-test" / "akinvox-build"
DEFAULT_OUTPUT_ROOT = ROOT / ".local-test" / "compare"
LANGUAGES = {
    "en": ("english-", ["AV_ariala", "AV_brody"]),
    "de": ("german-", ["AV_thorsten", "AV_bernd"]),
}


def _load_bundle(bundle_path: Path) -> dict[str, Any]:
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read build bundle {bundle_path}: {exc}") from exc
    if not isinstance(bundle, dict):
        raise SystemExit(f"Build bundle is not an object: {bundle_path}")
    return bundle


def _parity_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    if (
        bundle.get("profile") != PROFILE_KEY
        or bundle.get("model_version") != "2026.10.07"
    ):
        raise SystemExit(
            "Bundle profile or model version does not match the pinned target"
        )
    source = bundle.get("source", {})
    if source.get("revision") != "2b3fa45b39256382872823562da36d371a808e5a":
        raise SystemExit("Bundle source revision is not the pinned immutable commit")
    if bundle.get("onnx_checker", {}).get("status") != "passed":
        raise SystemExit("Bundle does not record a passing mandatory ONNX checker")
    components = bundle.get("components", {})
    if set(components) != {"model_en", "model_de"}:
        raise SystemExit(f"Unexpected ONNX component set: {sorted(components)}")

    languages: dict[str, Any] = {}
    for language, (prefix, expected_voices) in LANGUAGES.items():
        component_name = f"model_{language}"
        component = components[component_name]
        if component.get("onnx_checker") != "passed":
            raise SystemExit(f"{component_name} did not pass onnx.checker")
        contract = component.get("contract_validation", {})
        if contract.get("outputs") != {"audio": "float32", "duration": "int64"}:
            raise SystemExit(f"{component_name} has the wrong output contract")
        parity = component.get("parity", {})
        cases = parity.get("cases", [])
        if parity.get("status") != "pass" or len(cases) != 5:
            raise SystemExit(f"{component_name} is missing its five parity cases")
        if any(not str(case.get("fixture", "")).startswith(prefix) for case in cases):
            raise SystemExit(
                f"{component_name} parity includes a fixture for another language"
            )
        if [case.get("voice") for case in cases] != [
            expected_voices[0],
            expected_voices[1],
            expected_voices[0],
            expected_voices[1],
            expected_voices[0],
        ]:
            raise SystemExit(
                f"{component_name} parity does not cover the expected voices"
            )
        if [case.get("speed") for case in cases][-1] == 1.0:
            raise SystemExit(
                f"{component_name} parity does not include a non-default speed"
            )
        languages[language] = {
            "component": component_name,
            "status": parity["status"],
            "case_count": len(cases),
            "fixtures": [case["fixture"] for case in cases],
            "voices": [case["voice"] for case in cases],
            "speeds": [case["speed"] for case in cases],
            "onnx_checker": component["onnx_checker"],
            "contract_validation": contract,
        }

    audits = bundle.get("adapter_load_audit", {})
    if set(audits) != {"en", "de"}:
        raise SystemExit("Bundle is missing one or more adapter audits")
    for language, audit in audits.items():
        if (
            audit.get("target_count") != 201
            or audit.get("tensor_count") != 403
            or audit.get("consumed_tensors") != 403
            or audit.get("all_adapter_tensors_consumed") is not True
        ):
            raise SystemExit(
                f"{language} adapter audit does not account for every tensor"
            )

    forbidden_suffixes = {".pt", ".pth", ".py", ".whl"}
    forbidden = [
        name
        for name in bundle.get("files", {})
        if Path(name).suffix.lower() in forbidden_suffixes
    ]
    if forbidden:
        raise SystemExit(f"Build bundle contains forbidden runtime assets: {forbidden}")
    return {
        "profile": PROFILE_KEY,
        "model_version": "2026.10.07",
        "source_revision": source["revision"],
        "status": "pass",
        "component_count": len(components),
        "languages": languages,
        "adapter_audit": {
            language: {
                "target_count": audit["target_count"],
                "tensor_count": audit["tensor_count"],
                "consumed_tensors": audit["consumed_tensors"],
                "all_adapter_tensors_consumed": audit["all_adapter_tensors_consumed"],
            }
            for language, audit in audits.items()
        },
        "forbidden_runtime_assets": [],
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles", type=Path, default=ROOT / "scripts/kokoro_profiles.json"
    )
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--no-build", action="store_true", help="Inspect an existing build bundle"
    )
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args(argv)

    profiles = build_kokoro.load_profiles(args.profiles)
    try:
        profile = dict(profiles[PROFILE_KEY])
    except KeyError as exc:
        raise SystemExit(f"Missing build profile {PROFILE_KEY!r}") from exc
    profile["export_validation"] = {
        **dict(profile.get("export_validation") or {}),
        "export_seed": args.seed,
    }
    if not args.no_build:
        print(
            f"[{PROFILE_KEY}] building both language graphs and running parity",
            file=sys.stderr,
        )
        build_kokoro.build_profile(
            PROFILE_KEY,
            profile,
            args.build_root,
            opset=args.opset,
            seq_len=64,
            run_checker=True,
        )

    out_dir = args.build_root / PROFILE_KEY
    summary = _parity_summary(_load_bundle(out_dir / "bundle.json"))
    summary["seed"] = args.seed
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / PROFILE_KEY / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.no_build and not args.keep_build:
        shutil.rmtree(out_dir, ignore_errors=True)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
