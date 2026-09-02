#!/usr/bin/env python3
"""Run the local pre-release pykokoro smoke test for vi-ngoc-huyen."""

import sys

from common import run_cli

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--allow-frontend-mismatch" not in args:
        args.append("--allow-frontend-mismatch")
    raise SystemExit(run_cli("vi-ngoc-huyen", args))
