#!/usr/bin/env python3
"""Run the local pre-release smoke test for the oddadmix English 7M release."""

from common import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli("en-oddadmix-7m-distill"))
