#!/usr/bin/env python3
"""Local validator for datasets/codex_submission_2026."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
validator = ROOT / "scripts" / "validate_submission_2026.py"

if not validator.is_file():
    raise SystemExit(f"Repo validator not found at {validator}")

import subprocess

cmd = [sys.executable, str(validator), "--submission", str(HERE)] + sys.argv[1:]
raise SystemExit(subprocess.call(cmd))
