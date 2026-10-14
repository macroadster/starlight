#!/usr/bin/env python3
"""Local gate for grok_a_submission_2026. Delegates to scripts/validate_submission_2026.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "validate_submission_2026.py"),
        "--submission",
        str(HERE),
        *sys.argv[1:],
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
