#!/usr/bin/env python3
"""Local gate for opus_submission_2026: repo validator, then the end-to-end reply eval."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    gate = [sys.executable, str(ROOT / "scripts" / "validate_submission_2026.py"), "--submission", str(HERE)]
    rc = subprocess.call(gate + sys.argv[1:], cwd=str(ROOT))
    if rc != 0 or "--skip-llama" in sys.argv:
        return rc
    return subprocess.call([sys.executable, str(HERE / "eval" / "run_eval.py")], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
