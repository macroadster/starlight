#!/usr/bin/env python3
"""
Sign-off script for Project Starlight: status-first sign-off workflow.

Behavior:
- Read docs/status.md to determine if all tasks are complete.
- If complete, archive docs/status.md to docs/archive/YYYY-MM-DD.md.
- Regenerate docs/status.md from docs/plans/ and docs/coordination/ using next_tasks.py.
- Append a Sign-off Log entry to the archived file describing the date and scope of the sign-off.
- --dry-run shows what would happen without mutating files.
"""
from __future__ import annotations
import re
import datetime
import subprocess
import json
import sys
from pathlib import Path

STATUS_FILE = Path('docs/status.md')
ARCHIVE_DIR = Path('docs/archive')

STATUS_RE = re.compile(r"\(Status:\s*([^)]*)\)", re.IGNORECASE)


def statusmd_complete() -> tuple[bool, int, int]:
    """Return (complete, total_items, completed_items) based on docs/status.md."""
    if not STATUS_FILE.exists():
        return (False, 0, 0)
    text = STATUS_FILE.read_text(encoding='utf-8', errors='ignore')
    matches = STATUS_RE.findall(text)
    total = len(matches)
    completed = 0
    for val in matches:
        if not val:
            continue
        lower = val.strip().lower()
        if any(k in lower for k in ("completed", "done", "finish")):
            completed += 1
    return (total > 0 and completed == total, total, completed)


def append_signoff_log(archived_path: str, date_str: str, total: int, completed: int) -> None:
    header = f"### Sign-off — {date_str}\n"
    payload = f"- Completed {completed}/{total} tasks across plans and coordination.\n"
    footer = "- Sign-off event recorded by signoff.py\n"
    content_to_append = f"\n{header}{payload}{footer}"
    archived_file = Path(archived_path)
    if archived_file.exists():
        archived_file.write_text(archived_file.read_text(encoding='utf-8') + content_to_append, encoding='utf-8')


def regenerate_status_md() -> None:
    """Regenerate docs/status.md from docs/plans/ and docs/coordination/."""
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/next_tasks.py', '--json'],
            capture_output=True, text=True, check=True
        )
        tasks = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        tasks = []

    content = "# AI Consensus Documentation\n\n## Tasks\n\n"
    for task in tasks:
        status = "(Status: pending)"
        title = task.get('title', 'Unknown Task')
        date = task.get('date', 'Unknown Date')
        source = task.get('source', 'unknown')
        path = task.get('path', '')
        content += f"- {title} ({date}, {source}) {status}\n"
        if path:
            content += f"  - Path: {path}\n"
        content += "\n"

    STATUS_FILE.write_text(content, encoding='utf-8')


def archive_status_file() -> str | None:
    if not STATUS_FILE.exists():
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    dest = ARCHIVE_DIR / f'{today}.md'
    i = 1
    while dest.exists():
        dest = ARCHIVE_DIR / f'{today}_{i}.md'
        i += 1
    STATUS_FILE.rename(dest)
    return str(dest)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Sign off on status.md when all tasks are complete.')
    ap.add_argument('--dry-run', action='store_true', help='Preview actions without applying changes')
    ap.add_argument('--verbose', action='store_true', help='Verbose output')
    args = ap.parse_args()

    complete, total, completed = statusmd_complete()
    if not complete:
        print('Not all tasks are completed according to docs/status.md. Sign-off not triggered.')
        if args.verbose:
            print(f"Total items: {total}, Completed: {completed}")
        return 0

    date_str = datetime.date.today().isoformat()

    if args.dry_run:
        print(f'DRY RUN: would sign off on {date_str} with {completed}/{total} completed.')
        print('Would archive docs/status.md to docs/archive/YYYY-MM-DD.md.')
        print('Would regenerate docs/status.md from docs/plans/ and docs/coordination/.')
        print('Would append sign-off log to the archived file.')
        return 0

    # Archive current status.md
    archived = archive_status_file()
    if not archived:
        print('No docs/status.md found to archive.')
        return 1

    # Regenerate status.md
    regenerate_status_md()

    # Append sign-off log to archived file
    append_signoff_log(archived, date_str, total, completed)

    if args.verbose:
        print(f'Archived docs/status.md to {archived}')
        print(f'Regenerated docs/status.md')
        print(f'Appended sign-off log to {archived}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
