#!/usr/bin/env python3
"""
Next Tasks Reporter
Scan docs/plans and docs/coordination for dated tasks, order by date.
Usage:
  python3 scripts/next_tasks.py [--plans PATH] [--coordination PATH] [--limit N] [--json]
"""
from __future__ import annotations
import os
import sys
import glob
import re
import json
import datetime
import argparse
from pathlib import Path

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def extract_date_from_text(text: str):
    m = DATE_RE.search(text)
    if not m:
        return None
    dstr = m.group(0)
    try:
        return datetime.date.fromisoformat(dstr)
    except ValueError:
        return None


def title_from_content(text: str, fallback: str) -> str:
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # remove leading hashes
            t = line.lstrip('#').strip()
            if t:
                return t
    return fallback


def collect_md(path: Path, source_label: str):
    items = []
    if not path.exists() or not path.is_dir():
        return items
    for md in sorted(path.glob("*.md")):
        if not md.is_file():
            continue
        try:
            content = md.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            content = ''
        date = extract_date_from_text(content)
        if date is None:
            try:
                mtime = md.stat().st_mtime
                date = datetime.date.fromtimestamp(mtime)
            except Exception:
                date = None
        title = title_from_content(content, md.name)
        snippet_lines = content.splitlines()[:3]
        snippet = " ".join([l.strip() for l in snippet_lines if l.strip() != ''])
        items.append({
            'date': date.isoformat() if isinstance(date, datetime.date) else '',
            'source': source_label,
            'path': str(md),
            'title': title,
            'snippet': snippet,
        })
    return items


def main():
    parser = argparse.ArgumentParser(description='List upcoming tasks from docs/plans and docs/coordination ordered by date')
    parser.add_argument('--plans', default='docs/plans', help='Path to plans directory')
    parser.add_argument('--coordination', default='docs/coordination', help='Path to coordination directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of results')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()

    plan_path = Path(args.plans)
    coord_path = Path(args.coordination)

    results = []
    results.extend(collect_md(plan_path, 'plans'))
    results.extend(collect_md(coord_path, 'coordination'))

    # Normalize: empty date goes last
    def sort_key(item):
        d = item.get('date')
        if not d:
            return (datetime.date.max).isoformat()
        return d

    results.sort(key=sort_key)

    if args.limit is not None:
        results = results[:args.limit]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Human-friendly print
    w = 0
    for r in results:
        date = r['date'] or 'UNKNOWN'
        header = f"{date} | {r['source']:<13} | {r['title']}"
        path = f"{r['path']}"
        print(header)
        if r.get('snippet'):
            print(f"  {r['snippet']}")
        print(f"  {path}")
        print()
        w += 1
        if w > 100:
            break


if __name__ == '__main__':
    main()
