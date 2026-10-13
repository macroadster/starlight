#!/usr/bin/env python3
"""End-to-end §6.2-style eval through serve.sh (llama-cli is the only language engine).

chat:  eval/chat_prompts.jsonl, must_include_any / forbid_any on the reply.
look:  eval/stego_qa.jsonl (or --images DIR), payload must appear verbatim in the
       reply; clean images must not get a quoted payload.

    python3 datasets/opus_submission_2026/eval/run_eval.py
    python3 datasets/opus_submission_2026/eval/run_eval.py --no-lock   # grammar ablation
    python3 datasets/opus_submission_2026/eval/run_eval.py --val 30    # datasets/val sample
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
ROOT = SUB.parents[1]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def serve(prompt: str, image: str | None, no_lock: bool) -> str:
    cmd = [str(SUB / "serve.sh"), "-p", prompt]
    if image:
        cmd += ["--image", image]
    env = dict(os.environ, STARLIGHT_NO_LOCK="1" if no_lock else "0")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        return f"<serve.sh exit {proc.returncode}: {proc.stderr[-200:]}>"
    return proc.stdout.strip()


def rows(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def val_rows(n: int, seed: int) -> List[Dict]:
    val = ROOT / "datasets" / "val"
    seeds = {p.stem: p.read_text(encoding="utf-8").strip() for p in (val / "seeds").glob("*.md")}
    stego = sorted(p for p in (val / "stego").iterdir() if p.suffix.lower() in IMAGE_EXT)
    clean = sorted(p for p in (val / "clean").iterdir() if p.suffix.lower() in IMAGE_EXT)
    rng = random.Random(seed)
    out = []
    for p in rng.sample(stego, min(n, len(stego))):
        name = next((s for s in sorted(seeds, key=len, reverse=True) if p.stem.startswith(s + "_")), None)
        tech = json.loads(p.with_name(p.name + ".json").read_text())["embedding"]["technique"]
        out.append({"id": p.name, "image": str(p), "method": tech, "payload": seeds.get(name, ""), "clean": False})
    for p in rng.sample(clean, min(max(n // 3, 1), len(clean))):
        out.append({"id": p.name, "image": str(p), "method": "none", "payload": "", "clean": True})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-lock", action="store_true", help="disable the quote-lock grammar")
    ap.add_argument("--skip-chat", action="store_true")
    ap.add_argument("--val", type=int, default=0, help="also score N random datasets/val stego files")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json", default=None, help="write replies + verdicts here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    report: Dict[str, List[Dict]] = {"chat": [], "look": []}

    if not args.skip_chat:
        for rec in rows(HERE / "chat_prompts.jsonl"):
            reply = serve(rec["prompt"], None, args.no_lock)
            low = reply.lower()
            inc = rec.get("must_include_any") or []
            ok = (not inc or any(w.lower() in low for w in inc)) and not any(
                w.lower() in low for w in rec.get("forbid_any") or []
            )
            report["chat"].append({"id": rec["id"], "pass": ok, "reply": reply})
            if not args.quiet:
                print(f"[{'PASS' if ok else 'FAIL'}] chat/{rec['id']}: {reply[:160]!r}")

    looks = rows(HERE / "stego_qa.jsonl") + (val_rows(args.val, args.seed) if args.val else [])
    for rec in looks:
        img = rec["image"] if Path(rec["image"]).is_absolute() else str(ROOT / rec["image"])
        reply = serve("What secret is in this picture?", img, args.no_lock)
        if rec.get("clean"):
            low = reply.lower()
            ok = '"' not in reply and ("no secret" in low or "clean" in low)
        else:
            ok = bool(rec["payload"]) and rec["payload"] in reply
        report["look"].append({"id": rec["id"], "method": rec.get("method"), "clean": rec.get("clean"),
                               "pass": ok, "reply": reply})
        if not args.quiet:
            print(f"[{'PASS' if ok else 'FAIL'}] look/{rec['id']} ({rec.get('method')}): {reply[:140]!r}")

    chat_ok = sum(r["pass"] for r in report["chat"])
    stego = [r for r in report["look"] if not r["clean"]]
    clean = [r for r in report["look"] if r["clean"]]
    print(f"\nquote lock: {'off' if args.no_lock else 'on'}")
    if report["chat"]:
        print(f"chat probes:      {chat_ok}/{len(report['chat'])}")
    print(f"verbatim quotes:  {sum(r['pass'] for r in stego)}/{len(stego)}")
    print(f"clean honesty:    {sum(r['pass'] for r in clean)}/{len(clean)}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(r["pass"] for r in report["chat"] + report["look"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
