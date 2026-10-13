#!/usr/bin/env python3
"""Benchmark look.py extraction on the repo's labeled corpora.

Stego recall: payload recovered and equal to the source seed text (when the seed
can be found from the file stem). Clean honesty: no payload on any clean cover.
Compares against the reference grok_submission_2026 extractor chain.

    python3 datasets/opus_submission_2026/eval/bench_look.py --limit 300
    python3 datasets/opus_submission_2026/eval/bench_look.py --detector --limit 40
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
ROOT = SUB.parents[1]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
TECH_FROM_NAME = {"alpha": "alpha", "palette": "palette", "lsb": "lsb.rgb", "exif": "exif", "eoi": "raw", "raw": "raw"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


def _norm(s: str) -> str:
    return " ".join(s.replace("\x00", "").split())


def corpora() -> List[Path]:
    out = [ROOT / "datasets" / "val"]
    out += sorted(p for p in (ROOT / "datasets").glob("*_submission_2025") if p.is_dir())
    out += sorted(p for p in (ROOT / "datasets").glob("*_submission_2026") if p.is_dir())
    return [p for p in out if (p / "stego").is_dir() or (p / "clean").is_dir()]


def seeds_for(corpus: Path) -> Dict[str, str]:
    seeds = {}
    for d in (corpus / "seeds", corpus):
        if d.is_dir():
            for f in d.glob("*.md"):
                if f.name.lower() not in {"readme.md"}:
                    seeds[f.stem] = f.read_text(encoding="utf-8", errors="ignore").strip()
    return seeds


def technique(img: Path) -> Optional[str]:
    side = img.with_name(img.name + ".json")
    if side.is_file():
        try:
            return json.loads(side.read_text())["embedding"]["technique"]
        except Exception:  # noqa: BLE001
            pass
    return None


def seed_text(img: Path, seeds: Dict[str, str]) -> Optional[str]:
    stem = img.stem
    best = None
    for name, text in seeds.items():
        if stem.startswith(name + "_") and (best is None or len(name) > len(best[0])):
            best = (name, text)
    return best[1] if best else None


_OPUS = _REF = None


def _init() -> None:
    global _OPUS, _REF
    _OPUS = _load("opus_look", SUB / "look.py")
    _REF = _load("grok_look", ROOT / "datasets" / "grok_submission_2026" / "look.py")


def _ref_payload(img: Path) -> Tuple[str, str]:
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        r = _REF._extract_any(img, None, "none")
    p = str(r.get("payload") or "").replace("\x00", "").strip()
    return (p if _REF._meaningful(p) else ""), r.get("method") or "none"


def _job(args: Tuple[str, str, Optional[str], Optional[str], bool]) -> Dict[str, Any]:
    path, kind, tech, expect, with_det = args
    img = Path(path)
    obs = _OPUS.observe(img, use_detector=with_det)
    rp, rm = _ref_payload(img)
    return {
        "path": path,
        "kind": kind,
        "tech": tech,
        "expect": expect,
        "opus": (obs["payload"], obs["method"], obs["status"], obs.get("scan")),
        "ref": (rp, rm),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=300, help="max stego and clean files per corpus")
    ap.add_argument("--detector", action="store_true", help="also run the CNN (slow)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--show", type=int, default=8, help="failures to print per corpus")
    ap.add_argument("--out", default=None, help="write per-file results JSON here")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    jobs = []
    for corpus in corpora():
        seeds = seeds_for(corpus)
        stego = [p for p in (corpus / "stego").glob("*") if p.suffix.lower() in IMAGE_EXT]
        clean = [p for p in (corpus / "clean").glob("*") if p.suffix.lower() in IMAGE_EXT]
        rng.shuffle(stego)
        rng.shuffle(clean)
        for p in stego[: args.limit]:
            jobs.append((str(p), "stego", technique(p), seed_text(p, seeds), args.detector))
        for p in clean[: args.limit]:
            jobs.append((str(p), "clean", None, None, args.detector))
    workers = 1 if args.detector else args.workers
    if workers == 1:
        _init()
        results = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, initializer=_init) as ex:
            results = list(ex.map(_job, jobs, chunksize=16))

    def corpus_of(path: str) -> str:
        return Path(path).parents[1].name

    table = collections.defaultdict(lambda: collections.Counter())
    fails = collections.defaultdict(list)
    for r in results:
        c = corpus_of(r["path"])
        t = table[c]
        if r["kind"] == "clean":
            t["clean"] += 1
            t["opus_fp"] += bool(r["opus"][0])
            t["ref_fp"] += bool(r["ref"][0])
            t["opus_flag"] += r["opus"][2] == "stego"
            if r["opus"][0] or r["opus"][2] == "stego":
                fails[c].append(("FP", r["path"], r["opus"][:3]))
            continue
        t["stego"] += 1
        exp = r["expect"]
        for who in ("opus", "ref"):
            got = r[who][0]
            t[f"{who}_decoded"] += bool(got)
            if exp is not None:
                t["with_seed"] += who == "opus"
                ok = bool(got) and _norm(exp) in _norm(got)
                t[f"{who}_exact"] += ok
                if who == "opus" and not ok:
                    fails[c].append(("MISS", r["path"], r["tech"], got[:60]))
            if r["tech"]:
                t[f"{who}_method"] += TECH_FROM_NAME.get(r[who][1], r[who][1]) == r["tech"] and bool(got)
        if r["opus"][2] == "stego":
            t["opus_flag"] += 0
    hdr = f"{'corpus':28} {'stego':>5} {'seed':>5} | {'opus exact':>10} {'ref exact':>9} | {'opus meth':>9} {'ref meth':>8} | {'clean':>5} {'opus FP':>7} {'ref FP':>6}"
    print(hdr)
    print("-" * len(hdr))
    tot = collections.Counter()
    for c, t in table.items():
        tot.update(t)
        print(
            f"{c:28} {t['stego']:5} {t['with_seed']:5} | {t['opus_exact']:10} {t['ref_exact']:9} | "
            f"{t['opus_method']:9} {t['ref_method']:8} | {t['clean']:5} {t['opus_fp']:7} {t['ref_fp']:6}"
        )
    t = tot
    print("-" * len(hdr))
    print(
        f"{'TOTAL':28} {t['stego']:5} {t['with_seed']:5} | {t['opus_exact']:10} {t['ref_exact']:9} | "
        f"{t['opus_method']:9} {t['ref_method']:8} | {t['clean']:5} {t['opus_fp']:7} {t['ref_fp']:6}"
    )
    if args.show:
        for c, fl in fails.items():
            print(f"\n{c}: {len(fl)} issues")
            for f in fl[: args.show]:
                print("  ", f)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
