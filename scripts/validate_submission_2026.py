#!/usr/bin/env python3
"""Validate a 2026 Starlight Chat submission against COMPETITION_2026.md."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "README.md",
    "model/model_card.md",
    "model/method_config.json",
    "model/system_prompt.txt",
    "model/serve.json",
    "eval/chat_prompts.jsonl",
    "eval/stego_qa.jsonl",
]
TRACK_B_FILES = ["look.py", "serve.sh", "fuse_vlm.py"]
# v1.1: the observation is a hint. A payload field means the secret was
# handed to the model, so the block is not a legal scored prompt.
OBS_RE = re.compile(
    r"<starlight_observation>\s*"
    r"status:\s*(stego|clean|unknown)\s*"
    r"method:\s*(alpha|palette|lsb\.rgb|exif|raw|none)\s*"
    r"confidence:\s*([0-9]*\.?[0-9]+)\s*"
    r"bit_order:\s*(lsb-first|msb-first|none)\s*"
    r"detector:\s*(\S+)\s*"
    r"(?:notes:\s*(.*?)\s*)?"
    r"</starlight_observation>",
    re.DOTALL,
)
PAYLOAD_FIELD_RE = re.compile(r'payload:\s*"""')
SIZE_CAP = 2 * 1024**3
LIGHT_CAP = 800 * 1024**2


def _ok(name: str, ok: bool, detail: str) -> Dict[str, Any]:
    return {"gate": name, "pass": bool(ok), "detail": detail}


def load_serve(submission: Path) -> Dict[str, Any]:
    path = submission / "model" / "serve.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_gguf(submission: Path, serve: Dict[str, Any]) -> Optional[Path]:
    raw = serve.get("llm_gguf") or ""
    if not raw:
        return None
    candidates = [
        Path(raw),
        submission / raw,
        ROOT / raw,
        submission / "model" / Path(raw).name,
    ]
    env = os.environ.get("STARLIGHT_LLM_GGUF")
    if env:
        candidates.insert(0, Path(env))
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def check_layout(submission: Path, serve: Dict[str, Any]) -> Dict[str, Any]:
    missing = [rel for rel in REQUIRED_FILES if not (submission / rel).is_file()]
    track = str(serve.get("track", "B")).upper()
    if track in {"B", "C"}:
        missing.extend(rel for rel in TRACK_B_FILES if not (submission / rel).is_file())
    return _ok("layout", not missing, "ok" if not missing else f"missing: {missing}")


def check_serve_json(serve: Dict[str, Any]) -> Dict[str, Any]:
    required = ["track", "detector", "arch", "system_prompt"]
    missing = [k for k in required if not serve.get(k)]
    track = str(serve.get("track", "")).upper()
    if track not in {"A", "B", "C"}:
        missing.append("track must be A|B|C")
    return _ok("serve.json", not missing, "ok" if not missing else f"invalid: {missing}")


def check_look_schema(submission: Path, fixture: Optional[Path]) -> Dict[str, Any]:
    look = submission / "look.py"
    if not look.is_file():
        return _ok("look_schema", False, "look.py missing")
    if fixture is None or not fixture.is_file():
        return _ok("look_schema", False, "no fixture image to probe")
    cmd = [sys.executable, str(look), str(fixture), "--block"]
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return _ok("look_schema", False, "look.py timed out")
    text = proc.stdout or ""
    if proc.returncode != 0:
        return _ok(
            "look_schema",
            False,
            f"exit {proc.returncode}: {(proc.stderr or '')[-300:]}",
        )
    if PAYLOAD_FIELD_RE.search(text):
        return _ok(
            "look_schema",
            False,
            "payload field in the observation block (v1.1: the prompt must not carry the secret)",
        )
    match = OBS_RE.search(text)
    return _ok(
        "look_schema",
        bool(match),
        "matched Look Protocol" if match else f"no observation block in: {text[:240]!r}",
    )


def check_llama_load(gguf: Optional[Path]) -> Dict[str, Any]:
    if gguf is None:
        return _ok(
            "serve",
            False,
            "LLM GGUF not on disk (set STARLIGHT_LLM_GGUF or run fuse_vlm.py)",
        )
    llama = shutil.which("llama-cli")
    if not llama:
        return _ok("serve", False, "llama-cli not on PATH")
    size = gguf.stat().st_size
    if size > SIZE_CAP:
        return _ok("serve", False, f"{gguf.name} is {size} bytes > 2.0 GiB cap")
    cmd = [
        llama,
        "-m",
        str(gguf),
        "-p",
        "Hi",
        "-n",
        "16",
        "--no-display-prompt",
        "--no-conversation",
        "--single-turn",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return _ok("serve", False, "llama-cli timed out")
    out = (proc.stdout or "") + (proc.stderr or "")
    loaded = proc.returncode == 0 and (
        "error loading" not in out.lower() and "failed to load" not in out.lower()
    )
    detail = f"{gguf.name} ({size / 1024**2:.1f} MiB), exit={proc.returncode}"
    if not loaded:
        detail += f" tail={out[-240:]!r}"
    return _ok("serve", loaded, detail)


def first_fixture(submission: Path) -> Optional[Path]:
    fixtures = submission / "eval" / "fixtures"
    if fixtures.is_dir():
        images = [
            p
            for p in sorted(fixtures.iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
        ]
        stego = [p for p in images if not p.name.startswith("clean_")]
        if stego:
            return stego[0]
        if images:
            return images[0]
    qa_path = submission / "eval" / "stego_qa.jsonl"
    if qa_path.is_file():
        for line in qa_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            img = rec.get("image")
            if img:
                cand = (ROOT / img) if not Path(img).is_absolute() else Path(img)
                if cand.is_file():
                    return cand
    return None


def check_prompts(submission: Path) -> Dict[str, Any]:
    path = submission / "eval" / "chat_prompts.jsonl"
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
            n += 1
    return _ok("chat_prompts", n >= 8, f"{n} prompts (need >= 8)")


def check_stego_qa(submission: Path) -> Dict[str, Any]:
    path = submission / "eval" / "stego_qa.jsonl"
    methods = set()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        if rec.get("method"):
            methods.add(rec["method"])
    need = {"alpha", "palette", "lsb.rgb", "exif", "raw"}
    missing = sorted(need - methods)
    return _ok(
        "stego_qa",
        n >= 5 and not missing,
        f"{n} items, methods={sorted(methods)}"
        + (f", missing {missing}" if missing else ""),
    )


def _secret_fragments(payload: str) -> List[str]:
    """Contiguous slices a prompt must not contain. Short secrets use the whole line."""
    text = payload.strip()
    if not text:
        return []
    if len(text) <= 12:
        return [text]
    return [text[i : i + 8] for i in range(0, len(text) - 7, 8)]


def _leaks(haystack: str, payload: str) -> Optional[str]:
    for frag in _secret_fragments(payload):
        if frag and frag in haystack:
            return frag
    return None


def check_look_leak(submission: Path) -> Dict[str, Any]:
    """v1.1: look.py may hint status and method. It must not reveal the secret."""
    look = submission / "look.py"
    qa_path = submission / "eval" / "stego_qa.jsonl"
    if not look.is_file() or not qa_path.is_file():
        return _ok("look_leak", False, "look.py or stego_qa.jsonl missing")
    rows = []
    for line in qa_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    tested = 0
    failed: List[str] = []
    for rec in rows:
        rel = rec.get("image")
        if not rel:
            continue
        img = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        if not img.is_file():
            failed.append(f"{rec.get('id')}: missing {rel}")
            continue
        outputs: List[str] = []
        timed_out = False
        for flag in ("--json", "--block"):
            try:
                proc = subprocess.run(
                    [sys.executable, str(look), str(img), flag],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                failed.append(f"{rec.get('id')}: look {flag} timeout")
                timed_out = True
                break
            if proc.returncode != 0:
                failed.append(f"{rec.get('id')}: look {flag} exit {proc.returncode}")
                timed_out = True
                break
            outputs.append(proc.stdout or "")
        if timed_out:
            continue
        json_text, block_text = outputs
        try:
            obs = json.loads(json_text)
        except json.JSONDecodeError:
            failed.append(f"{rec.get('id')}: look not JSON")
            continue
        tested += 1
        payload = rec.get("payload") or ""
        clean = bool(rec.get("clean"))
        status = obs.get("status")
        if "payload" in obs:
            failed.append(f"{rec.get('id')}: JSON still has a payload field")
        if PAYLOAD_FIELD_RE.search(block_text):
            failed.append(f"{rec.get('id')}: block still has a payload field")
        if clean:
            if status == "stego":
                failed.append(f"{rec.get('id')}: clean image reported as stego")
            continue
        leaked = _leaks(json_text, payload) or _leaks(block_text, payload)
        if leaked:
            failed.append(f"{rec.get('id')}: prompt leaks secret fragment {leaked!r}")
    if tested == 0:
        return _ok("look_leak", False, "no on-disk fixtures to score")
    return _ok(
        "look_leak",
        not failed,
        f"{tested} fixtures told the model nothing" if not failed else f"{tested} tested; {failed}",
    )


def check_detector(serve: Dict[str, Any]) -> Dict[str, Any]:
    det = serve.get("detector") or ""
    det_gguf = serve.get("detector_gguf") or ""
    paths = [ROOT / det, ROOT / det_gguf]
    found = [str(p) for p in paths if p.is_file()]
    named = "detector_balanced.pth" in det or "starlight.gguf" in (det + det_gguf)
    return _ok(
        "detector",
        bool(found) and named,
        f"found={found}" if found else f"missing detector ({det!r}, {det_gguf!r})",
    )


def lightweight_note(gguf: Optional[Path], required: bool) -> Dict[str, Any]:
    if gguf is None or not gguf.is_file():
        return _ok(
            "lightweight",
            not required,
            "no GGUF to measure"
            + ("" if required else " (skipped; run fuse_vlm.py to score size)"),
        )
    size = gguf.stat().st_size
    if size <= LIGHT_CAP:
        pts = 10
    elif size >= SIZE_CAP:
        pts = 0
    else:
        pts = round(10 * (SIZE_CAP - size) / (SIZE_CAP - LIGHT_CAP), 1)
    return _ok(
        "lightweight",
        size <= SIZE_CAP,
        f"{size / 1024**2:.1f} MiB → {pts}/10 lightweight points",
    )


def run(submission: Path, skip_llama: bool) -> Tuple[List[Dict[str, Any]], bool]:
    serve = load_serve(submission)
    gguf = resolve_gguf(submission, serve)
    fixture = first_fixture(submission)
    gates = [
        check_layout(submission, serve),
        check_serve_json(serve),
        check_prompts(submission),
        check_stego_qa(submission),
        check_detector(serve),
        check_look_schema(submission, fixture),
        check_look_leak(submission),
        lightweight_note(gguf, required=not skip_llama),
    ]
    if skip_llama:
        gates.append(_ok("serve", True, "skipped (--skip-llama)"))
    else:
        gates.append(check_llama_load(gguf))
    passed = all(g["pass"] for g in gates)
    return gates, passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        default="datasets/grok_submission_2026",
        help="Path to datasets/<handle>_submission_2026",
    )
    parser.add_argument(
        "--skip-llama",
        action="store_true",
        help="Skip llama-cli load (layout / look only)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    submission = Path(args.submission)
    if not submission.is_absolute():
        submission = (Path.cwd() / submission).resolve()
    if not submission.is_dir():
        print(f"submission not found: {submission}", file=sys.stderr)
        return 2
    gates, passed = run(submission, skip_llama=args.skip_llama)
    report = {
        "submission": str(submission),
        "passed": passed,
        "gates": gates,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Starlight Chat 2026 gate — {submission.name}")
        for g in gates:
            mark = "PASS" if g["pass"] else "FAIL"
            print(f"  [{mark}] {g['gate']}: {g['detail']}")
        print("ALL GATES PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
