#!/usr/bin/env python3
"""Bind the official detector to a Track A llama.cpp pair.

Downloads (or hard-links) the instruct GGUF named in model/serve.json, then
builds model/mmproj.gguf from embedding_head. Does not mutate detector weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HF_BASE = "https://huggingface.co"
SIBLING_GGUF = (
    ROOT
    / "datasets"
    / "grok_submission_2026"
    / "model"
    / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)


def load_serve() -> Dict[str, Any]:
    return json.loads((HERE / "model" / "serve.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dest_for(serve: Dict[str, Any], key: str, default: str) -> Path:
    raw = serve.get(key) or default
    path = Path(raw)
    if not path.is_absolute():
        path = HERE / path
    return path


def ensure_llm(serve: Dict[str, Any], dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if SIBLING_GGUF.is_file() and SIBLING_GGUF.stat().st_size > 1_000_000:
        try:
            os.link(SIBLING_GGUF, dest)
            print(f"linked {SIBLING_GGUF} -> {dest}", file=sys.stderr)
            return dest
        except OSError as exc:
            print(f"hardlink failed ({exc}); copying", file=sys.stderr)
            dest.write_bytes(SIBLING_GGUF.read_bytes())
            return dest
    repo = serve.get("hf_repo")
    filename = serve.get("hf_file")
    if not repo or not filename:
        raise SystemExit("serve.json missing hf_repo / hf_file and local GGUF is absent")
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def write_manifest(serve: Dict[str, Any], gguf: Path, mmproj: Path) -> Path:
    det = ROOT / serve.get("detector", "models/detector_balanced.pth")
    det_gguf = ROOT / serve.get("detector_gguf", "models/starlight.gguf")
    parity = HERE / "model" / "embedding_parity.json"
    manifest = {
        "track": "A",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_gguf": str(gguf.relative_to(HERE)) if gguf.is_relative_to(HERE) else str(gguf),
        "llm_bytes": gguf.stat().st_size if gguf.is_file() else 0,
        "llm_sha256": sha256_file(gguf) if gguf.is_file() else None,
        "mmproj_gguf": str(mmproj.relative_to(HERE)) if mmproj.is_relative_to(HERE) else str(mmproj),
        "mmproj_bytes": mmproj.stat().st_size if mmproj.is_file() else 0,
        "mmproj_sha256": sha256_file(mmproj) if mmproj.is_file() else None,
        "arch": serve.get("arch"),
        "quant": serve.get("quant"),
        "hf_repo": serve.get("hf_repo"),
        "hf_file": serve.get("hf_file"),
        "detector": str(det.relative_to(ROOT)) if det.exists() else str(det),
        "detector_exists": det.is_file(),
        "detector_gguf": str(det_gguf.relative_to(ROOT)) if det_gguf.exists() else str(det_gguf),
        "detector_gguf_exists": det_gguf.is_file(),
        "parity": "model/embedding_parity.json" if parity.is_file() else None,
        "system_prompt": "model/system_prompt.txt",
        "note": (
            "Track A: llama-cli loads the SLM GGUF plus mmproj.gguf and takes "
            "--image. The projector's patch is a linear fit to embedding_head. "
            "The decoded secret is not in the prompt or the weights."
        ),
    }
    out = HERE / "model" / "fusion_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-mmproj", action="store_true")
    parser.add_argument("--covers", type=int, default=12)
    args = parser.parse_args()
    serve = load_serve()
    dest = dest_for(serve, "llm_gguf", "model/llm.gguf")
    mmproj = dest_for(serve, "mmproj_gguf", "model/mmproj.gguf")
    if not args.skip_download:
        ensure_llm(serve, dest)
    elif not dest.is_file():
        print(f"GGUF missing at {dest}", file=sys.stderr)
        return 1
    if not args.skip_mmproj:
        cmd = [sys.executable, str(HERE / "export_mmproj.py"), "--covers", str(args.covers)]
        build = subprocess.run(cmd, cwd=str(HERE), check=False)
        if build.returncode != 0:
            return build.returncode
    manifest = write_manifest(serve, dest, mmproj)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
