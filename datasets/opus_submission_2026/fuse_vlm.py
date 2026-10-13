#!/usr/bin/env python3
"""Bind the official Starlight detector to the llama.cpp SLM GGUF (Track B).

Downloads the upstream instruct GGUF named in model/serve.json when missing,
checks its sha256, and writes model/fusion_manifest.json with detector hashes.
Detector weights are used as-is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HF_BASE = "https://huggingface.co"


def load_serve() -> Dict[str, Any]:
    return json.loads((HERE / "model" / "serve.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(serve: Dict[str, Any], dest: Path) -> None:
    url = f"{HF_BASE}/{serve['hf_repo']}/resolve/main/{serve['hf_file']}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def llama_version() -> Optional[str]:
    exe = shutil.which("llama-cli")
    if not exe:
        return None
    proc = subprocess.run([exe, "--version"], capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="only write the manifest")
    args = parser.parse_args()
    serve = load_serve()
    gguf = HERE / serve["llm_gguf"]
    if not gguf.is_file() and not args.skip_download:
        download(serve, gguf)
    digest = sha256_file(gguf) if gguf.is_file() else None
    expected = serve.get("llm_sha256")
    if digest and expected and digest != expected:
        raise SystemExit(f"sha256 mismatch for {gguf.name}: {digest} != {expected}")

    det = ROOT / serve["detector"]
    det_gguf = ROOT / serve["detector_gguf"]
    manifest = {
        "track": serve["track"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_gguf": serve["llm_gguf"],
        "llm_bytes": gguf.stat().st_size if gguf.is_file() else 0,
        "llm_sha256": digest,
        "hf_repo": serve["hf_repo"],
        "hf_file": serve["hf_file"],
        "arch": serve["arch"],
        "quant": serve["quant"],
        "detector": serve["detector"],
        "detector_sha256": sha256_file(det) if det.is_file() else None,
        "detector_gguf": serve["detector_gguf"],
        "detector_gguf_sha256": sha256_file(det_gguf) if det_gguf.is_file() else None,
        "llama_cpp": llama_version(),
        "look": "look.py",
        "system_prompt": serve["system_prompt"],
        "note": (
            "Track B: llama.cpp loads only the SLM GGUF. look.py runs the official "
            "BalancedStarlightDetector plus spec extractors, injects the observation block, "
            "and emits a GBNF grammar that pins the verbatim payload in the reply."
        ),
    }
    out = HERE / "model" / "fusion_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out)
    if gguf.is_file():
        print(f"llm_gguf={gguf} ({gguf.stat().st_size} bytes, sha256 ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
