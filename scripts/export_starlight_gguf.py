#!/usr/bin/env python3
"""Export BalancedStarlightDetector weights to a Trin-compatible GGUF v3 (F32) file.

Tensor names match PyTorch state_dict keys exactly. Layout is little-endian
float32 row-major (torch contiguous CPU). Writer mirrors Trin
internal/weights.WriteGGUFF32 (GGUF magic, version 3, general.alignment=32,
string meta, sorted tensor infos, 32-byte padded data section).

Usage:
  python3 scripts/export_starlight_gguf.py \\
    --input models/detector_balanced.pth \\
    --output models/starlight.gguf \\
    --name-map models/starlight_gguf_map.json

  # CI / no checkpoint:
  python3 scripts/export_starlight_gguf.py --init-random \\
    --output models/starlight.gguf \\
    --name-map models/starlight_gguf_map.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Project root on sys.path for `import trainer`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

GGUF_MAGIC = 0x46554747  # "GGUF" LE
GGUF_VERSION = 3
GGML_F32 = 0
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
ALIGNMENT = 32

DEFAULT_META = {
    "general.name": "starlight-detector-balanced-v2",
    "starlight.meta_weight": "0.3",
    "starlight.proj_dim": "64",
    "starlight.fusion_dim": "448",
    "starlight.arch": "BalancedStarlightDetector",
}


def _put_u32(buf: bytearray, v: int) -> None:
    buf.extend(struct.pack("<I", v & 0xFFFFFFFF))


def _put_u64(buf: bytearray, v: int) -> None:
    buf.extend(struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF))


def _put_str(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    _put_u64(buf, len(b))
    buf.extend(b)


def _align_pad(n: int, align: int = ALIGNMENT) -> int:
    if n % align == 0:
        return 0
    return align - (n % align)


def tensor_role(name: str, shape: List[int]) -> str:
    """Heuristic role label for the name map."""
    base = name.rsplit(".", 1)[-1] if "." in name else name
    parent = name.rsplit(".", 1)[0] if "." in name else name
    idx = parent.rsplit(".", 1)[-1] if "." in parent else ""

    if "running_mean" in name or "running_var" in name or "num_batches" in name:
        return "bn"
    # Sequential BN blocks sit at indices 1, 4, 7 in this model
    if idx in ("1", "4", "7") and base in ("weight", "bias") and "conv" in name:
        return "bn"
    if base == "bias":
        return "bias"
    if base == "weight":
        if len(shape) == 4:
            return "conv"
        if len(shape) == 3:
            return "conv1d"
        if len(shape) == 2:
            return "linear"
        if len(shape) == 1:
            return "bn" if "bn" in name.lower() else "weight"
    return "param"


def write_gguf_f32(
    path: Path,
    tensors: Dict[str, List[float]],
    shapes: Dict[str, List[int]],
    meta: Optional[Dict[str, str]] = None,
) -> None:
    """Write a minimal GGUF v3 file with F32 tensors (matches Trin WriteGGUFF32)."""
    if meta is None:
        meta = {}

    names = sorted(tensors.keys())
    buf = bytearray()

    # magic + version
    _put_u32(buf, GGUF_MAGIC)
    _put_u32(buf, GGUF_VERSION)

    n_kv = 1 + len(meta)  # general.alignment + user strings
    _put_u64(buf, len(names))
    _put_u64(buf, n_kv)

    # general.alignment as UINT32
    _put_str(buf, "general.alignment")
    _put_u32(buf, GGUF_TYPE_UINT32)
    _put_u32(buf, ALIGNMENT)

    # Stable meta key order for reproducibility
    for k in sorted(meta.keys()):
        v = meta[k]
        _put_str(buf, k)
        _put_u32(buf, GGUF_TYPE_STRING)
        _put_str(buf, str(v))

    # Tensor infos; build data section with per-tensor 32-byte alignment
    data = bytearray()
    off = 0
    for name in names:
        vals = tensors[name]
        shape = shapes.get(name)
        if shape is None:
            shape = [len(vals)]
        _put_str(buf, name)
        _put_u32(buf, len(shape))
        for d in shape:
            _put_u64(buf, int(d))
        _put_u32(buf, GGML_F32)
        # align data offset
        pad = _align_pad(off, ALIGNMENT)
        if pad:
            data.extend(b"\x00" * pad)
            off += pad
        _put_u64(buf, off)
        for v in vals:
            data.extend(struct.pack("<f", float(v)))
        off += len(vals) * 4

    # pad header to alignment, then append data
    pad = _align_pad(len(buf), ALIGNMENT)
    if pad:
        buf.extend(b"\x00" * pad)
    buf.extend(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))


def build_model(init_random: bool, checkpoint: Optional[Path]) -> torch.nn.Module:
    from trainer import BalancedStarlightDetector

    model = BalancedStarlightDetector(meta_weight=0.3, proj_dim=64)
    model.eval()

    if init_random:
        return model

    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Provide a valid --input .pth path, or pass --init-random for a "
            "random-initialized BalancedStarlightDetector (CI / shape export)."
        )

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if "state_dict" in raw:
            state = raw["state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            state = raw["model"]
        else:
            # Assume bare state_dict
            state = raw
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(raw)}")

    # Strip common prefixes
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:8]}...", file=sys.stderr)
    if unexpected:
        print(
            f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:8]}...",
            file=sys.stderr,
        )
    return model


def export_state_dict(
    model: torch.nn.Module,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]], Dict[str, Any]]:
    """Extract float tensors from state_dict; skip integer buffers."""
    sd = model.state_dict()
    tensors: Dict[str, List[float]] = {}
    shapes: Dict[str, List[int]] = {}
    entries: Dict[str, Any] = {}
    skipped: List[Dict[str, Any]] = []

    for name, t in sd.items():
        if not t.is_floating_point():
            skipped.append(
                {
                    "name": name,
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                    "reason": "non-float buffer (e.g. num_batches_tracked); not needed for eval",
                }
            )
            continue
        t32 = t.detach().cpu().contiguous().float()
        shape = list(t32.shape)
        # 0-d scalar → keep empty shape as torch reports; prefer [1] only if needed
        flat = t32.reshape(-1).tolist()
        tensors[name] = flat
        shapes[name] = shape
        nbytes = t32.numel() * 4
        entries[name] = {
            "shape": shape,
            "dtype": "F32",
            "nbytes": nbytes,
            "role": tensor_role(name, shape),
            "numel": t32.numel(),
        }

    name_map: Dict[str, Any] = {
        "arch": "BalancedStarlightDetector",
        "general.name": DEFAULT_META["general.name"],
        "meta": dict(DEFAULT_META),
        "n_tensors_exported": len(tensors),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "tensors": entries,
        "notes": [
            "Tensor names match PyTorch state_dict keys exactly.",
            "Layout: row-major float32, same as torch contiguous CPU float32.",
            "BN num_batches_tracked (int64) skipped — eval uses running_mean/var.",
            "GGUF writer matches Trin internal/weights.WriteGGUFF32.",
        ],
    }
    return tensors, shapes, name_map


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Export BalancedStarlightDetector to Trin-compatible GGUF F32"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("models/detector_balanced.pth"),
        help="Path to .pth checkpoint (state_dict or wrapped)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/starlight.gguf"),
        help="Output GGUF path",
    )
    p.add_argument(
        "--name-map",
        type=Path,
        default=Path("models/starlight_gguf_map.json"),
        help="Output JSON name map path",
    )
    p.add_argument(
        "--init-random",
        action="store_true",
        help="Build random-init model (no checkpoint required)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used with --init-random (default 0)",
    )
    args = p.parse_args(argv)

    if args.init_random:
        torch.manual_seed(args.seed)

    model = build_model(args.init_random, args.input if not args.init_random else None)
    tensors, shapes, name_map = export_state_dict(model)

    write_gguf_f32(args.output, tensors, shapes, DEFAULT_META)

    args.name_map.parent.mkdir(parents=True, exist_ok=True)
    args.name_map.write_text(json.dumps(name_map, indent=2, sort_keys=False) + "\n")

    total_bytes = sum(e["nbytes"] for e in name_map["tensors"].values())
    print(f"Exported {name_map['n_tensors_exported']} F32 tensors "
          f"({total_bytes} bytes payload) → {args.output}")
    print(f"Skipped {name_map['n_skipped']} non-float buffers")
    print(f"Name map → {args.name_map}")
    print(f"Meta: {DEFAULT_META}")
    sample = sorted(tensors.keys())[:5]
    print(f"Sample names: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
