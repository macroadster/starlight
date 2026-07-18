#!/usr/bin/env python3
"""Parity check: GGUF tensors vs BalancedStarlightDetector state_dict.

Reads a GGUF written by export_starlight_gguf.py (Trin WriteGGUFF32 layout),
compares every exported float tensor to the live model (checkpoint or
--init-random with matching seed). Fails if max |Δ| >= 1e-6.

Optional forward pass dumps logits to fixtures/parity_logits.json.

Usage:
  python3 scripts/parity_starlight_gguf.py \\
    --gguf models/starlight.gguf --init-random

  python3 scripts/parity_starlight_gguf.py \\
    --gguf models/starlight.gguf --input models/detector_balanced.pth
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import writer helpers / model loader
from scripts.export_starlight_gguf import (  # noqa: E402
    ALIGNMENT,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGML_F32,
    build_model,
    export_state_dict,
)

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


class GGUFReader:
    """Minimal pure-Python GGUF v3 reader (F32 tensors + string meta)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.i = 0
        self.version = 0
        self.alignment = 32
        self.meta: Dict[str, str] = {}
        self.tensors: List[Dict[str, Any]] = []
        self.by_name: Dict[str, Dict[str, Any]] = {}
        self.data_base = 0
        self._parse()

    def _need(self, n: int) -> None:
        if self.i + n > len(self.data):
            raise EOFError(f"GGUF truncated at {self.i}+{n}")

    def _u32(self) -> int:
        self._need(4)
        v = struct.unpack_from("<I", self.data, self.i)[0]
        self.i += 4
        return v

    def _u64(self) -> int:
        self._need(8)
        v = struct.unpack_from("<Q", self.data, self.i)[0]
        self.i += 8
        return v

    def _f32(self) -> float:
        self._need(4)
        v = struct.unpack_from("<f", self.data, self.i)[0]
        self.i += 4
        return v

    def _str(self) -> str:
        n = self._u64()
        if n > 1 << 20:
            raise ValueError(f"string too long: {n}")
        self._need(n)
        s = self.data[self.i : self.i + n].decode("utf-8")
        self.i += n
        return s

    def _skip_value(self, vt: int) -> Any:
        """Read/skip a GGUF value; return string or stringified for meta."""
        # Type codes: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
        if vt == 0:  # UINT8
            self._need(1)
            v = self.data[self.i]
            self.i += 1
            return v
        if vt == 1:  # INT8
            self._need(1)
            v = struct.unpack_from("<b", self.data, self.i)[0]
            self.i += 1
            return v
        if vt == 2:  # UINT16
            self._need(2)
            v = struct.unpack_from("<H", self.data, self.i)[0]
            self.i += 2
            return v
        if vt == 3:  # INT16
            self._need(2)
            v = struct.unpack_from("<h", self.data, self.i)[0]
            self.i += 2
            return v
        if vt == 4:  # UINT32
            return self._u32()
        if vt == 5:  # INT32
            self._need(4)
            v = struct.unpack_from("<i", self.data, self.i)[0]
            self.i += 4
            return v
        if vt == 6:  # FLOAT32
            return self._f32()
        if vt == 7:  # BOOL
            self._need(1)
            v = self.data[self.i] != 0
            self.i += 1
            return v
        if vt == 8:  # STRING
            return self._str()
        if vt == 9:  # ARRAY
            et = self._u32()
            n = self._u64()
            return [self._skip_value(et) for _ in range(n)]
        if vt == 10:  # UINT64
            return self._u64()
        if vt == 11:  # INT64
            self._need(8)
            v = struct.unpack_from("<q", self.data, self.i)[0]
            self.i += 8
            return v
        if vt == 12:  # FLOAT64
            self._need(8)
            v = struct.unpack_from("<d", self.data, self.i)[0]
            self.i += 8
            return v
        raise ValueError(f"unsupported GGUF value type {vt}")

    def _parse(self) -> None:
        magic = self._u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"bad GGUF magic: {magic:#010x}")
        self.version = self._u32()
        if self.version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {self.version}")
        n_tensors = self._u64()
        n_kv = self._u64()

        for _ in range(n_kv):
            key = self._str()
            vt = self._u32()
            val = self._skip_value(vt)
            if key == "general.alignment":
                if isinstance(val, int) and val > 0:
                    self.alignment = int(val)
            if isinstance(val, str):
                self.meta[key] = val
            else:
                self.meta[key] = str(val)

        for _ in range(n_tensors):
            name = self._str()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            typ = self._u32()
            off = self._u64()
            info = {
                "name": name,
                "n_dims": n_dims,
                "dims": dims,
                "type": typ,
                "offset": off,
            }
            self.tensors.append(info)
            self.by_name[name] = info

        # data section starts at next alignment boundary
        pos = self.i
        align = self.alignment or ALIGNMENT
        if pos % align != 0:
            pos = ((pos // align) + 1) * align
        self.data_base = pos

    def list_names(self) -> List[str]:
        return [t["name"] for t in self.tensors]

    def load_f32(self, name: str) -> Tuple[List[float], List[int]]:
        info = self.by_name.get(name)
        if info is None:
            raise KeyError(f"tensor not found: {name}")
        if info["type"] != GGML_F32:
            raise TypeError(f"tensor {name} type={info['type']} (need F32=0)")
        dims = info["dims"]
        n_elem = 1
        for d in dims:
            n_elem *= int(d)
        # empty shape (0-d) → 1 element
        if len(dims) == 0:
            n_elem = 1
        base = self.data_base + int(info["offset"])
        need = n_elem * 4
        if base < 0 or base + need > len(self.data):
            raise EOFError(
                f"tensor {name} OOB: base={base} need={need} file={len(self.data)}"
            )
        vals = list(struct.unpack_from(f"<{n_elem}f", self.data, base))
        shape = [int(d) for d in dims]
        return vals, shape


def compare_tensors(
    model: torch.nn.Module, reader: GGUFReader, atol: float = 1e-6
) -> Dict[str, Any]:
    tensors, shapes, name_map = export_state_dict(model)
    gguf_names = set(reader.list_names())
    model_names = set(tensors.keys())

    missing_in_gguf = sorted(model_names - gguf_names)
    extra_in_gguf = sorted(gguf_names - model_names)

    max_abs = 0.0
    worst_name = ""
    per_tensor: Dict[str, float] = {}
    shape_mismatches: List[str] = []

    for name in sorted(model_names & gguf_names):
        expected = tensors[name]
        exp_shape = shapes[name]
        got, got_shape = reader.load_f32(name)
        # Compare shapes (empty vs [1] edge cases)
        if list(got_shape) != list(exp_shape):
            # Allow 0-d vs empty
            if not (len(exp_shape) == 0 and got_shape == [1]):
                shape_mismatches.append(
                    f"{name}: model={exp_shape} gguf={got_shape}"
                )
        if len(got) != len(expected):
            shape_mismatches.append(
                f"{name}: numel model={len(expected)} gguf={len(got)}"
            )
            continue
        # max abs
        local_max = 0.0
        for a, b in zip(expected, got):
            d = abs(a - b)
            if d > local_max:
                local_max = d
        per_tensor[name] = local_max
        if local_max > max_abs:
            max_abs = local_max
            worst_name = name

    ok = (
        max_abs < atol
        and not missing_in_gguf
        and not extra_in_gguf
        and not shape_mismatches
    )
    return {
        "ok": ok,
        "max_abs_error": max_abs,
        "worst_tensor": worst_name,
        "n_model_float": len(model_names),
        "n_gguf": len(gguf_names),
        "n_compared": len(model_names & gguf_names),
        "missing_in_gguf": missing_in_gguf,
        "extra_in_gguf": extra_in_gguf,
        "shape_mismatches": shape_mismatches,
        "atol": atol,
        "meta": dict(reader.meta),
        "skipped": name_map.get("skipped", []),
    }


def optional_forward(
    model: torch.nn.Module, image_path: Path, out_json: Path
) -> Optional[Dict[str, Any]]:
    try:
        from scripts.starlight_utils import load_unified_input
    except Exception as e:
        print(f"[warn] cannot import load_unified_input: {e}", file=sys.stderr)
        return None

    if not image_path.exists():
        print(f"[warn] sample image missing: {image_path}", file=sys.stderr)
        return None

    batch = load_unified_input(str(image_path))
    # load_unified_input returns 8 tensors without batch dim; lsb is HWC
    if not isinstance(batch, (list, tuple)) or len(batch) < 8:
        print(f"[warn] unexpected load_unified_input type: {type(batch)}", file=sys.stderr)
        return None
    pixel_tensor, meta, alpha, lsb, palette, palette_lsb, fmt, content = batch[:8]
    # Match trainer Dataset: permute lsb HWC -> CHW
    if lsb.dim() == 3 and lsb.shape[-1] == 3:
        lsb = lsb.permute(2, 0, 1)
    args = tuple(
        t.unsqueeze(0) for t in (
            pixel_tensor, meta, alpha, lsb, palette, palette_lsb, fmt, content
        )
    )

    model.eval()
    with torch.no_grad():
        stego, method, bit_order, method_id, method_probs, bit_order_id, emb = model(
            *args
        )

    result = {
        "image": str(image_path),
        "stego_logits": stego.reshape(-1).tolist(),
        "method_logits": method.reshape(-1).tolist(),
        "bit_order_logits": bit_order.reshape(-1).tolist(),
        "method_id": int(method_id.item()),
        "bit_order_id": int(bit_order_id.item()),
        "embedding_norm": float(emb.norm().item()),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Forward logits → {out_json}")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Parity check Starlight GGUF vs state_dict")
    p.add_argument("--gguf", type=Path, default=Path("models/starlight.gguf"))
    p.add_argument("--input", type=Path, default=Path("models/detector_balanced.pth"))
    p.add_argument("--init-random", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument(
        "--forward-image",
        type=Path,
        default=None,
        help="Optional image for a forward pass dump",
    )
    p.add_argument(
        "--logits-out",
        type=Path,
        default=Path("fixtures/parity_logits.json"),
    )
    args = p.parse_args(argv)

    if not args.gguf.exists():
        print(f"ERROR: GGUF not found: {args.gguf}", file=sys.stderr)
        return 2

    if args.init_random:
        torch.manual_seed(args.seed)

    model = build_model(args.init_random, args.input if not args.init_random else None)
    reader = GGUFReader(args.gguf)

    report = compare_tensors(model, reader, atol=args.atol)

    print(f"GGUF: {args.gguf} ({args.gguf.stat().st_size} bytes)")
    print(f"version={reader.version} alignment={reader.alignment}")
    print(f"meta general.name={reader.meta.get('general.name')}")
    print(f"meta starlight.arch={reader.meta.get('starlight.arch')}")
    print(f"n_model_float={report['n_model_float']} n_gguf={report['n_gguf']} "
          f"compared={report['n_compared']}")
    print(f"max|Δ|={report['max_abs_error']:.3e} worst={report['worst_tensor']!r}")
    if report["missing_in_gguf"]:
        print(f"MISSING in GGUF: {report['missing_in_gguf'][:10]}")
    if report["extra_in_gguf"]:
        print(f"EXTRA in GGUF: {report['extra_in_gguf'][:10]}")
    if report["shape_mismatches"]:
        print(f"SHAPE mismatches: {report['shape_mismatches'][:10]}")

    # Optional forward
    img = args.forward_image
    if img is None:
        candidate = _ROOT / "datasets/sample_submission_2025/clean/clean-0004.png"
        if candidate.exists():
            img = candidate
    if img is not None:
        optional_forward(model, img, args.logits_out)

    if not report["ok"]:
        print("PARITY FAILED", file=sys.stderr)
        return 1
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
