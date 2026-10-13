#!/usr/bin/env python3
"""Generate the opus 2026 eval set: procedural covers, five methods, spec variants.

Covers the canonical five Starlight methods plus the variants STEGO_FORMAT_SPEC
v2.1 allows (both bit orders for lsb.rgb and palette, RGB payloads inside RGBA,
EXIF in PNG without an encoding header, AI42-prefixed tails after PNG IEND,
UTF-8 payloads). Canonical embeddings use scripts/stego_tool.py; variants use the
small embedders below. Every stego file is read back with look.py before it is
kept. Writes ./clean, ./stego (+ sidecars), eval/fixtures/, eval/stego_qa.jsonl.

    python3 data_generator.py --limit 9
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import piexif
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scripts.stego_tool import embed_alpha, embed_lsb, embed_palette  # noqa: E402

import look  # noqa: E402

COVER_SIZE = 160
REL = Path("datasets") / HERE.name


def make_cover(seed: int, size: int = COVER_SIZE) -> Image.Image:
    """Gradient sky, a few soft shapes, and sensor-like noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    ang = rng.uniform(0, 2 * np.pi)
    grad = np.cos(ang) * xx + np.sin(ang) * yy
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    c0, c1 = rng.integers(20, 120, 3), rng.integers(120, 235, 3)
    rgb = c0 + (c1 - c0) * grad[..., None]
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(2, 6))):
        x, y = rng.integers(0, size, 2)
        r = int(rng.integers(size // 12, size // 4))
        fill = tuple(int(v) for v in rng.integers(30, 250, 3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    arr = np.asarray(img).astype(np.float32) + rng.normal(0, 3.0, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _bits(payload: bytes, order: str) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="little" if order == "lsb-first" else "big")


def embed_rgb_lsb(cover: Image.Image, payload: bytes, order: str) -> Image.Image:
    arr = np.asarray(cover.convert("RGB")).copy()
    flat = arr.reshape(-1)
    bits = _bits(payload + b"\x00", order)
    if bits.size > flat.size:
        raise ValueError("payload too large for RGB LSB")
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return Image.fromarray(arr, "RGB")


def embed_palette_lsb(cover: Image.Image, payload: bytes, order: str) -> Image.Image:
    pal = cover.convert("P", palette=Image.ADAPTIVE, colors=256)
    idx = np.asarray(pal).copy()
    flat = idx.reshape(-1)
    bits = _bits(payload + b"\x00", order)
    if bits.size > flat.size:
        raise ValueError("payload too large for palette LSB")
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    out = Image.fromarray(idx, "P")
    out.putpalette(pal.getpalette())
    return out


def _png(img: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", **kw)
    return buf.getvalue()


def _gif(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "GIF", optimize=False)
    return buf.getvalue()


def _jpeg(img: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90, **kw)
    return buf.getvalue()


def _exif(comment: bytes) -> bytes:
    return piexif.dump({"0th": {}, "Exif": {piexif.ExifIFD.UserComment: comment}, "GPS": {}, "1st": {}, "thumbnail": None})


def sidecar(category: str, technique: str, ai42: bool, bit_order: str | None, clean_file: str) -> Dict:
    emb = {"category": category, "technique": technique, "ai42": ai42}
    if bit_order:
        emb["bit_order"] = bit_order
    return {"embedding": emb, "clean_file": clean_file}


# name -> (method, ext, bit_order, ai42, category, builder(cover, payload) -> (stego, clean) bytes)
Builder = Callable[[Image.Image, bytes], Tuple[bytes, bytes]]
RECIPES: Dict[str, Tuple[str, str, str | None, bool, str, Builder]] = {
    "opus_protocol": ("alpha", ".png", "lsb-first", True, "pixel",
                      lambda c, p: (_png(embed_alpha(c, p)), _png(c.convert("RGBA")))),
    "opus_ledger": ("lsb.rgb", ".png", "msb-first", False, "pixel",
                    lambda c, p: (_png(embed_lsb(c, p)), _png(c.convert("RGBA")))),
    "opus_bitorder": ("lsb.rgb", ".png", "lsb-first", False, "pixel",
                      lambda c, p: (_png(embed_rgb_lsb(c, p, "lsb-first")), _png(c))),
    "opus_colors": ("palette", ".gif", "msb-first", False, "pixel",
                    lambda c, p: (_gif(embed_palette(c, p)), _gif(c.convert("P", palette=Image.ADAPTIVE)))),
    "opus_utf8": ("palette", ".gif", "lsb-first", False, "pixel",
                  lambda c, p: (_gif(embed_palette_lsb(c, p, "lsb-first")), _gif(c.convert("P", palette=Image.ADAPTIVE)))),
    "opus_comment": ("exif", ".jpg", None, False, "metadata",
                     lambda c, p: (_jpeg(c, exif=_exif(b"ASCII\x00\x00\x00" + p + b"\x00")), _jpeg(c))),
    "opus_question": ("exif", ".png", None, False, "metadata",
                      lambda c, p: (_png(c, exif=_exif(p + b"\x00")), _png(c))),
    "opus_postscript": ("raw", ".jpg", None, False, "eoi",
                        lambda c, p: (_jpeg(c) + p + b"\x00", _jpeg(c))),
    "opus_tail": ("raw", ".png", None, False, "eoi",
                  lambda c, p: (_png(c) + b"AI42" + p + b"\x00", _png(c))),
}
CLEAN_FIXTURES = {"opus_protocol", "opus_comment", "opus_colors"}


def read_seeds() -> List[Tuple[str, str]]:
    out = []
    for path in sorted((HERE / "seeds").glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if '"""' in text or "\n" in text:
            raise SystemExit(f"{path.name}: public seeds must be one line without triple quotes")
        out.append((path.stem, text))
    return out


def generate(limit: int) -> None:
    clean_dir, stego_dir = HERE / "clean", HERE / "stego"
    fixtures = HERE / "eval" / "fixtures"
    for d in (clean_dir, stego_dir, fixtures):
        d.mkdir(parents=True, exist_ok=True)
    qa: List[Dict] = []
    for i, (name, text) in enumerate(read_seeds()[:limit]):
        if name not in RECIPES:
            print(f"skip {name}: no recipe")
            continue
        method, ext, order, ai42, category, build = RECIPES[name]
        cover = make_cover(seed=2142 + 7 * i)
        stego_bytes, clean_bytes = build(cover, text.encode("utf-8"))
        stem = f"{name}_{method.replace('.', '_')}_{i:03d}"
        stego_path, clean_path = stego_dir / f"{stem}{ext}", clean_dir / f"{stem}{ext}"
        stego_path.write_bytes(stego_bytes)
        clean_path.write_bytes(clean_bytes)
        meta = sidecar(category, method, ai42, order, clean_path.name)
        (stego_dir / f"{stem}{ext}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        got = look.observe(stego_path, use_detector=False)
        if got["payload"] != text or got["method"] != method:
            raise SystemExit(f"{stem}: read-back mismatch {got['method']} {got['payload'][:60]!r}")
        if look.observe(clean_path, use_detector=False)["payload"]:
            raise SystemExit(f"{stem}: clean cover decodes a payload")

        (fixtures / stego_path.name).write_bytes(stego_bytes)
        qa.append({"id": name, "image": str(REL / "eval" / "fixtures" / stego_path.name),
                   "method": method, "bit_order": order or "none", "payload": text, "clean": False})
        if name in CLEAN_FIXTURES:
            cname = f"clean_{stem}{ext}"
            (fixtures / cname).write_bytes(clean_bytes)
            qa.append({"id": f"clean_{name}", "image": str(REL / "eval" / "fixtures" / cname),
                       "method": "none", "bit_order": "none", "payload": "", "clean": True})
        print(f"wrote {stem}{ext} ({method}{', ' + order if order else ''})")
    with (HERE / "eval" / "stego_qa.jsonl").open("w", encoding="utf-8") as fh:
        for rec in qa:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(qa)} rows -> eval/stego_qa.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=len(RECIPES))
    args = parser.parse_args()
    generate(args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
