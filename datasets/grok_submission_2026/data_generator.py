#!/usr/bin/env python3
"""Generate a small 2026 eval set: clean covers + five Starlight methods.

Run from datasets/grok_submission_2026/. Images land in ./clean and ./stego
(gitignored). Tiny copies used by CI are written to eval/fixtures/.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import piexif
except ImportError:
    piexif = None

from scripts.stego_tool import embed_alpha, embed_lsb, embed_palette


def read_seeds(seeds_dir: Path) -> list[tuple[str, str]]:
    items = []
    for path in sorted(seeds_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            items.append((path.stem, text))
    return items


def make_cover(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx + np.sin(ang) * yy) / size
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    c0 = rng.integers(32, 200, size=3)
    c1 = rng.integers(32, 200, size=3)
    rgb = c0 + (c1 - c0) * grad[..., None]
    rgb = np.clip(rgb + rng.normal(0, 2.0, rgb.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def sidecar(category: str, technique: str, ai42: bool, bit_order: str | None = None) -> dict:
    emb = {"category": category, "technique": technique, "ai42": ai42}
    if bit_order:
        emb["bit_order"] = bit_order
    return {"embedding": emb}


def save_pair(
    clean_dir: Path,
    stego_dir: Path,
    fixtures: Path,
    stem: str,
    clean_img: Image.Image,
    stego_bytes: bytes,
    suffix: str,
    meta: dict,
    also_fixture: bool,
) -> None:
    clean_path = clean_dir / f"{stem}{suffix}"
    stego_path = stego_dir / f"{stem}{suffix}"
    fmt = {".png": "PNG", ".gif": "GIF", ".jpg": "JPEG"}[suffix]
    clean_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)
    clean_img.save(clean_path, fmt, quality=90)
    stego_path.write_bytes(stego_bytes)
    (stego_dir / f"{stem}{suffix}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    if also_fixture:
        fixtures.mkdir(parents=True, exist_ok=True)
        (fixtures / f"{stem}{suffix}").write_bytes(stego_bytes)
        if suffix == ".png":
            clean_img.save(fixtures / f"clean_{stem}{suffix}", fmt)


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def encode_gif(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "GIF")
    return buf.getvalue()


def encode_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def embed_exif_jpeg(cover: Image.Image, payload: bytes) -> bytes:
    if piexif is None:
        raise RuntimeError("piexif is required for EXIF embedding")
    img = cover.convert("RGB")
    exif_bytes = piexif.dump(
        {
            "0th": {},
            "Exif": {piexif.ExifIFD.UserComment: b"ASCII\x00\x00\x00" + payload + b"\x00"},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
    )
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90, exif=exif_bytes)
    return buf.getvalue()


def embed_eoi_jpeg(cover: Image.Image, payload: bytes) -> bytes:
    raw = encode_jpeg(cover)
    eoi = raw.rfind(b"\xff\xd9")
    if eoi < 0:
        raise RuntimeError("JPEG missing EOI")
    return raw[: eoi + 2] + payload + b"\x00"


def generate(limit: int, fixtures_only: bool) -> None:
    seeds = read_seeds(HERE / "seeds")
    if not seeds:
        raise SystemExit("no seeds/*.md")
    clean_dir = HERE / "clean"
    stego_dir = HERE / "stego"
    fixtures = HERE / "eval" / "fixtures"
    methods = [
        "alpha",
        "lsb.rgb",
        "palette",
        "exif",
        "raw",
    ]
    named = {
        "seed_protocol": "alpha",
        "seed_common_sense": "lsb.rgb",
        "seed_hidden_history": "palette",
        "seed_2142": "exif",
        "sample_seed": "raw",
    }
    n = 0
    for i, (name, text) in enumerate(seeds):
        if n >= limit:
            break
        payload = text.encode("utf-8")
        cover = make_cover(seed=2026 + i)
        method = named.get(name, methods[i % len(methods)])
        stem = f"{name}_{method.replace('.', '_')}_000"
        fixture = n < 6
        if method == "alpha":
            stego = embed_alpha(cover, payload)
            save_pair(
                clean_dir,
                stego_dir,
                fixtures,
                stem,
                cover,
                encode_png(stego),
                ".png",
                sidecar("pixel", "alpha", True, "lsb-first"),
                fixture,
            )
        elif method == "lsb.rgb":
            stego = embed_lsb(cover, payload + b"\x00")
            save_pair(
                clean_dir,
                stego_dir,
                fixtures,
                stem,
                cover,
                encode_png(stego.convert("RGB")),
                ".png",
                sidecar("pixel", "lsb.rgb", False, "msb-first"),
                fixture,
            )
        elif method == "palette":
            stego = embed_palette(cover, payload + b"\x00")
            save_pair(
                clean_dir,
                stego_dir,
                fixtures,
                stem,
                cover,
                encode_gif(stego),
                ".gif",
                sidecar("pixel", "palette", False, "msb-first"),
                fixture,
            )
        elif method == "exif":
            save_pair(
                clean_dir,
                stego_dir,
                fixtures,
                stem,
                cover,
                embed_exif_jpeg(cover, payload),
                ".jpg",
                sidecar("metadata", "exif", False),
                fixture,
            )
        else:
            save_pair(
                clean_dir,
                stego_dir,
                fixtures,
                stem,
                cover,
                embed_eoi_jpeg(cover, payload),
                ".jpg",
                sidecar("eoi", "raw", False),
                fixture,
            )
        n += 1
        print(f"wrote {stem} ({method})")
        if fixtures_only and n >= min(limit, 5):
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Stop after the public eval fixtures exist",
    )
    args = parser.parse_args()
    generate(limit=args.limit, fixtures_only=args.fixtures_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
