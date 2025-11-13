#!/usr/bin/env python3
"""
data_generator.py

Generator + Validator + Summary for Project Starlight dataset contributions.

Features:
- Generates clean and stego image pairs from markdown payloads.
- Auto-detects algorithm name from embedding function names.
- Creates clean/ and stego/ directories if missing.
- Follows filename format: {payload_name}_{algorithm}_{index:03d}.{ext}
- Supports PNG, WebP, GIF (palette-based), and JPEG (EXIF/EOI).
- Uses minimal perceptual distortion for palette-based GIF embedding.
- Generates 10 image pairs per payload per algorithm by default.
- Integrated validation (naming, pairing, algorithm-format consistency).
- Summary report output (JSON + console).
- CLI options: --validate-only, --regen, --limit N
"""

import os
import io
import re
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Callable, List, Optional
from PIL import Image, ImageDraw

try:
    import piexif
except Exception:
    piexif = None

# --- CONFIGURATION ---
DEFAULT_IMG_SIZE = (256, 256)
IMAGES_PER_PAYLOAD_PER_ALGO = 10
CLEAN_DIR = Path("clean")
STEGO_DIR = Path("stego")
SUMMARY_PATH = Path("dataset_summary.json")

ALGO_FORMATS = {
    "lsb": ["png", "webp"],
    "alpha": ["png", "webp"],
    "exif": ["jpg"],
    "eoi": ["jpg"],
    "palette": ["gif"],
}

VALID_FILENAME_RE = re.compile(r"^(.+)_([a-z0-9]+)_(\d{3})\.([a-z0-9]+)$")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# === Utility Functions ===
def ensure_dirs():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    STEGO_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Ensured clean/ and stego/ directories exist.")


def find_md_payloads() -> List[Path]:
    return sorted([p for p in Path("seeds").glob("*.md") if p.is_file()])


def load_payload(md_path: Path) -> bytes:
    # Payload is loaded as UTF-8 bytes
    return md_path.read_text(encoding="utf-8").encode("utf-8")


# --- NEW: ASCII CHECK UTILITY ---
def is_pure_ascii(data: bytes) -> bool:
    """Checks if the byte string contains only 7-bit ASCII characters (values < 128)."""
    return all(b < 128 for b in data)
# --------------------------------


def next_available_index(payload_name: str, algorithm: str) -> int:
    pattern = re.compile(rf"^{re.escape(payload_name)}_{re.escape(algorithm)}_(\d{{3}})\.")
    used = {int(m.group(1)) for f in CLEAN_DIR.glob("*") if (m := pattern.match(f.name))}
    for i in range(1000):
        if i not in used:
            return i
    raise RuntimeError("No available indices left for this payload/algorithm.")


def save_image(img: Image.Image, path: Path, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".webp":
        # Pillow defaults to lossy unless explicitly set to lossless=True
        img.save(path, format="WEBP", lossless=True)

    elif ext == ".png":
        # PNG is inherently lossless, but avoid palette mode
        if img.mode == "P":
            img = img.convert("RGBA")
        img.save(path, format="PNG", optimize=True)

    elif ext == ".gif":
        # Preserve GIF frames if any
        img.save(path, format="GIF", save_all=True)

    elif ext in [".jpg", ".jpeg"]:
        # JPEG is lossy; only use if dataset explicitly requires it
        img.convert("RGB").save(path, format="JPEG", quality=95)

    else:
        # Default fallback for other formats
        img.save(path)

# === Clean Image Generator ===
def generate_clean_image(size=DEFAULT_IMG_SIZE, mode="RGBA", seed=None) -> Image.Image:
    if seed is not None:
        random.seed(seed)
    w, h = size
    img = Image.new(mode, size, color=(255, 255, 255, 255) if "A" in mode else (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        c = int(200 * (y / h)) + 30
        color = (c, 255 - c, (c * 2) % 255, 255) if "A" in mode else (c, 255 - c, (c * 2) % 255)
        draw.line([(0, y), (w, y)], fill=color)
    for _ in range(6):
        cx, cy, r = random.randint(0, w), random.randint(0, h), random.randint(10, min(w, h) // 4)
        fill = tuple(random.randint(0, 255) for _ in range(3)) + ((255,) if "A" in mode else ())
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    return img


# === Embedding Implementations ===
def embed_lsb(cover: Image.Image, payload: bytes) -> Image.Image:
    img = cover.convert("RGBA")
    pixels = list(img.getdata())
    bits = "".join(f"{b:08b}" for b in payload) + "00000000"
    max_bits = len(pixels) * 3
    if len(bits) > max_bits:
        raise ValueError("Payload too large for LSB embedding.")
    new_data, i = [], 0
    for r, g, b, a in pixels:
        if i < len(bits): r = (r & 0xFE) | int(bits[i]); i += 1
        if i < len(bits): g = (g & 0xFE) | int(bits[i]); i += 1
        if i < len(bits): b = (b & 0xFE) | int(bits[i]); i += 1
        new_data.append((r, g, b, a))
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    return out

def embed_alpha(cover: Image.Image, payload: bytes) -> Image.Image:
    img = cover.convert("RGBA")
    prefix = b"AI42"
    payload = prefix + payload + b"\x00"  # null terminator
    pixels = list(img.getdata())
    
    if len(payload) * 8 > len(pixels):
        raise ValueError("Payload too large for alpha embedding.")

    new_data = []
    pixel_index = 0

    for byte_val in payload:
        for bit_index in range(8):  # LSB-first
            r, g, b, a = pixels[pixel_index]
            bit_to_embed = (byte_val >> bit_index) & 0x01
            a = (a & 0xFE) | bit_to_embed
            new_data.append((r, g, b, a))
            pixel_index += 1

    # Append remaining pixels unmodified
    new_data.extend(pixels[pixel_index:])
    
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    return out

def embed_palette(cover: Image.Image, payload: bytes) -> Image.Image:
    """Minimal-change palette embedding for GIF."""
    img = cover.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    data = list(img.getdata())
    
    # Append the null terminator bit string (8 bits)
    bits = "".join(f"{b:08b}" for b in payload) + "00000000"
    
    required_bits = len(bits)
    available_indices = len(data)
    
    if required_bits > available_indices:
        # New, clearer error message
        raise ValueError(
            f"Payload too large for palette embedding. Requires {required_bits} bits, "
            f"but only {available_indices} indices are available."
        )
        
    new_data, i = [], 0
    for idx in data:
        if i < required_bits:
            # LSB insertion
            idx = (idx & 0xFE) | int(bits[i]); i += 1
        new_data.append(idx)
        
    # The image is guaranteed to have enough capacity now, but keeping the logic clean.
    out = Image.new("P", img.size)
    out.putdata(new_data)
    palette = img.getpalette()
    if palette is not None:
        out.putpalette(palette)
    return out

def embed_exif(cover: Image.Image, payload: bytes) -> Image.Image:
    if piexif is None:
        raise RuntimeError("piexif not installed; cannot perform EXIF embedding.")
    img = cover.convert("RGB")
    exif = {"0th": {}, "Exif": {piexif.ExifIFD.UserComment: b"ASCII\x00\x00\x00" + payload[:2000]}, "GPS": {}, "1st": {}, "thumbnail": None}
    img.info["exif_bytes"] = piexif.dump(exif)
    return img


def embed_eoi(cover: Image.Image, payload: bytes) -> Image.Image:
    img = cover.convert("RGB")
    img.info["eoi_append"] = payload
    return img


# === JPEG helpers ===
def save_jpeg_with_exif(img, path, exif_bytes, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", exif=exif_bytes, quality=quality)


def save_jpeg_with_eoi_append(img, path, append_bytes, quality=95):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    data = buf.getvalue()
    eoi = b"\xff\xd9"
    i = data.rfind(eoi)
    if i == -1:
        raise RuntimeError("JPEG EOI not found.")
    new_data = data + append_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(new_data)


# === Core Generation ===
def algorithm_name(func: Callable) -> str:
    n = func.__name__.lower()
    return n.split("embed_")[1] if n.startswith("embed_") else n


def generate_for_payload(payload_name: str, payload: bytes, funcs: List[Callable], force_regen=False, limit=None):
    stats = {}
    for f in funcs:
        algo = algorithm_name(f)
        
        # --- NEW: UTF-8 (Non-ASCII) CHECK FOR GIF PALETTE ---
        if algo == "palette" and not is_pure_ascii(payload):
            logging.warning(f"Skipping {payload_name} for 'palette': Payload contains non-ASCII (UTF-8) characters which are unstable in LSB.")
            continue # Skip this payload/algorithm combination entirely if it contains non-ASCII
        # ----------------------------------------------------
        
        current_payload = payload
        formats = ALGO_FORMATS.get(algo, ["png"])
        idx = next_available_index(payload_name, algo)
        count = 0
        limit = limit or IMAGES_PER_PAYLOAD_PER_ALGO
        while count < limit:
            ext = formats[count % len(formats)]
            fname = f"{payload_name}_{algo}_{idx:03d}.{ext}"
            clean, stego = CLEAN_DIR / fname, STEGO_DIR / fname
            if clean.exists() and not force_regen:
                idx += 1
                continue
            seed = random.randint(0, 2**32 - 1)
            mode = "RGBA" if ext in ["png", "webp"] else "RGB"
            img = generate_clean_image(mode=mode, seed=seed)
            save_image(img, clean)
            try:
                stego_img = f(img, current_payload)
            except Exception as e:
                logging.warning(f"Skipping {fname}: {e}")
                clean.unlink(missing_ok=True)
                idx += 1
                continue
            try:
                if algo == "exif":
                    save_jpeg_with_exif(stego_img, stego, stego_img.info.get("exif_bytes"))
                elif algo == "eoi":
                    save_jpeg_with_eoi_append(stego_img, stego, stego_img.info.get("eoi_append", b""))
                else:
                    save_image(stego_img, stego)
                
                # --- Create JSON Sidecar ---
                json_path = stego.with_suffix(stego.suffix + '.json')
                embedding_data = {}
                if algo == 'lsb':
                    embedding_data = {"category": "pixel", "technique": "lsb.rgb", "ai42": False, "bit_order": "msb-first"}
                elif algo == 'alpha':
                    embedding_data = {"category": "pixel", "technique": "alpha", "ai42": True, "bit_order": "lsb-first"}
                elif algo == 'palette':
                    embedding_data = {"category": "pixel", "technique": "palette", "ai42": False, "bit_order": "msb-first"}
                elif algo == 'exif':
                    embedding_data = {"category": "metadata", "technique": "exif", "ai42": False}
                elif algo == 'eoi':
                    embedding_data = {"category": "eoi", "technique": "raw", "ai42": False}
                
                if embedding_data:
                    sidecar_content = {
                        "embedding": embedding_data,
                        "clean_file": clean.name
                    }
                    json_path.write_text(json.dumps(sidecar_content, indent=2))

            except Exception as e:
                logging.warning(f"Failed to save stego {fname}: {e}")
                clean.unlink(missing_ok=True)
                idx += 1
                continue
            logging.info(f"Wrote clean/stego: {fname}")
            stats[algo] = stats.get(algo, 0) + 1
            count += 1
            idx += 1
    return stats


# === Validation + Summary ===
def validate_dataset() -> dict:
    logging.info("\n--- Validating dataset ---")
    result = {"valid": True, "errors": []}
    clean_files = {f.name for f in CLEAN_DIR.glob("*") if f.is_file() and not f.name.endswith('.json')}
    stego_files = {f.name for f in STEGO_DIR.glob("*") if f.is_file() and not f.name.endswith('.json')}

    # Missing pairs
    missing_in_stego = clean_files - stego_files
    missing_in_clean = stego_files - clean_files
    for f in sorted(missing_in_stego):
        result["errors"].append(f"Missing stego for {f}")
    for f in sorted(missing_in_clean):
        result["errors"].append(f"Missing clean for {f}")

    # Format and consistency
    for f in clean_files | stego_files:
        m = VALID_FILENAME_RE.match(f)
        if not m:
            result["errors"].append(f"Invalid filename format: {f}")
            continue
        _, algo, _, ext = m.groups()
        valid_exts = ALGO_FORMATS.get(algo, [])
        if valid_exts and ext not in valid_exts:
            result["errors"].append(f"Algorithm-format mismatch: {f}")

    result["valid"] = len(result["errors"]) == 0
    if result["valid"]:
        logging.info("✅ Validation passed: all files consistent and properly named.")
    else:
        logging.error("❌ Validation failed:")
        for err in result["errors"]:
            logging.error(f"  - {err}")
    return result


def summarize_dataset(validation: dict):
    payloads = {f.stem.split("_")[0] for f in CLEAN_DIR.glob("*") if f.is_file()}
    algos = set()
    for f in CLEAN_DIR.glob("*"):
        if f.is_file():
            match = VALID_FILENAME_RE.match(f.name)
            if match:
                algos.add(match.group(2))
    summary = {
        "payload_count": len(payloads),
        "algorithms_used": sorted(list(algos)),
        "clean_files": len(list(CLEAN_DIR.glob("*"))),
        "stego_files": len(list(STEGO_DIR.glob("*"))),
        "validation_passed": validation.get("valid", False),
        "errors": validation.get("errors", []),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    logging.info(f"📊 Summary written to {SUMMARY_PATH}")
    logging.info(json.dumps(summary, indent=2))
    return summary


# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Project Starlight Dataset Generator + Validator + Summary")
    parser.add_argument("--validate-only", action="store_true", help="Run validation only.")
    parser.add_argument("--regen", action="store_true", help="Force regeneration even if files exist.")
    parser.add_argument("--limit", type=int, help="Limit images per payload per algorithm.")
    args = parser.parse_args()

    ensure_dirs()
    if args.validate_only:
        validation = validate_dataset()
        summarize_dataset(validation)
        return

    md_files = find_md_payloads()
    if not md_files:
        logging.warning("No .md payloads found. Add one and re-run.")
        return

    funcs = [embed_lsb, embed_alpha, embed_palette, embed_exif, embed_eoi]
    total_stats = {}
    for md in md_files:
        name, payload = md.stem, load_payload(md)
        stats = generate_for_payload(name, payload, funcs, force_regen=args.regen, limit=args.limit)
        for k, v in stats.items():
            total_stats[k] = total_stats.get(k, 0) + v

    validation = validate_dataset()
    summary = summarize_dataset(validation)
    summary["generated_counts"] = total_stats
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    logging.info("✅ Dataset generation and validation complete.")


if __name__ == "__main__":
    main()
