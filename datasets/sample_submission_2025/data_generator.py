#!/usr/bin/env python3
"""
Data Generator - ALIGNED WITH STEGO_FORMAT_SPEC.md

Key Format Rules:
1. Alpha: LSB-first (byte-reversed) bit order with AI42 hint
2. LSB: MSB-first, null-terminated (no hint)
3. Palette: MSB-first, null-terminated
4. EXIF: UserComment with ASCII encoding header
5. EOI: Raw append after JPEG EOI marker
"""

import os

import io

import sys

import random

import logging

import argparse

import json

from pathlib import Path

from typing import Callable

from PIL import Image



import itertools



try:

    from tqdm import tqdm

except ImportError:

    def tqdm(iterable, *args, **kwargs):

        print("tqdm not found, progress bar will not be shown. To install: pip install tqdm", file=sys.stderr)

        return iterable







try:



    import piexif



except ImportError:



    piexif = None


def embed_alpha(cover: Image.Image, payload: bytes) -> Image.Image:
    """
    Embed in alpha channel using LSB-first (byte-reversed) bit order.
    SPEC: AI42 (LSB-first) + Message (LSB-first) + 0x00 (LSB-first)
    Embeds the provided payload.
    """
    img = cover.convert("RGBA")
    
    pixels = list(img.getdata())
    
    prefix = b"AI42"
    full_payload = prefix + payload + b" \x00"
    
    if len(full_payload) * 8 > len(pixels):
        raise ValueError(f"Payload too large for alpha embedding. Needs {len(full_payload) * 8} bits, has {len(pixels)}.")
    
    new_data = []
    pixel_index = 0
    
    for byte_val in full_payload:
        # Embed each byte in LSB-first order (bit 0 = LSB of byte)
        for bit_index in range(8):
            r, g, b, a = pixels[pixel_index]
            # Extract bit in LSB-first order
            bit_to_embed = (byte_val >> bit_index) & 0x01
            a = (a & 0xFE) | bit_to_embed
            new_data.append((r, g, b, a))
            pixel_index += 1
    
    new_data.extend(pixels[pixel_index:])
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    return out


def embed_lsb(cover: Image.Image, payload: bytes) -> Image.Image:
    """
    Embed in RGB channels using MSB-first bit order.
    SPEC: Generic LSB, null-terminated, no hint. Always produces an RGB image.
    Embeds the provided payload.
    """
    img = cover.convert("RGB")  # Always convert to RGB
    pixels = list(img.getdata())
    
    max_lsb_bits = len(pixels) * 3 # Each pixel has 3 RGB bits
    
    # MSB-first encoding with null terminator
    bits = "".join(f"{b:08b}" for b in payload) + "00000000"
    
    if len(bits) > max_lsb_bits:
        raise ValueError(f"Payload too large for LSB embedding. Needs {len(bits)} bits, has {max_lsb_bits}.")
    
    new_data = []
    bit_index = 0
    
    for r, g, b in pixels:
        if bit_index < len(bits):
            r = (r & 0xFE) | int(bits[bit_index])
            bit_index += 1
        if bit_index < len(bits):
            g = (g & 0xFE) | int(bits[bit_index])
            bit_index += 1
        if bit_index < len(bits):
            b = (b & 0xFE) | int(bits[bit_index])
            bit_index += 1
        new_data.append((r, g, b))
    
    out = Image.new("RGB", img.size)
    out.putdata(new_data)
    return out

def embed_palette(cover: Image.Image, payload: bytes) -> Image.Image:
    """
    Embed in palette indices using MSB-first bit order.
    SPEC: MSB-first, null-terminated. Embeds the provided payload.
    """
    if cover.mode != 'P':
        img = cover.convert("P", palette=Image.ADAPTIVE, colors=256)
    else:
        img = cover.copy()

    if img.palette is None:
        raise ValueError("Image must have a palette for this technique.")

    indices = list(img.getdata())
    
    # Convert payload to MSB-first bits + null terminator
    bits = ""
    for byte in payload:
        bits += format(byte, '08b')  # MSB-first
    bits += "00000000"  # null terminator

    if len(bits) > len(indices):
        raise ValueError(
            f"Payload too large for palette index embedding. "
            f"Capacity: {len(indices)} bits, Required: {len(bits)} bits."
        )

    for i in range(len(bits)):
        indices[i] = (indices[i] & 0xFE) | int(bits[i])

    img.putdata(indices)
    return img


def embed_exif(cover: Image.Image, payload: bytes) -> Image.Image:
    """
    Embed in EXIF UserComment with ASCII encoding header.
    SPEC: ASCII\x00\x00\x00 prefix + message. Embeds the provided payload.
    """
    if piexif is None:
        raise RuntimeError("piexif not installed; cannot perform EXIF embedding.")
    
    img = cover.convert("RGB")
    
    # SPEC: Use ASCII encoding header
    full_payload = b"ASCII\x00\x00\x00" + payload
    if len(full_payload) > 65535:
        raise ValueError("Payload too large for EXIF UserComment (max 65535 bytes).")

    exif_dict = {
        "0th": {},
        "Exif": {
            piexif.ExifIFD.UserComment: full_payload
        },
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }
    
    img.info["exif_bytes"] = piexif.dump(exif_dict)
    return img


def embed_eoi(cover: Image.Image, payload: bytes) -> Image.Image:
    """
    Embed after JPEG EOI marker.
    SPEC: Raw append after EOI marker (may have hint prefix). Embeds the provided payload.
    """
    img = cover.convert("RGB")
    
    img.info["eoi_append"] = payload
    return img


# === SAVING LOGIC ===

def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".webp":
        img.save(path, format="WEBP", lossless=True)
    elif ext == ".png":
        if img.mode == "P":
            img = img.convert("RGBA")
        img.save(path, format="PNG")
    elif ext == ".gif":
        img.save(path, format="GIF", save_all=True)
    elif ext in [".jpg", ".jpeg"]:
        img.convert("RGB").save(path, format="JPEG", quality=95)
    elif ext == ".bmp":
        img.save(path, format="BMP")
    else:
        img.save(path)


def save_jpeg_with_exif(img, path, exif_bytes, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", exif=exif_bytes, quality=quality)


def save_jpeg_with_eoi_append(img, path, append_bytes, quality=95):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    data = buf.getvalue()
    if b'\xff\xd9' not in data:
        raise RuntimeError("JPEG EOI marker not found.")
    new_data = data + append_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(new_data)


# === MAIN SCRIPT ===

def main():
    parser = argparse.ArgumentParser(
        description="Generate stego validation set based on markdown payloads."
    )
    parser.add_argument("--clean_source", type=str, default="clean",
                        help="Directory of clean source images.")
    parser.add_argument("--output_stego", type=str, default="stego",
                        help="Directory to save generated stego images.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Limit the number of clean images to use for generation.")
    args = parser.parse_args()

    source_dir = Path(args.clean_source)
    stego_dir = Path(args.output_stego)

    # Load markdown payloads from the seeds directory
    script_dir = Path(__file__).parent
    md_files = sorted(list((script_dir / "seeds").glob("*.md")))
    if not md_files:
        logging.error(f"No markdown payload files found in {script_dir}")
        sys.exit(1)
    logging.info(f"Found {len(md_files)} markdown payloads.")

    if not source_dir.is_dir():
        logging.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    stego_dir.mkdir(parents=True, exist_ok=True)

    # Filter and verify source images
    logging.info("Verifying source images...")
    source_images_unverified = [p for p in source_dir.iterdir() if p.is_file()]
    source_images = []
    for path in source_images_unverified:
        if path.name == '.DS_Store':
            continue
        try:
            with Image.open(path) as img:
                img.verify()  # Fast check for corruption
            source_images.append(path)
        except Exception as e:
            logging.warning(f"Skipping non-image or corrupt file: {path.name} ({e})")
    
    logging.info(f"Found {len(source_images)} valid clean images.")

    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        source_images = source_images[:args.limit]
        logging.info(f"Using a limited set of {len(source_images)} clean images.")

    # Pre-load payloads
    payloads = {}
    for md_file in md_files:
        try:
            payloads[md_file] = md_file.read_bytes()
        except Exception as e:
            logging.warning(f"Could not read payload file {md_file.name}, skipping: {e}")

    # Define which algorithms are suitable for which source file extensions
    algo_to_suitable_exts = {
        "eoi": {".jpg", ".jpeg"},
        "exif": {".jpg", ".jpeg"},
        "alpha": {".png"},
        "palette": {".gif", ".bmp"},
        "lsb": {".png", ".webp", ".bmp"}
    }

    # Map algorithm names to functions
    algorithms = {
        "lsb": embed_lsb, "alpha": embed_alpha, "palette": embed_palette,
        "exif": embed_exif, "eoi": embed_eoi
    }

    # Create a list of all generation tasks
    tasks = list(itertools.product(payloads.items(), algorithms.items(), source_images))
    if not tasks:
        logging.info("No tasks to perform. Exiting.")
        return

    logging.info(f"Preparing to generate {len(tasks)} stego images...")

    total_generated_count = 0
    image_indices = {}  # To track index for each payload-algorithm pair

    for (md_file, payload_content), (algo_name, embed_func), clean_path in tqdm(tasks, desc="Generating stego images", unit="image"):
        try:
            # === NEW: Enforce format alignment ===
            source_ext = clean_path.suffix.lower()
            suitable_exts = algo_to_suitable_exts.get(algo_name)
            if suitable_exts and source_ext not in suitable_exts:
                continue # Skip this combination as it's not format-aligned

            payload_name = md_file.stem

            # Get and increment index for filename
            index_key = (payload_name, algo_name)
            image_index = image_indices.get(index_key, 0)

            cover_img = Image.open(clean_path)
            
            # This check is now redundant due to the suitability mapping, but kept for safety
            if cover_img.mode not in ['RGB', 'RGBA', 'P']:
                cover_img = cover_img.convert('RGB')

            stego_img = embed_func(cover_img.copy(), payload_content)

            # === NEW: Preserve the original file format ===
            output_format = source_ext

            # Format output filename
            stego_filename = f"{payload_name}_{algo_name}_{image_index:03d}{output_format}"
            stego_path = stego_dir / stego_filename

            # Save the stego image
            if algo_name == "exif":
                save_jpeg_with_exif(stego_img, stego_path, stego_img.info.get("exif_bytes"))
            elif algo_name == "eoi":
                save_jpeg_with_eoi_append(stego_img, stego_path, stego_img.info.get("eoi_append", b""))
            else:
                save_image(stego_img, stego_path)

            # Create and save the JSON sidecar
            json_path = stego_path.with_suffix(stego_path.suffix + '.json')
            embedding_data = {}
            if algo_name == 'alpha':
                embedding_data = {"category": "pixel", "technique": "alpha", "ai42": True}
            elif algo_name == 'palette':
                # NOTE: This generator uses MSB-first for palette, contrary to spec v2.0. Documenting actual behavior.
                embedding_data = {"category": "pixel", "technique": "palette", "ai42": False, "bit_order": "msb-first"}
            elif algo_name == 'lsb':
                embedding_data = {"category": "pixel", "technique": "lsb.rgb", "ai42": False, "bit_order": "msb-first"}
            elif algo_name == 'exif':
                embedding_data = {"category": "metadata", "technique": "exif", "ai42": False}
            elif algo_name == 'eoi':
                embedding_data = {"category": "eoi", "technique": "raw", "ai42": False}

            if embedding_data:
                sidecar_content = {
                    "embedding": embedding_data,
                    "clean_file": clean_path.name
                }
                with open(json_path, 'w') as f:
                    json.dump(sidecar_content, f, indent=2)

            image_indices[index_key] = image_index + 1
            total_generated_count += 1

        except Exception as e:
            logging.warning(f"Skipped: {clean_path.name} with {md_file.name} ({algo_name}) - {e}")

    logging.info(f"\nGeneration complete. Total stego images created: {total_generated_count}")


if __name__ == "__main__":
    main()
