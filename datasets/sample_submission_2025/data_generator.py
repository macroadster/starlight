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

from pathlib import Path

from typing import Callable

from PIL import Image





# --- PAYLOADS ---

ALL_MARKDOWN_CONTENT = b""



def load_all_markdown_content(root_dir: Path) -> bytes:

    """Loads all .md files from the datasets directory into a single byte string."""

    all_content = b""

    # Use a consistent set of markdown files for deterministic payloads

    md_files = sorted(list(root_dir.glob("datasets/**/*.md")))

    if not md_files:

        logging.error("No markdown payload files found. Exiting.")

        sys.exit(1)

    

    logging.info(f"Loading {len(md_files)} markdown files for payloads...")

    for md_file in md_files:

        try:

            all_content += md_file.read_bytes() + b"\n\n"

        except Exception as e:

            logging.warning(f"Could not read payload file {md_file}: {e}")

    

    return all_content



def get_text_payload(num_bytes: int) -> bytes:

    """Gets a random chunk of text from the global markdown content."""

    if not ALL_MARKDOWN_CONTENT or num_bytes == 0:

        return b""

    

    # To avoid always starting at the beginning, pick a random start point

    start_index = random.randint(0, len(ALL_MARKDOWN_CONTENT) - 1)

    

    # Get a chunk of text, wrapping around if necessary

    chunk = (ALL_MARKDOWN_CONTENT[start_index:] + ALL_MARKDOWN_CONTENT[:start_index])[:num_bytes]

    return chunk





try:

    import piexif

except ImportError:

    piexif = None


def embed_alpha(cover: Image.Image) -> Image.Image:
    """
    Embed in alpha channel using LSB-first (byte-reversed) bit order.
    SPEC: AI42 (LSB-first) + Message (LSB-first) + 0x00 (LSB-first)
    Dynamically generates a random payload to fill 50% of capacity.
    """
    img = cover.convert("RGBA")
    
    pixels = list(img.getdata())
    max_alpha_bits = len(pixels) # Each pixel has 1 alpha bit
    
    # Generate payload from markdown to fill 50% of capacity
    payload_bytes_to_embed = int((max_alpha_bits * 0.5) // 8)
    payload = get_text_payload(payload_bytes_to_embed)

    prefix = b"AI42"
    full_payload = prefix + payload + b" \x00"
    
    if len(full_payload) * 8 > len(pixels):
        # This should ideally not happen with 50% fill, but good to keep
        raise ValueError("Payload too large for alpha embedding.")
    
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


def embed_lsb(cover: Image.Image) -> Image.Image:
    """
    Embed in RGB channels using MSB-first bit order.
    SPEC: Generic LSB, null-terminated, no hint. Always produces an RGB image.
    Dynamically generates a random payload to fill 50% of capacity.
    """
    img = cover.convert("RGB")  # Always convert to RGB
    pixels = list(img.getdata())
    
    max_lsb_bits = len(pixels) * 3 # Each pixel has 3 RGB bits
    
    # Generate payload from markdown to fill 50% of capacity
    payload_bytes_to_embed = int((max_lsb_bits * 0.5) // 8)
    payload = get_text_payload(payload_bytes_to_embed)

    # MSB-first encoding with null terminator
    bits = "".join(f"{b:08b}" for b in payload) + "00000000"
    
    if len(bits) > max_lsb_bits:
        # This should ideally not happen with 50% fill, but good to keep
        raise ValueError("Payload too large for LSB embedding.")
    
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

def embed_palette(cover: Image.Image) -> Image.Image:
    """
    Embed in palette indices using MSB-first bit order.
    SPEC: MSB-first, null-terminated. Dynamically generates a random payload to fill 50% of capacity.
    """
    if cover.mode != 'P':
        img = cover.convert("P", palette=Image.ADAPTIVE, colors=256)
    else:
        img = cover.copy()

    if img.palette is None:
        raise ValueError("Image must have a palette for this technique.")

    indices = list(img.getdata())
    max_palette_bits = len(indices) # Each index is 1 bit for LSB

    # Generate payload from markdown to fill 50% of capacity
    payload_bytes_to_embed = int((max_palette_bits * 0.5) // 8)
    payload = get_text_payload(payload_bytes_to_embed)

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


def embed_exif(cover: Image.Image) -> Image.Image:
    """
    Embed in EXIF UserComment with ASCII encoding header.
    SPEC: ASCII\x00\x00\x00 prefix + message. Dynamically generates a random payload.
    """
    if piexif is None:
        raise RuntimeError("piexif not installed; cannot perform EXIF embedding.")
    
    img = cover.convert("RGB")
    
    # Generate a text payload. EXIF capacity is not directly image-size dependent,
    # so a fixed, reasonably large size (e.g., 1KB) is used.
    payload_size = 1024 # 1KB payload
    payload = get_text_payload(payload_size)

    # SPEC: Use ASCII encoding header
    exif_dict = {
        "0th": {},
        "Exif": {
            piexif.ExifIFD.UserComment: b"ASCII\x00\x00\x00" + payload
        },
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }
    
    img.info["exif_bytes"] = piexif.dump(exif_dict)
    return img


def embed_eoi(cover: Image.Image) -> Image.Image:
    """
    Embed after JPEG EOI marker.
    SPEC: Raw append after EOI marker (may have hint prefix). Dynamically generates a random payload.
    """
    img = cover.convert("RGB")
    
    # Generate a text payload. EOI capacity is not directly image-size dependent,
    # so a fixed, reasonably large size (e.g., 1KB) is used.
    payload_size = 1024 # 1KB payload
    payload = get_text_payload(payload_size)

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
    global ALL_MARKDOWN_CONTENT
    parser = argparse.ArgumentParser(
        description="Generate stego validation set. Creates all algorithm types for each clean image."
    )
    parser.add_argument("--clean_source", type=str, default="clean",
                        help="Directory of clean source images.")
    parser.add_argument("--output_stego", type=str, default="stego",
                        help="Directory to save generated stego images.")
    args = parser.parse_args()

    source_dir = Path(args.clean_source)
    stego_dir = Path(args.output_stego)

    # Load markdown content for payloads
    project_root = Path(__file__).parent.parent.parent 
    ALL_MARKDOWN_CONTENT = load_all_markdown_content(project_root)

    if not source_dir.is_dir():
        logging.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    stego_dir.mkdir(parents=True, exist_ok=True)

    # Define canonical output formats. LSB/Alpha are handled dynamically.
    algo_to_format = {
        "palette": ".bmp",
        "exif": ".jpg",
        "eoi": ".jpg"
    }

    # Map algorithm names to functions
    algorithms = {
        "lsb": embed_lsb,
        "alpha": embed_alpha,
        "palette": embed_palette,
        "exif": embed_exif,
        "eoi": embed_eoi
    }

    source_images = [p for p in source_dir.iterdir() if p.is_file()]
    logging.info(f"Found {len(source_images)} clean images in {source_dir}")
    count = 0

    for clean_path in source_images:
        try:
            cover_img = Image.open(clean_path)
            # Ensure image is in a base mode that works for most conversions
            if cover_img.mode not in ['RGB', 'RGBA']:
                cover_img = cover_img.convert('RGB')
        except Exception as e:
            logging.warning(f"Skipping non-image or corrupt file: {clean_path.name} ({e})")
            continue

        # For each clean image, generate all stego types
        for algo_name, embed_func in algorithms.items():
            try:
                # Use a fresh copy of the image for each algorithm
                img_copy = cover_img.copy()

                stego_img = embed_func(img_copy)

                # For pixel-based algos, use original format if lossless, else PNG
                if algo_name in ['lsb', 'alpha']:
                    lossless_formats = ['.png', '.bmp', '.gif', '.tiff', '.webp']
                    if clean_path.suffix.lower() in lossless_formats:
                        output_format = clean_path.suffix
                    else:
                        output_format = '.png'  # Default for lossy originals
                else:
                    output_format = algo_to_format[algo_name]

                stego_filename = f"{clean_path.stem}_{algo_name}{output_format}"
                stego_path = stego_dir / stego_filename

                if algo_name == "exif":
                    save_jpeg_with_exif(stego_img, stego_path,
                                        stego_img.info.get("exif_bytes"))
                elif algo_name == "eoi":
                    save_jpeg_with_eoi_append(stego_img, stego_path,
                                              stego_img.info.get("eoi_append", b""))
                else:
                    save_image(stego_img, stego_path)

                logging.info(f"Generated: {stego_path.name}")
                count += 1

            except Exception as e:
                logging.warning(f"Failed to generate {algo_name} for {clean_path.name}: {e}")

    logging.info(f"\nGeneration complete. Total stego images created: {count}")


if __name__ == "__main__":
    main()
