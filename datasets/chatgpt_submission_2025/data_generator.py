#!/usr/bin/env python3
"""
data_generator.py

Generates for each *.md seed (UTF-8):
  - clean/<seed>.png   (lossless cover)
  - stego/<seed>.png   (PNG with LSB embedding)
  - clean/<seed>.gif   (palette GIF cover)
  - stego/<seed>.gif   (GIF with LSB embedding on palette indices)
  - clean/<seed>.webp  (lossless WebP cover)
  - stego/<seed>.webp  (lossless WebP with LSB embedding)

Behavior:
 - Strict verification by reloading files from disk and verifying payload bytes via SHA-256.
 - Fail-fast: on first error the script prints [FATAL] and exits with code 2.
 - Manifest written as manifest_<timestamp>.yaml with no plaintext payloads.
"""
import os
import glob
import sys
import hashlib
import numpy as np
from PIL import Image
import yaml
from datetime import datetime, timezone

# ---------------- utilities ----------------

def bytes_to_bits(bts: bytes):
    """Return list of bits (MSB-first within each byte)."""
    out = []
    for byte in bts:
        out.extend([int(ch) for ch in format(byte, "08b")])
    return out

def bits_to_bytes(bits):
    """Convert list of bits back to bytes, ignoring trailing incomplete byte."""
    out = bytearray()
    for i in range(0, len(bits) - (len(bits) % 8), 8):
        out.append(int("".join(map(str, bits[i:i+8])), 2))
    return bytes(out)

def payload_sha256_bytes(payload_bytes):
    h = hashlib.sha256()
    h.update(payload_bytes)
    return h.hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------- LSB helpers (generic) ----------------

def embed_lsb_image(image_array: np.ndarray, payload_bits: list):
    """
    Embed payload_bits into image_array least-significant bits.
    image_array: ndarray of dtype uint8, any shape (H,W) or (H,W,C)
    Returns a new ndarray with embedding applied (same dtype & shape).
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be numpy.ndarray")
    flat = image_array.flatten().astype(np.uint8)
    n = min(len(payload_bits), flat.size)
    if n == 0:
        return image_array.copy()
    out_flat = flat.copy()
    for i in range(n):
        out_flat[i] = (out_flat[i] & 0xFE) | int(payload_bits[i])
    return out_flat.reshape(image_array.shape)

def extract_lsb_from_image(image_array: np.ndarray, num_bits: int):
    flat = image_array.flatten()
    num_bits = min(num_bits, flat.size)
    return [int(x & 1) for x in flat[:num_bits]]

# ---------------- main ----------------

def generate_images():
    clean_dir = "./clean"
    stego_dir = "./stego"
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)

    seed_files = sorted(glob.glob("*.md"))
    if not seed_files:
        print("[FATAL] No markdown seed files found in current directory.")
        sys.exit(1)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seeds": []
    }

    for seed_file in seed_files:
        seed_label = os.path.splitext(os.path.basename(seed_file))[0]
        try:
            # Read payload as bytes
            with open(seed_file, "r", encoding="utf-8") as f:
                payload_text = f.read()
            payload_bytes = payload_text.encode("utf-8")
            payload_bits = bytes_to_bits(payload_bytes)
            nbits = len(payload_bits)

            # Create a random 512x512 RGB cover
            H, W = 512, 512
            img_array = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

            # Compute capacities for each format (bits)
            cap_png = img_array.size  # H*W*3
            # GIF will be a palette image (single channel indices)
            # Convert to palette to get indices array shape (same HxW)
            gif_image = Image.fromarray(img_array).convert("P", palette=Image.ADAPTIVE, colors=256)
            gif_array = np.array(gif_image)
            cap_gif = gif_array.size  # H*W
            # WebP lossless works on full RGB
            cap_webp = img_array.size

            min_capacity = min(cap_png, cap_gif, cap_webp)
            if nbits > min_capacity:
                raise RuntimeError(
                    f"Payload from {seed_file} requires {nbits} bits but only {min_capacity} bits are available (min across PNG/GIF/WebP)."
                )

            # ---------- PNG (lossless) ----------
            clean_png = os.path.join(clean_dir, f"{seed_label}.png")
            stego_png = os.path.join(stego_dir, f"{seed_label}.png")
            Image.fromarray(img_array).save(clean_png, "PNG")

            stego_png_array = embed_lsb_image(img_array.copy(), payload_bits)
            Image.fromarray(stego_png_array).save(stego_png, "PNG")

            # Verify PNG by reloading and extracting
            with Image.open(stego_png) as im:
                re_png = np.array(im.convert("RGB"), dtype=np.uint8)
            rec_bits_png = extract_lsb_from_image(re_png, nbits)
            rec_bytes_png = bits_to_bytes(rec_bits_png)
            if payload_sha256_bytes(rec_bytes_png) != payload_sha256_bytes(payload_bytes):
                raise RuntimeError(f"PNG-LSB payload SHA mismatch for {seed_label}")
            changed_png = int((img_array != re_png).sum())
            if nbits > 0 and changed_png == 0:
                raise RuntimeError(f"No pixel changes detected after PNG embedding for {seed_label}")
            print(f"[OK] PNG LSB validated for {seed_label} (bits={nbits}, changed_pixels={changed_png})")

            # ---------- GIF (palette indexes) ----------
            clean_gif = os.path.join(clean_dir, f"{seed_label}.gif")
            stego_gif = os.path.join(stego_dir, f"{seed_label}.gif")

            # Save clean palette GIF (use adaptive palette)
            # Keep the palette object for later reuse so colors remain consistent
            gif_image.save(clean_gif, "GIF", optimize=False)

            # Embed into the palette index array (2D)
            stego_gif_array = embed_lsb_image(gif_array.copy(), payload_bits)

            # Recreate a palette image with the original palette to preserve colors
            im_stego_gif = Image.fromarray(stego_gif_array.astype(np.uint8), mode="P")
            palette = gif_image.getpalette()
            if palette is not None:
                im_stego_gif.putpalette(palette)
            # Save stego GIF
            im_stego_gif.save(stego_gif, "GIF", optimize=False)

            # Verify GIF by reloading and extracting (strict check)
            gif_check = Image.open(stego_gif).convert("P")
            re_gif = np.array(gif_check, dtype=np.uint8)
            rec_bits_gif = extract_lsb_from_image(re_gif, nbits)
            rec_bytes_gif = bits_to_bytes(rec_bits_gif)
            if payload_sha256_bytes(rec_bytes_gif) != payload_sha256_bytes(payload_bytes):
                raise RuntimeError(f"GIF-LSB payload SHA mismatch for {seed_label}")
            changed_gif = int((gif_array != re_gif).sum())
            if nbits > 0 and changed_gif == 0:
                raise RuntimeError(f"No index changes detected after GIF embedding for {seed_label}")
            print(f"[OK] GIF LSB validated for {seed_label} (bits={nbits}, changed_indices={changed_gif})")

            # ---------- WebP (lossless) ----------
            clean_webp = os.path.join(clean_dir, f"{seed_label}.webp")
            stego_webp = os.path.join(stego_dir, f"{seed_label}.webp")

            # Save lossless WebP cover
            # Pillow may or may not accept lossless kwarg depending on build; capture errors
            try:
                Image.fromarray(img_array).save(clean_webp, "WEBP", lossless=True)
            except TypeError:
                # Fall back to quality=100 if lossless not supported in this Pillow build
                Image.fromarray(img_array).save(clean_webp, "WEBP", quality=100)

            webp_image = Image.open(clean_webp)
            webp_array = np.array(webp_image, dtype=np.uint8)
            stego_webp_array = embed_lsb_image(webp_array.copy(), payload_bits)

            # Save stego WebP (attempt lossless; fallback to high-quality)
            try:
                Image.fromarray(stego_webp_array).save(stego_webp, "WEBP", lossless=True)
            except TypeError:
                Image.fromarray(stego_webp_array).save(stego_webp, "WEBP", quality=100)

            # Verify WebP by reloading and extracting
            re_webp = np.array(Image.open(stego_webp).convert("RGB"), dtype=np.uint8)
            rec_bits_webp = extract_lsb_from_image(re_webp, nbits)
            rec_bytes_webp = bits_to_bytes(rec_bits_webp)
            if payload_sha256_bytes(rec_bytes_webp) != payload_sha256_bytes(payload_bytes):
                raise RuntimeError(f"WEBP-LSB payload SHA mismatch for {seed_label}")
            changed_webp = int((img_array != re_webp).sum())
            if nbits > 0 and changed_webp == 0:
                raise RuntimeError(f"No pixel changes detected after WebP embedding for {seed_label}")
            print(f"[OK] WEBP LSB validated for {seed_label} (bits={nbits}, changed_pixels={changed_webp})")

            # ---------- manifest entry (no plaintext) ----------
            manifest["seeds"].append({
                "seed_file": seed_file,
                "seed_sha256": sha256_file(seed_file),
                "payload_sha256": payload_sha256_bytes(payload_bytes),
                "payload_bits": nbits,
                "clean_png": clean_png,
                "stego_png": stego_png,
                "clean_gif": clean_gif,
                "stego_gif": stego_gif,
                "clean_webp": clean_webp,
                "stego_webp": stego_webp,
                "clean_png_sha256": sha256_file(clean_png),
                "stego_png_sha256": sha256_file(stego_png),
                "clean_gif_sha256": sha256_file(clean_gif),
                "stego_gif_sha256": sha256_file(stego_gif),
                "clean_webp_sha256": sha256_file(clean_webp),
                "stego_webp_sha256": sha256_file(stego_webp),
                "changed_pixels_png": changed_png,
                "changed_indices_gif": changed_gif,
                "changed_pixels_webp": changed_webp,
                "formats": ["png", "gif", "webp"]
            })

        except Exception as e:
            print(f"[FATAL] Error processing {seed_file}: {e}")
            sys.exit(2)

    # Write manifest
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_filename = f"manifest_{ts}.yaml"
    with open(manifest_filename, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, sort_keys=False)

    print(f"\nManifest written to {manifest_filename}")
    for s in manifest["seeds"]:
        print(f"  - {s['seed_file']} → png:{s['stego_png']} gif:{s['stego_gif']} webp:{s['stego_webp']} (bits={s['payload_bits']})")


if __name__ == "__main__":
    generate_images()

