import os
import glob
import hashlib
import numpy as np
from PIL import Image
import yaml
from datetime import datetime, UTC

def embed_lsb(image_array, payload_bits):
    """Very simple LSB embedding (for demonstration only)."""
    flat = image_array.flatten()
    for i, bit in enumerate(payload_bits):
        if i >= len(flat):
            break
        flat[i] = (flat[i] & 0xFE) | bit
    return flat.reshape(image_array.shape)

def extract_lsb(stego_array, num_bits):
    flat = stego_array.flatten()
    return [int(x & 1) for x in flat[:num_bits]]

def text_to_bits(text):
    return [int(b) for byte in text.encode("utf-8") for b in format(byte, "08b")]

def bits_to_text(bits):
    data = bytearray()
    for b in range(0, len(bits), 8):
        byte = bits[b:b+8]
        if len(byte) < 8:
            break
        data.append(int("".join(map(str, byte)), 2))
    return data.decode("utf-8", errors="ignore")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_images(sanity_check=True):
    clean_dir = "./clean"
    stego_dir = "./stego"
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)

    # find all .md seed files
    seed_files = glob.glob("*.md")
    if not seed_files:
        print("[WARN] No markdown seed files found. Exiting.")
        return

    manifest = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "seeds": []
    }

    for seed_file in seed_files:
        with open(seed_file, "r", encoding="utf-8") as f:
            payload_text = f.read()
        payload_bits = text_to_bits(payload_text)



        seed_label = os.path.splitext(os.path.basename(seed_file))[0]

        # Create random clean image
        img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        # --- Capacity check (UTF-8 aware) ---
        # One bit per channel per pixel, so total bits = array.size
        max_capacity = img_array.size  # total pixels × channels (1 bit per channel)
        if len(payload_bits) > max_capacity:
            print(f"[WARN] Payload from {seed_file} is too large "
                  f"({len(payload_bits)} bits needed > {max_capacity} bits available). "
                  "Truncating.")
            payload_bits = payload_bits[:max_capacity]

        # Save clean images
        clean_png = os.path.join(clean_dir, f"{seed_label}.png")
        clean_jpg = os.path.join(clean_dir, f"{seed_label}.jpeg")
        Image.fromarray(img_array).save(clean_png, "PNG")
        Image.fromarray(img_array).save(clean_jpg, "JPEG", quality=85)

        # Stego image
        stego_array = embed_lsb(img_array.copy(), payload_bits)
        stego_png = os.path.join(stego_dir, f"{seed_label}.png")
        stego_jpg = os.path.join(stego_dir, f"{seed_label}.jpeg")
        Image.fromarray(stego_array).save(stego_png, "PNG")
        Image.fromarray(stego_array).save(stego_jpg, "JPEG", quality=85)

        if sanity_check:
            recovered_bits = extract_lsb(stego_array, len(payload_bits))
            recovered_text = bits_to_text(recovered_bits)
            assert recovered_text == payload_text, (
                f"Sanity check failed for {seed_label}"
            )
            print(f"[OK] Sanity check passed for {seed_label}")

        manifest["seeds"].append({
            "seed_file": seed_file,
            "seed_sha256": sha256_file(seed_file),
            "clean_png": clean_png,
            "clean_jpeg": clean_jpg,
            "stego_png": stego_png,
            "stego_jpeg": stego_jpg,
            "clean_png_sha256": sha256_file(clean_png),
            "clean_jpeg_sha256": sha256_file(clean_jpg),
            "stego_png_sha256": sha256_file(stego_png),
            "stego_jpeg_sha256": sha256_file(stego_jpg),
        })

    # Save manifest
    ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    manifest_filename = f"manifest_{ts}.yaml"
    with open(manifest_filename, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, sort_keys=False)

    print(f"\nManifest written to {manifest_filename}")
    for s in manifest["seeds"]:
        print(f"  - {s['seed_file']} → {s['stego_png']} / {s['stego_jpeg']}")

if __name__ == "__main__":
    generate_images()

