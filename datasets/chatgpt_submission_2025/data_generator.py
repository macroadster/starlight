import os
import hashlib
import numpy as np
from PIL import Image
import yaml
from datetime import datetime, UTC

def embed_lsb(image_array, payload_bits):
    """
    Very simple LSB embedding (for demonstration only).
    """
    flat = image_array.flatten()
    for i, bit in enumerate(payload_bits):
        if i >= len(flat):
            break
        flat[i] = (flat[i] & 0xFE) | bit  # safe mask
    return flat.reshape(image_array.shape)

def extract_lsb(stego_array, num_bits):
    """
    Extract num_bits from stego image's least significant bits.
    """
    flat = stego_array.flatten()
    return [int(x & 1) for x in flat[:num_bits]]

def bits_to_text(bits):
    chars = []
    for b in range(0, len(bits), 8):
        byte = bits[b:b+8]
        if len(byte) < 8:
            break
        chars.append(chr(int("".join(map(str, byte)), 2)))
    return "".join(chars)

def text_to_bits(text):
    return [int(b) for char in text for b in format(ord(char), "08b")]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_images(num_images=10, sanity_check=True):
    clean_dir = "./clean"
    stego_dir = "./stego"
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)

    seed_file = "sample_seed.md"

    # Load seed markdown as payload
    with open(seed_file, "r", encoding="utf-8") as f:
        payload_text = f.read()
    payload_bits = text_to_bits(payload_text)

    manifest = {
        "seed_file": seed_file,
        "seed_sha256": sha256_file(seed_file),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "count": num_images,
        "images": []
    }

    for i in range(num_images):
        # Create random clean image
        img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        clean_path = f"{clean_dir}/cover_{i:03d}.png"
        Image.fromarray(img_array).save(clean_path, "PNG")

        # Embed payload into copy
        stego_array = embed_lsb(img_array.copy(), payload_bits)
        stego_path = f"{stego_dir}/cover_{i:03d}.png"
        Image.fromarray(stego_array).save(stego_path, "PNG")

        if sanity_check:
            recovered_bits = extract_lsb(stego_array, len(payload_bits))
            recovered_text = bits_to_text(recovered_bits)
            assert recovered_text == payload_text, (
                f"Sanity check failed for {stego_path}: "
                "recovered payload does not match seed!"
            )
            print(f"[OK] Sanity check passed for {stego_path}")

        manifest["images"].append({
            "clean": clean_path,
            "stego": stego_path,
            "clean_sha256": sha256_file(clean_path),
            "stego_sha256": sha256_file(stego_path)
        })

    # Save manifest as YAML with timestamped filename
    ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    manifest_filename = f"manifest_{ts}.yaml"
    with open(manifest_filename, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, sort_keys=False)

    print(f"\nManifest written to {manifest_filename} with {manifest['count']} entries.")
    print(f"Seed SHA256: {manifest['seed_sha256']}")

if __name__ == "__main__":
    generate_images()  # default: 10 images

