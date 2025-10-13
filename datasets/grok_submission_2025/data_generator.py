import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar for large batches
import piexif  # For EXIF metadata; pip install piexif
import argparse  # For command-line arguments

def generate_clean_image(output_path, size=(512, 512), seed=None, format='JPEG'):
    """
    Generate a 512x512 clean image (JPEG or PNG) with a colorful gradient pattern.
    Uses a seed for reproducible diversity across images in a batch.
    Args:
        output_path (str): Path to save (e.g., ./clean/sample_seed_exif_001.jpeg).
        size (tuple): Image dimensions (default: 512x512).
        seed (int): Random seed for noise variation (ensures unique patterns).
        format (str): 'JPEG' (Q=85) or 'PNG' (lossless; better for LSB stego).
    """
    try:
        if seed is not None:
            np.random.seed(seed)  # Ensure unique patterns per image
        x, y = np.meshgrid(np.linspace(0, 1, size[0]), np.linspace(0, 1, size[1]))
        r = (np.sin(2 * np.pi * x * 5) + np.cos(2 * np.pi * y * 3)) * 127.5 + 127.5
        g = (np.sin(2 * np.pi * x * 3) + np.cos(2 * np.pi * y * 5)) * 127.5 + 127.5
        b = (np.sin(2 * np.pi * x * 4) + np.cos(2 * np.pi * y * 4)) * 127.5 + 127.5
        noise = np.random.normal(0, 10, size)
        img_array = np.stack([np.clip(r + noise, 0, 255), np.clip(g + noise, 0, 255), np.clip(b + noise, 0, 255)], axis=-1).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(img_array)
        if format == 'JPEG':
            img.save(output_path, 'JPEG', quality=85)  # Q=85
        else:  # PNG
            img.save(output_path, 'PNG', compress_level=0)  # Lossless
        print(f"Clean image saved to: {output_path} ({format})")
    except Exception as e:
        print(f"Error generating clean image: {str(e)}")

def add_exif_metadata(img_path, payload):
    """
    Add payload to JPEG EXIF UserComment field.
    Args:
        img_path (str): Path to JPEG image.
        payload (str): Markdown text to store (up to ~65 KB).
    """
    try:
        # Encode payload as ASCII (EXIF UserComment requires 8-byte header)
        exif_dict = {"Exif": {piexif.ExifIFD.UserComment: b"ASCII\0\0\0" + payload.encode('ascii')[:65527]}}
        exif_bytes = piexif.dump(exif_dict)
        img = Image.open(img_path)
        img.save(img_path, 'JPEG', quality=85, exif=exif_bytes)
        print(f"Added EXIF UserComment to {img_path} ({len(payload)} chars)")
    except Exception as e:
        print(f"Error adding EXIF metadata to {img_path}: {str(e)}")

def embed_lsb(clean_img_path, stego_img_path, payload, payload_size=0.2):
    """
    Embed a payload into a PNG image using LSB (Least Significant Bit) technique.
    LSB replaces the LSB of pixel channels with payload bits—simple, keyless.
    Args:
        clean_img_path (str): Path to clean PNG.
        stego_img_path (str): Path to save stego PNG.
        payload (str or None): Text from markdown seed to embed (None for random).
        payload_size (float): Bits per pixel (fixed: 0.2 bpnzAC approximation).
    """
    try:
        img = Image.open(clean_img_path).convert('RGB')
        img_array = np.array(img)

        # Prepare payload (random if None or empty)
        if payload and isinstance(payload, str) and len(payload) > 0:
            print(f"Embedding provided payload: {len(payload)} chars")
            binary_payload = ''.join(format(ord(c), '08b') for c in payload)
        else:
            print("No payload provided; generating random payload")
            total_bits = int(payload_size * img_array.size)
            binary_payload = ''.join(map(str, np.random.randint(0, 2, total_bits)))
        
        total_bits = int(payload_size * img_array.size)
        binary_payload = binary_payload[:total_bits]

        flat = img_array.flatten()
        if len(binary_payload) > len(flat):
            raise ValueError(f"Payload size ({len(binary_payload)} bits) exceeds image capacity ({len(flat)} bits).")

        for i in range(len(binary_payload)):
            flat[i] = (flat[i] & 0xFE) | int(binary_payload[i])

        stego_array = flat.reshape(img_array.shape)
        stego_img = Image.fromarray(stego_array)
        stego_img.save(stego_img_path, 'PNG', compress_level=0)
        print(f"Generated LSB stego image: {stego_img_path} (PNG)")
    except Exception as e:
        print(f"Error with LSB embedding: {str(e)}")

def extract_lsb(stego_img_path, payload_size=0.2):
    """
    Extract payload from a PNG LSB stego image.
    Reads the least significant bits to reconstruct the embedded binary data and converts to text.
    Args:
        stego_img_path (str): Path to stego PNG.
        payload_size (float): Bits per pixel used during embedding (fixed: 0.2).
    Returns:
        str: Extracted text payload (or empty string if extraction fails).
    """
    try:
        img = Image.open(stego_img_path).convert('RGB')
        img_array = np.array(img)
        flat = img_array.flatten()
        total_bits = int(payload_size * img_array.size)

        # Extract LSBs
        binary_payload = ''.join(str(flat[i] & 1) for i in range(min(total_bits, len(flat))))
        
        # Decode all bytes without stopping
        extracted_text = ''
        for i in range(0, len(binary_payload) - 7, 8):
            byte = binary_payload[i:i+8]
            if len(byte) == 8:
                char_code = int(byte, 2)
                extracted_text += chr(char_code)  # No range check or break
        return extracted_text
    except Exception as e:
        print(f"Error extracting LSB payload from {stego_img_path}: {str(e)}")
        return ""

def verify_exif_metadata(img_path, expected_payload):
    """
    Verify that JPEG EXIF UserComment matches the expected payload.
    Args:
        img_path (str): Path to JPEG stego image.
        expected_payload (str): Original markdown content.
    """
    try:
        exif_dict = piexif.load(img_path)
        user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        extracted = user_comment[8:].decode('ascii') if len(user_comment) > 8 else ""
        if extracted == expected_payload[:65527]:  # Truncate to EXIF limit
            print(f"EXIF verification passed: {img_path} matches payload.")
        else:
            print(f"EXIF verification failed: {img_path} does not match payload.")
            print(f"Expected (truncated): {expected_payload[:50]}...")
            print(f"Extracted: {extracted[:50]}...")
    except Exception as e:
        print(f"Error verifying EXIF metadata for {img_path}: {str(e)}")

def verify_images(seed_file, stego_paths, seed_payload, format, payload_size=0.2):
    """
    Verify that stego images contain the correct payload.
    - PNG: Verifies LSB-extracted payload (truncated to ~6.5 KB).
    - JPEG: Verifies EXIF UserComment (up to ~65 KB).
    Args:
        seed_file (str): Name of the seed file (e.g., 'sample_seed.md').
        stego_paths (list): List of stego image paths to verify.
        seed_payload (str): Original markdown content.
        format (str): 'JPEG' (EXIF) or 'PNG' (LSB).
        payload_size (float): Bits per pixel for LSB (fixed: 0.2).
    """
    print(f"Verifying {format} stego images for {seed_file}...")
    for stego_path in stego_paths:
        if format == 'PNG':
            max_chars = int(payload_size * 512 * 512 * 3 / 8)  # Corrected capacity
            extracted = extract_lsb(stego_path, payload_size)
            seed_truncated = seed_payload[:max_chars]
            if extracted.startswith(seed_truncated):
                print(f"LSB verification passed: {stego_path} matches {seed_file} payload.")
            else:
                print(f"LSB verification failed: {stego_path} does not match {seed_file} payload.")
                print(f"Expected (truncated): {seed_truncated[:50]}...")
                print(f"Extracted: {extracted[:50]}...")
        else:  # JPEG
            verify_exif_metadata(stego_path, seed_payload)

def generate_images(num_images=1, formats=['JPEG', 'PNG']):
    """
    Generate clean and stego images for Project Starlight (Option 3).
    Hardcoded relative paths (run from dataset/grok_submission_2025/):
        - Clean images: ./clean/
        - Stego images: ./stego/
        - Markdown seeds: ./ (all .md files)
        - Payload size: 0.2 bpnzAC (for PNG LSB; JPEG uses EXIF UserComment)
        - JPEG quality: 85 (fixed; within 75-95)
        - Formats: List of 'JPEG' (EXIF metadata) or 'PNG' (LSB)
        - Stego key: None (keyless LSB/EXIF)
    Behavior:
        - Iterates over each .md file, generating a batch of num_images clean + stego pairs per format.
        - Labels images with seed basename and algorithm (e.g., sample_seed_exif_001.jpeg).
        - If no seeds, generates a random batch labeled 'random' (no payload for verification).
        - Verifies payloads: PNG (LSB extraction), JPEG (EXIF UserComment).
        - Uses tqdm for progress feedback.
    Args:
        num_images (int): Pairs per seed batch (default: 1; override via --limit or NUM_IMAGES env var).
        formats (list): List of formats to generate: 'JPEG' (EXIF metadata) or 'PNG' (LSB).
    """
    clean_dir = "./clean"
    stego_dir = "./stego"
    seed_dir = "./"
    payload_size = 0.2  # Fixed: 0.2 bpnzAC for PNG LSB
    quality = 85  # Fixed: JPEG quality

    num_images = int(os.environ.get('NUM_IMAGES', num_images))
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    
    # Validate
    if 'JPEG' in formats and not 75 <= quality <= 95:
        raise ValueError("JPEG quality must be 75-95.")
    for fmt in formats:
        if fmt not in ['JPEG', 'PNG']:
            raise ValueError("Formats must be 'JPEG' or 'PNG'.")

    # Identify all .md seed files
    seed_files = [f for f in os.listdir(seed_dir) if f.endswith('.md')]
    if not seed_files:
        print("No .md seeds found; generating random batch.")
        seed_basename = "random"
        for format in formats:
            algorithm = 'exif' if format == 'JPEG' else 'lsb'  # Algorithm name for filename
            stego_paths = []
            for i in tqdm(range(num_images), desc=f"Batch (random, {format})"):
                img_name = f"{seed_basename}_{algorithm}_{i:03d}.{format.lower()}"
                clean_path = os.path.join(clean_dir, img_name)
                stego_path = os.path.join(stego_dir, img_name)
                generate_clean_image(clean_path, seed=i, format=format)
                if format == 'PNG':
                    embed_lsb(clean_path, stego_path, payload=None, payload_size=payload_size)
                    stego_paths.append(stego_path)
                else:  # JPEG
                    generate_clean_image(stego_path, seed=i, format=format)  # Copy clean to stego
                    add_exif_metadata(stego_path, "")
                    stego_paths.append(stego_path)
            verify_images("random", stego_paths, "", format, payload_size)
    else:
        for seed_file in seed_files:
            seed_basename = os.path.splitext(seed_file)[0]
            with open(os.path.join(seed_dir, seed_file), 'r', encoding='utf-8') as f:
                seed_payload = f.read()
            for format in formats:
                algorithm = 'exif' if format == 'JPEG' else 'lsb'  # Algorithm name for filename
                stego_paths = []
                for i in tqdm(range(num_images), desc=f"Batch ({seed_basename}, {format})"):
                    img_name = f"{seed_basename}_{algorithm}_{i:03d}.{format.lower()}"
                    clean_path = os.path.join(clean_dir, img_name)
                    stego_path = os.path.join(stego_dir, img_name)
                    generate_clean_image(clean_path, seed=i, format=format)
                    if format == 'PNG':
                        embed_lsb(clean_path, stego_path, payload=seed_payload, payload_size=payload_size)
                        stego_paths.append(stego_path)
                    else:  # JPEG
                        generate_clean_image(stego_path, seed=i, format=format)  # Copy clean to stego
                        add_exif_metadata(stego_path, seed_payload)
                        stego_paths.append(stego_path)
                verify_images(seed_file, stego_paths, seed_payload, format, payload_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clean and stego images for Project Starlight.")
    parser.add_argument('--limit', type=int, default=1, help='Number of images to generate per payload per algorithm (overrides NUM_IMAGES env var).')
    parser.add_argument('--formats', type=str, default='JPEG,PNG', help='Comma-separated formats to use: JPEG (exif), PNG (lsb).')
    args = parser.parse_args()

    formats = [f.strip().upper() for f in args.formats.split(',')]

    try:
        generate_images(num_images=args.limit, formats=formats)
        print("Image generation completed.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Notes:
# - JPEG: Stores payload in EXIF UserComment (~65 KB capacity, keyless).
# - PNG: Embeds payload via LSB (0.2 bpnzAC, ~6.5 KB for 512x512, keyless).
# - Verification: PNG (LSB extraction), JPEG (EXIF UserComment).
# - Override: python data_generator.py --limit 10 --formats PNG
# - Dependencies: pip install piexif Pillow numpy tqdm
# - Ensure Python 3.8+ for compatibility.
