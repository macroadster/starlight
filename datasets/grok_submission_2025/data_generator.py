import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar for large batches
import piexif  # For EXIF metadata; pip install piexif
import argparse  # For command-line arguments
import logging  # For logging outputs to file
import json
from pathlib import Path


def generate_clean_image(
    output_path, size=(512, 512), seed=None, format="JPEG", pattern_type="linear"
):
    """
    Generate a 512x512 clean image (JPEG or PNG) with varied patterns for diversity.
    Uses a seed for reproducible diversity across images in a batch.
    Supports 'linear' (smooth directional gradients) or 'radial' (circular gradients) for better texture variety,
    making LSB artifacts more detectable in different contexts without periodic masking.
    This eases steganalysis training by providing low-complexity but diverse backgrounds.
    Args:
        output_path (str): Path to save (e.g., ./clean/sample_seed_exif_001.jpeg).
        size (tuple): Image dimensions (default: 512x512).
        seed (int): Random seed for variation (ensures unique images).
        format (str): 'JPEG' (Q=85) or 'PNG' (lossless; better for LSB stego).
        pattern_type (str): 'linear' or 'radial' for gradient style (randomized if not specified).
    """
    try:
        if seed is not None:
            np.random.seed(seed)  # Ensure unique patterns per image

        # Randomize pattern type if not specified
        if pattern_type == "random":
            pattern_type = np.random.choice(["linear", "radial"])

        # Randomize start and end colors for each channel
        start_color = np.random.randint(0, 256, 3)
        end_color = np.random.randint(0, 256, 3)

        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))

        if pattern_type == "linear":
            # Linear: Randomized direction
            angle = np.random.uniform(0, 360)
            angle_rad = np.deg2rad(angle)
            grad = (
                np.cos(angle_rad) * x + np.sin(angle_rad) * y + 1
            ) / 2  # Normalize to [0,1]
        else:  # 'radial'
            # Radial: Distance from randomized center
            center_x = np.random.uniform(-0.5, 0.5)
            center_y = np.random.uniform(-0.5, 0.5)
            grad = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            grad = grad / np.max(grad)  # Normalize to [0,1]

        # Interpolate between start and end colors
        r = start_color[0] + (end_color[0] - start_color[0]) * grad
        g = start_color[1] + (end_color[1] - start_color[1]) * grad
        b = start_color[2] + (end_color[2] - start_color[2]) * grad

        # Add minimal noise to avoid perfect uniformity (but keep low for easier stego detection)
        noise_level = np.random.uniform(0, 1)  # Even lower noise
        noise = np.random.normal(0, noise_level, size)

        img_array = np.stack(
            [
                np.clip(r + noise, 0, 255),
                np.clip(g + noise, 0, 255),
                np.clip(b + noise, 0, 255),
            ],
            axis=-1,
        ).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(img_array)
        if format == "JPEG":
            img.save(output_path, "JPEG", quality=85)  # Q=85
        else:  # PNG
            img.save(output_path, "PNG", compress_level=0)  # Lossless
        logging.info(
            f"Clean image saved to: {output_path} ({format}, pattern: {pattern_type})"
        )
    except Exception as e:
        logging.error(f"Error generating clean image: {str(e)}")


def add_exif_metadata(img_path, payload):
    """
    Add payload to EXIF UserComment field for JPEG, PNG, WebP.
    Args:
        img_path (str): Path to image (JPEG, PNG, WebP).
        payload (str): Markdown text to store (up to ~65 KB).
    """
    try:
        # Encode payload as ASCII (EXIF UserComment requires 8-byte header)
        exif_dict = {
            "Exif": {
                piexif.ExifIFD.UserComment: b"ASCII\0\0\0"
                + payload.encode("ascii", errors="ignore")[:65527]
            }
        }
        exif_bytes = piexif.dump(exif_dict)
        img = Image.open(img_path)
        ext = Path(img_path).suffix.lower()
        if ext == ".webp":
            img.save(img_path, format="WEBP", lossless=True, exif=exif_bytes)
        elif ext == ".png":
            img.save(img_path, format="PNG", exif=exif_bytes)
        elif ext in [".jpg", ".jpeg"]:
            img.save(img_path, format="JPEG", quality=85, exif=exif_bytes)
        else:
            # Fallback
            img.save(img_path, exif=exif_bytes)
        logging.info(f"Added EXIF UserComment to {img_path} ({len(payload)} chars)")
    except Exception as e:
        logging.error(f"Error adding EXIF metadata to {img_path}: {str(e)}")

def embed_eoi(img_path, payload):
    """
    Embed payload by appending after the image data (EOI method).
    Supports JPEG, PNG, GIF, WebP.
    Args:
        img_path (str): Path to image.
        payload (str): Text to append.
    """
    try:
        with open(img_path, 'rb') as f:
            raw = f.read()
        payload_bytes = payload.encode('ascii', errors='ignore')

        # Detect format and find insertion point
        if raw.startswith(b'\xFF\xD8'):  # JPEG
            eoi_pos = raw.rfind(b'\xFF\xD9')
            if eoi_pos == -1:
                raise ValueError("No EOI marker found in JPEG")
            insert_pos = eoi_pos + 2
        elif raw.startswith(b'\x89PNG'):  # PNG
            iend_pos = raw.rfind(b'IEND')
            if iend_pos == -1:
                raise ValueError("No IEND found in PNG")
            insert_pos = iend_pos + 12  # After IEND chunk
        elif raw.startswith(b'GIF8'):  # GIF
            term_pos = raw.rfind(b';')
            if term_pos == -1:
                raise ValueError("No terminator found in GIF")
            insert_pos = term_pos + 1
        elif raw.startswith(b'RIFF') and raw[8:12] == b'WEBP':  # WebP
            vp8x_pos = raw.rfind(b'VP8X')
            if vp8x_pos == -1:
                raise ValueError("No VP8X found in WebP")
            insert_pos = vp8x_pos + 10
        else:
            raise ValueError("Unsupported format for EOI embedding")

        # Append payload
        new_raw = raw[:insert_pos] + payload_bytes
        with open(img_path, 'wb') as f:
            f.write(new_raw)
        logging.info(f"Embedded EOI payload to {img_path} ({len(payload)} chars)")
    except Exception as e:
        logging.error(f"Error embedding EOI to {img_path}: {str(e)}")


def embed_lsb(clean_img_path, stego_img_path, payload, payload_size=0.4):
    """
    Embed a payload into a PNG image using LSB (Least Significant Bit) technique.
    Increased default payload_size to 0.4 bpp for stronger artifacts, easing detection.
    LSB replaces the LSB of pixel channels with payload bits—simple, keyless.
    Args:
        clean_img_path (str): Path to clean PNG.
        stego_img_path (str): Path to save stego PNG.
        payload (str or None): Text from markdown seed to embed (None for random).
        payload_size (float): Bits per pixel (default: 0.4 for better detectability).
    """
    try:
        img = Image.open(clean_img_path).convert("RGB")
        img_array = np.array(img)

        # Prepare payload (random if None or empty)
        if payload and isinstance(payload, str) and len(payload) > 0:
            logging.info(f"Embedding provided payload: {len(payload)} chars")
            binary_payload = "".join(
                format(ord(c), "08b")[::-1] for c in (payload + "\x00")
            )
        else:
            logging.info("No payload provided; generating random payload")
            total_bits = int(payload_size * img_array.size)
            binary_payload = "".join(map(str, np.random.randint(0, 2, total_bits)))

        total_bits = int(payload_size * img_array.size)
        binary_payload = binary_payload[:total_bits]

        flat = img_array.flatten()
        if len(binary_payload) > len(flat):
            raise ValueError(
                f"Payload size ({len(binary_payload)} bits) exceeds image capacity ({len(flat)} bits)."
            )

        for i in range(len(binary_payload)):
            flat[i] = (flat[i] & 0xFE) | int(binary_payload[i])

        stego_array = flat.reshape(img_array.shape)
        stego_img = Image.fromarray(stego_array)
        stego_img.save(stego_img_path, "PNG", compress_level=0)
        logging.info(
            f"Generated LSB stego image: {stego_img_path} (PNG, payload_size={payload_size})"
        )
    except Exception as e:
        logging.error(f"Error with LSB embedding: {str(e)}")


def extract_lsb(stego_img_path, payload_size=0.4):
    """
    Extract payload from a PNG LSB stego image.
    Reads the least significant bits to reconstruct the embedded binary data and converts to text.
    Args:
        stego_img_path (str): Path to stego PNG.
        payload_size (float): Bits per pixel used during embedding (default: 0.4).
    Returns:
        str: Extracted text payload (or empty string if extraction fails).
    """
    try:
        img = Image.open(stego_img_path).convert("RGB")
        img_array = np.array(img)
        flat = img_array.flatten()
        total_bits = int(payload_size * img_array.size)

        # Extract LSBs
        binary_payload = "".join(
            str(flat[i] & 1) for i in range(min(total_bits, len(flat)))
        )

        # Decode all bytes without stopping
        extracted_text = ""
        for i in range(0, len(binary_payload) - 7, 8):
            byte = binary_payload[i : i + 8]
            if len(byte) == 8:
                char_code = int(byte, 2)
                extracted_text += chr(char_code)  # No range check or break
        return extracted_text
    except Exception as e:
        logging.error(f"Error extracting LSB payload from {stego_img_path}: {str(e)}")
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
        extracted = user_comment[8:].decode("ascii") if len(user_comment) > 8 else ""
        if extracted == expected_payload[:65527]:  # Truncate to EXIF limit
            logging.info(f"EXIF verification passed: {img_path} matches payload.")
        else:
            logging.warning(
                f"EXIF verification failed: {img_path} does not match payload."
            )
            logging.warning(f"Expected (truncated): {expected_payload[:50]}...")
            logging.warning(f"Extracted: {extracted[:50]}...")
    except Exception as e:
        logging.error(f"Error verifying EXIF metadata for {img_path}: {str(e)}")

def verify_eoi(img_path, expected_payload):
    """
    Verify that appended EOI tail matches the expected payload.
    Args:
        img_path (str): Path to stego image.
        expected_payload (str): Original markdown content.
    """
    try:
        with open(img_path, 'rb') as f:
            raw = f.read()
        expected_bytes = expected_payload.encode('ascii')

        # Detect format and extract tail
        if raw.startswith(b'\xFF\xD8'):  # JPEG
            eoi_pos = raw.rfind(b'\xFF\xD9')
            if eoi_pos != -1:
                extracted = raw[eoi_pos + 2:]
            else:
                extracted = b""
        elif raw.startswith(b'\x89PNG'):  # PNG
            iend_pos = raw.rfind(b'IEND')
            if iend_pos != -1:
                extracted = raw[iend_pos + 12:]
            else:
                extracted = b""
        elif raw.startswith(b'GIF8'):  # GIF
            term_pos = raw.rfind(b';')
            if term_pos != -1:
                extracted = raw[term_pos + 1:]
            else:
                extracted = b""
        elif raw.startswith(b'RIFF') and raw[8:12] == b'WEBP':  # WebP
            vp8x_pos = raw.rfind(b'VP8X')
            if vp8x_pos != -1:
                extracted = raw[vp8x_pos + 10:]
            else:
                extracted = b""
        else:
            extracted = b""

        if extracted == expected_bytes:
            logging.info(f"EOI verification passed: {img_path} matches payload.")
        else:
            logging.warning(
                f"EOI verification failed: {img_path} does not match payload."
            )
            logging.warning(f"Expected: {expected_payload[:50]}...")
            logging.warning(f"Extracted: {extracted.decode('ascii', errors='ignore')[:50]}...")
    except Exception as e:
        logging.error(f"Error verifying EOI for {img_path}: {str(e)}")


def verify_images(seed_file, stego_paths, seed_payload, method, payload_size=0.4):
    """
    Verify that stego images contain the correct payload.
    - lsb: Verifies LSB-extracted payload (truncated to ~13 KB at 0.4 bpp).
    - exif: Verifies EXIF UserComment (up to ~65 KB).
    - eoi: Verifies appended tail.
    Args:
        seed_file (str): Name of the seed file (e.g., 'sample_seed.md').
        stego_paths (list): List of stego image paths to verify.
        seed_payload (str): Original markdown content.
        method (str): 'lsb', 'exif', 'eoi'.
        payload_size (float): Bits per pixel for LSB (default: 0.4).
    """
    logging.info(f"Verifying {method} stego images for {seed_file}...")
    for _ in tqdm(range(len(stego_paths)), desc=f"Verifying ({seed_file}, {method})"):
        stego_path = stego_paths[_]  # Use index from tqdm
        if method == "lsb":
            max_chars = int(payload_size * 512 * 512 * 3 / 8)  # Updated capacity
            extracted = extract_lsb(stego_path, payload_size)
            seed_truncated = seed_payload[:max_chars]
            if extracted.startswith(seed_truncated):
                logging.info(
                    f"LSB verification passed: {stego_path} matches {seed_file} payload."
                )
            else:
                logging.warning(
                    f"LSB verification failed: {stego_path} does not match {seed_file} payload."
                )
                logging.warning(f"Expected (truncated): {seed_truncated[:50]}...")
                logging.warning(f"Extracted: {extracted[:50]}...")
        elif method == "exif":
            verify_exif_metadata(stego_path, seed_payload)
        elif method == "eoi":
            verify_eoi(stego_path, seed_payload)


def generate_images(num_images=5, methods=["exif", "lsb", "eoi"], payload_size=0.4):
    """
    Generate clean and stego images for Project Starlight (Option 3).
    Increased default num_images to 5 for more data pairs.
    Hardcoded relative paths (run from dataset/grok_submission_2025/):
        - Clean images: ./clean/
        - Stego images: ./stego/
        - Markdown seeds: ./ (all .md files)
        - Payload size: Default 0.4 bpp for PNG LSB (configurable; higher for easier detection)
        - JPEG quality: 85 (fixed; within 75-95)
        - Methods: List of 'exif' (JPEG), 'lsb' (PNG), 'eoi' (JPEG)
        - Stego key: None (keyless)
    Behavior:
        - Iterates over each .md file, generating a batch of num_images clean + stego pairs per method.
        - Randomizes pattern_type ('linear' or 'radial') for diversity.
        - Labels images with seed basename and algorithm (e.g., sample_seed_eoi_001.jpeg).
        - If no seeds, generates a random batch labeled 'random' (no payload for verification).
        - Verifies payloads: LSB extraction, EXIF UserComment, EOI tail.
        - Uses tqdm for progress feedback.
    Args:
        num_images (int): Pairs per seed batch (default: 5; override via --limit or NUM_IMAGES env var).
        methods (list): List of methods to generate: 'exif', 'lsb', 'eoi'.
        payload_size (float): Bits per pixel for LSB (default: 0.4).
    """
    clean_dir = "./clean"
    stego_dir = "./stego"
    seed_dir = "./seeds"
    quality = 85  # Fixed: JPEG quality

    num_images = int(os.environ.get("NUM_IMAGES", num_images))

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)

    # Identify all .md seed files
    seed_files = [f for f in os.listdir(seed_dir) if f.endswith(".md")]
    if not seed_files:
        print("No .md seeds found; generating random batch.")
        seed_basename = "random"
        for method in methods:
            if method == "exif":
                format = "JPEG"
                algorithm = "exif"
            elif method == "lsb":
                format = "PNG"
                algorithm = "lsb"
            elif method == "eoi":
                format = "JPEG"
                algorithm = "eoi"
            else:
                continue
            stego_paths = []
            for i in tqdm(range(num_images), desc=f"Batch (random, {method})"):
                img_name = f"{seed_basename}_{algorithm}_{i:03d}.{format.lower()}"
                clean_path = os.path.join(clean_dir, img_name)
                stego_path = os.path.join(stego_dir, img_name)
                pattern = np.random.choice(["linear", "radial"])
                generate_clean_image(
                    clean_path, seed=i, format=format, pattern_type=pattern
                )
                if method == "lsb":
                    embed_lsb(
                     clean_path, stego_path, payload=None, payload_size=payload_size
                    )
                    stego_paths.append(stego_path)
                elif method == "exif":
                    generate_clean_image(
                     stego_path, seed=i, format=format, pattern_type=pattern
                    )  # Copy clean to stego
                    add_exif_metadata(stego_path, "")
                    stego_paths.append(stego_path)
                elif method == "eoi":
                    generate_clean_image(
                     stego_path, seed=i, format=format, pattern_type=pattern
                    )  # Copy clean to stego
                    embed_eoi(stego_path, "")
                    stego_paths.append(stego_path)
            verify_images("random", stego_paths, "", method, payload_size)
    else:
        for seed_file in seed_files:
            seed_basename = os.path.splitext(seed_file)[0]
            with open(os.path.join(seed_dir, seed_file), "r", encoding="utf-8") as f:
                seed_payload = f.read()
            for method in methods:
                if method == "exif":
                    format = "PNG"  # Support PNG for EXIF
                    algorithm = "exif"
                elif method == "lsb":
                    format = "PNG"
                    algorithm = "lsb"
                elif method == "eoi":
                    format = "JPEG"
                    algorithm = "eoi"
                else:
                    continue
                stego_paths = []
                for i in tqdm(
                    range(num_images), desc=f"Batch ({seed_basename}, {method})"
                ):
                    img_name = f"{seed_basename}_{algorithm}_{i:03d}.{format.lower()}"
                    clean_path = os.path.join(clean_dir, img_name)
                    stego_path = os.path.join(stego_dir, img_name)
                    pattern = np.random.choice(["linear", "radial"])
                    generate_clean_image(
                     clean_path, seed=i, format=format, pattern_type=pattern
                    )
                     if method == "lsb":
                     embed_lsb(
                         clean_path,
                         stego_path,
                         payload=seed_payload,
                         payload_size=payload_size,
                     )
                     stego_paths.append(stego_path)

                     # --- Create JSON Sidecar ---
                     json_path = stego_path + ".json"
                     embedding_data = {
                         "category": "pixel",
                         "technique": "lsb.rgb",
                         "ai42": False,
                         "bit_order": "lsb-first",
                     }
                     sidecar_content = {
                         "embedding": embedding_data,
                         "clean_file": os.path.basename(clean_path),
                     }
                     with open(json_path, "w") as f:
                         json.dump(sidecar_content, f, indent=2)
                     elif method == "exif":
                     generate_clean_image(
                         stego_path, seed=i, format=format, pattern_type=pattern
                     )  # Copy clean to stego
                     add_exif_metadata(stego_path, seed_payload)
                     stego_paths.append(stego_path)
                     elif method == "eoi":
                     generate_clean_image(
                         stego_path, seed=i, format=format, pattern_type=pattern
                     )  # Copy clean to stego
                     embed_eoi(stego_path, seed_payload)
                     stego_paths.append(stego_path)
                verify_images(
                    seed_file, stego_paths, seed_payload, format, payload_size
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate clean and stego images for Project Starlight."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of images to generate per payload per algorithm (overrides NUM_IMAGES env var).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="exif,lsb,eoi",
        help="Comma-separated methods to use: exif (JPEG), lsb (PNG), eoi (JPEG/PNG).",
    )
    parser.add_argument(
        "--payload_size",
        type=float,
        default=0.4,
        help="Bits per pixel for PNG LSB embedding (higher eases detection).",
    )
    args = parser.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",")]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("generation.log")],
    )

    try:
        generate_images(
            num_images=args.limit, methods=methods, payload_size=args.payload_size
        )
        print("Image generation completed. Check generation.log for details.")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        print("Error occurred. Check generation.log for details.")

# Notes:
# - Added 'radial' gradients for diversity, randomized per image—helps model generalize without masking artifacts.
# - Increased default LSB payload_size to 0.4 bpp (~13 KB capacity) for more obvious changes, improving detection confidence.
# - Default num_images=5 to generate more pairs (~4x data if rerun).
# - For even easier training, try --payload_size 0.6 or integrate real images (e.g., via external download to seed_dir).
# - exif: Stores payload in JPEG EXIF UserComment (~65 KB capacity, keyless).
# - lsb: Embeds payload via PNG LSB (configurable bpp, ~13 KB at 0.4 for 512x512, keyless).
# - eoi: Appends payload after image data (supports JPEG, PNG, GIF, WebP).
# - Verification: LSB extraction, EXIF UserComment, EOI tail.
# - Override: python data_generator.py --limit 10 --methods lsb,eoi --payload_size 0.5
# - Dependencies: pip install piexif Pillow numpy tqdm
# - Ensure Python 3.8+ for compatibility.
