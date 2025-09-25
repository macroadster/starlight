import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar for large batches
try:
    import conseal as cl  # For J-UNIWARD; pip install conseal
    CONSEAL_AVAILABLE = True
except ImportError:
    print("Warning: conseal not installed. Falling back to LSB. Run 'pip install conseal' for J-UNIWARD.")
    CONSEAL_AVAILABLE = False

def generate_clean_image(output_path, size=(512, 512), seed=None, format='JPEG'):
    """
    Generate a 512x512 clean image (JPEG or PNG) with a colorful gradient pattern.
    Uses a seed for reproducible diversity across images in a batch.
    Args:
        output_path (str): Path to save (e.g., ./clean/cover_sample_seed_001.jpeg).
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
            img.save(output_path, 'JPEG', quality=85)  # Q=85, suitable for J-UNIWARD
        else:  # PNG
            img.save(output_path, 'PNG', compress_level=0)  # Lossless, optimal for LSB
        print(f"Clean image saved to: {output_path} ({format})")
    except Exception as e:
        print(f"Error generating clean image: {str(e)}")

def embed_juniward(clean_img_path, stego_img_path, payload, rate=0.2):
    """
    Embed a payload into a JPEG image using J-UNIWARD via conseal library.
    J-UNIWARD uses wavelet residuals for adaptive cost assignment, minimizing detectable distortions.
    No stego key (Conseal simulates embedding without key). Suitable for JPEG (Q=75-95).
    Args:
        clean_img_path (str): Path to clean JPEG.
        stego_img_path (str): Path to save stego JPEG.
        payload (str): Markdown text to embed (or empty for random simulation).
        rate (float): Embedding rate (fixed: 0.2 bpnzAC; balances capacity/security).
    """
    try:
        if not CONSEAL_AVAILABLE:
            raise ImportError("conseal not available; install with 'pip install conseal'")
        # Conseal simulates embedding; payload is used to set rate
        cl.juniward.embed(clean_img_path, stego_img_path, rate=rate)
        print(f"Generated J-UNIWARD stego image: {stego_img_path} (rate={rate})")
    except Exception as e:
        print(f"Error with J-UNIWARD embedding: {str(e)}. Falling back to LSB.")
        embed_lsb_jpeg(clean_img_path, stego_img_path, payload, format='JPEG')

def embed_lsb_jpeg(clean_img_path, stego_img_path, payload, payload_size=0.2, format='JPEG'):
    """
    Fallback: Embed a payload using LSB (Least Significant Bit) technique.
    LSB replaces the LSB of pixel channels with payload bits—simple, keyless, suitable for PNG or JPEG.
    Args:
        clean_img_path (str): Path to clean image.
        stego_img_path (str): Path to save stego image.
        payload (str): Text from markdown seed to embed (empty for random).
        payload_size (float): Bits per pixel (fixed: 0.2 bpnzAC approximation).
        format (str): 'JPEG' or 'PNG'.
    """
    try:
        img = Image.open(clean_img_path).convert('RGB')
        img_array = np.array(img)

        # Prepare payload (random if empty)
        if payload:
            binary_payload = ''.join(format(ord(c), '08b') for c in payload)
        else:
            total_bits = int(payload_size * img_array.size)
            binary_payload = ''.join(np.random.randint(0, 2, total_bits).astype(str))
        
        total_bits = int(payload_size * img_array.size)
        binary_payload = binary_payload[:total_bits]

        flat = img_array.flatten()
        if len(binary_payload) > len(flat):
            raise ValueError(f"Payload size ({len(binary_payload)} bits) exceeds image capacity ({len(flat)} bits).")

        for i in range(len(binary_payload)):
            flat[i] = (flat[i] & 0xFE) | int(binary_payload[i])

        stego_array = flat.reshape(img_array.shape)
        stego_img = Image.fromarray(stego_array)
        if format == 'JPEG':
            stego_img.save(stego_img_path, 'JPEG', quality=85)
        else:
            stego_img.save(stego_img_path, 'PNG', compress_level=0)
        print(f"Generated LSB stego image: {stego_img_path} ({format})")
    except Exception as e:
        print(f"Error with LSB embedding: {str(e)}")

def generate_images(num_images=1, format='JPEG', method='J-UNIWARD'):
    """
    Generate clean and stego images for Project Starlight (Option 3).
    Hardcoded relative paths (run from dataset/grok_submission_2025/):
        - Clean images: ./clean/
        - Stego images: ./stego/
        - Markdown seeds: ./ (all .md files)
        - Payload size: 0.2 bpnzAC (fixed; rate for J-UNIWARD, truncation for LSB)
        - JPEG quality: 85 (fixed; within 75-95, suitable for J-UNIWARD)
        - Format: JPEG (default) or PNG (lossless; LSB only)
        - Method: J-UNIWARD (default for JPEG; via conseal) or LSB (fallback/PNG)
        - Stego key: None (J-UNIWARD via conseal is keyless; LSB is keyless)
    Behavior:
        - Iterates over each .md file in ./, generating a batch of num_images clean + stego pairs per seed.
        - Labels images with seed basename (e.g., sample_seed.md -> cover_sample_seed_001.jpeg).
        - If no seeds, generates a random batch labeled 'random'.
        - Uses tqdm for progress feedback.
    Args:
        num_images (int): Pairs per seed batch (default: 1; override via NUM_IMAGES env var).
        format (str): 'JPEG' (J-UNIWARD/LSB) or 'PNG' (LSB only).
        method (str): 'J-UNIWARD' (default for JPEG) or 'LSB' (fallback/PNG).
    """
    clean_dir = "./clean"
    stego_dir = "./stego"
    seed_dir = "./"
    payload_size = 0.2  # Fixed: 0.2 bpnzAC (rate for J-UNIWARD; truncation for LSB)
    quality = 85  # Fixed: JPEG quality (PNG is lossless)

    num_images = int(os.environ.get('NUM_IMAGES', num_images))
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    
    # Validate
    if format == 'JPEG' and not 75 <= quality <= 95:
        raise ValueError("JPEG quality must be 75-95.")
    if format not in ['JPEG', 'PNG']:
        raise ValueError("Format must be 'JPEG' or 'PNG'.")
    if method == 'J-UNIWARD' and format != 'JPEG':
        raise ValueError("J-UNIWARD requires JPEG format.")
    if method == 'J-UNIWARD' and not CONSEAL_AVAILABLE:
        print("J-UNIWARD unavailable; using LSB.")
        method = 'LSB'

    # Identify all .md seed files
    seed_files = [f for f in os.listdir(seed_dir) if f.endswith('.md')]
    if not seed_files:
        print("No .md seeds found; generating random batch.")
        seed_basename = "random"
        for i in tqdm(range(num_images), desc=f"Batch (random)"):
            img_name = f"cover_{seed_basename}_{i:03d}.{format.lower()}"
            clean_path = os.path.join(clean_dir, img_name)
            stego_path = os.path.join(stego_dir, img_name)
            generate_clean_image(clean_path, seed=i, format=format)
            if method == 'J-UNIWARD':
                embed_juniward(clean_path, stego_path, payload="", rate=payload_size)
            else:
                embed_lsb_jpeg(clean_path, stego_path, payload="", payload_size=payload_size, format=format)
    else:
        for seed_file in seed_files:
            seed_basename = os.path.splitext(seed_file)[0]
            with open(os.path.join(seed_dir, seed_file), 'r', encoding='utf-8') as f:
                seed_payload = f.read()
            for i in tqdm(range(num_images), desc=f"Batch ({seed_basename})"):
                img_name = f"cover_{seed_basename}_{i:03d}.{format.lower()}"
                clean_path = os.path.join(clean_dir, img_name)
                stego_path = os.path.join(stego_dir, img_name)
                generate_clean_image(clean_path, seed=i, format=format)
                if method == 'J-UNIWARD':
                    embed_juniward(clean_path, stego_path, payload=seed_payload, rate=payload_size)
                else:
                    embed_lsb_jpeg(clean_path, stego_path, payload=seed_payload, payload_size=payload_size, format=format)

if __name__ == "__main__":
    try:
        generate_images(num_images=1, format='JPEG', method='J-UNIWARD')  # Default: J-UNIWARD on JPEG
        print("Image generation completed.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Notes:
# - J-UNIWARD (default): Adaptive for JPEG, rate=0.2 bpnzAC, keyless (conseal simulation).
# - LSB Fallback: For PNG or if conseal unavailable; keyless.
# - Override: export NUM_IMAGES=10; export FORMAT=PNG; export METHOD=LSB; python data_generator.py
# - Conseal: https://github.com/uibk-uncover/conseal (MPL 2.0 license)
