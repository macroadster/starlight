import os
import numpy as np
from PIL import Image

def generate_clean_image(output_path, size=(512, 512)):
    """
    Generate a 512x512 clean JPEG image with a colorful gradient pattern.
    Args:
        output_path (str): Path to save the clean image (e.g., ./clean/cover_001.jpeg).
        size (tuple): Image dimensions (default: 512x512).
    """
    try:
        # Create a colorful gradient with noise
        x, y = np.meshgrid(np.linspace(0, 1, size[0]), np.linspace(0, 1, size[1]))
        r = (np.sin(2 * np.pi * x * 5) + np.cos(2 * np.pi * y * 3)) * 127.5 + 127.5
        g = (np.sin(2 * np.pi * x * 3) + np.cos(2 * np.pi * y * 5)) * 127.5 + 127.5
        b = (np.sin(2 * np.pi * x * 4) + np.cos(2 * np.pi * y * 4)) * 127.5 + 127.5
        noise = np.random.normal(0, 10, size)
        img_array = np.stack([np.clip(r + noise, 0, 255), np.clip(g + noise, 0, 255), np.clip(b + noise, 0, 255)], axis=-1).astype(np.uint8)

        # Save as JPEG
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(img_array).save(output_path, 'JPEG', quality=85)
        print(f"Clean image saved to: {output_path}")
    except Exception as e:
        print(f"Error generating clean image: {str(e)}")

def embed_lsb_jpeg(clean_img_path, stego_img_path, payload=None, payload_size=0.2, quality=85):
    """
    Embed a payload into a JPEG image using LSB technique.
    Args:
        clean_img_path (str): Path to clean JPEG image.
        stego_img_path (str): Path to save stego JPEG image.
        payload (str, optional): Text from markdown seed to embed; if None, use random bits.
        payload_size (float): Bits per pixel (0.1-0.4 bpnzac) for random payload.
        quality (int): JPEG quality factor (75-95).
    """
    try:
        # Load clean image
        img = Image.open(clean_img_path).convert('RGB')
        img_array = np.array(img)

        # Prepare payload
        if payload is not None:
            binary_payload = ''.join(format(ord(c), '08b') for c in payload)
        else:
            total_bits = int(payload_size * img_array.size)
            binary_payload = ''.join(np.random.randint(0, 2, total_bits).astype(str))
        
        # Flatten image array for LSB embedding
        flat = img_array.flatten()
        if len(binary_payload) > len(flat):
            raise ValueError(f"Payload size ({len(binary_payload)} bits) exceeds image capacity ({len(flat)} bits).")

        # Embed payload in LSB
        for i in range(len(binary_payload)):
            flat[i] = (flat[i] & 0xFE) | int(binary_payload[i])

        # Reshape and save as JPEG
        stego_array = flat.reshape(img_array.shape)
        stego_img = Image.fromarray(stego_array)
        stego_img.save(stego_img_path, 'JPEG', quality=quality)
        print(f"Generated stego image: {stego_img_path}")
    except Exception as e:
        print(f"Error processing {clean_img_path}: {str(e)}")

def generate_images(num_images=1):
    """
    Generate clean and stego JPEG images for Project Starlight.
    Hardcoded relative paths (run from dataset/grok_submission_2025/):
        - Clean images: ./clean/
        - Stego images: ./stego/
        - Markdown seeds: ./
        - Payload size: 0.2 bpnzac
        - JPEG quality: 85
    Args:
        num_images (int): Number of clean/stego image pairs to generate (default: 1).
    """
    clean_dir = "./clean"
    stego_dir = "./stego"
    seed_dir = "./"
    payload_size = 0.2
    quality = 85

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    
    # Validate inputs
    if not 0.1 <= payload_size <= 0.4:
        raise ValueError("Payload size must be between 0.1 and 0.4 bpnzac.")
    if not 75 <= quality <= 95:
        raise ValueError("JPEG quality must be between 75 and 95.")

    # Load markdown seeds if provided
    seed_payloads = []
    if seed_dir and os.path.exists(seed_dir):
        for seed_file in os.listdir(seed_dir):
            if seed_file.endswith('.md'):
                with open(os.path.join(seed_dir, seed_file), 'r', encoding='utf-8') as f:
                    seed_payloads.append(f.read())
    
    # Generate clean and stego images
    for i in range(num_images):
        img_name = f"cover_{i:03d}.jpeg"
        clean_path = os.path.join(clean_dir, img_name)
        stego_path = os.path.join(stego_dir, img_name)  # Use same filename
        # Generate clean image
        generate_clean_image(clean_path)
        # Generate stego image
        payload = seed_payloads[i % len(seed_payloads)] if seed_payloads else None
        embed_lsb_jpeg(clean_path, stego_path, payload, payload_size, quality)

if __name__ == "__main__":
    try:
        generate_images(num_images=1)
        print("Image generation completed.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Note: For advanced embedding (e.g., J-UNIWARD), integrate tools like MATLAB's J-UNIWARD
# (https://github.com/daniellerch/steganalysis) via subprocess or a Python port.