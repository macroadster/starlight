import os
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
NUM_IMAGES_PER_PAYLOAD = 10      # Number of image pairs to generate for each .md file
RESOLUTION = 512
IMAGE_FORMAT = 'PNG'             # Using PNG for lossless LSB steganography
HINT_BYTES = b'0xAI42'           # The multi-byte sequence to signal hidden data
TERMINATOR_BYTES = b'\x00'       # A null byte to signal the end of the message

def load_all_payloads(submission_dir="."):
    """
    Identifies all markdown seed files and loads their content,
    returning a list of (base_filename, bitstream) tuples.
    """
    all_payload_data = []
    # Identify all files ending with .md in the current directory
    seed_filenames = [f for f in os.listdir(submission_dir) if f.endswith('.md')]

    if not seed_filenames:
        print("No markdown seed files found. Using random payload as fallback for 1 batch.")
        # Fallback to a single batch of random data
        random_size = int((RESOLUTION * RESOLUTION * 3 * 0.2) / 8) # Using 0.2 bpnzac as a default
        random_bits = [bit for byte in os.urandom(random_size) for bit in [(byte >> i) & 1 for i in range(7, -1, -1)]]
        return [("random_fallback", random_bits)]

    print(f"Found {len(seed_filenames)} unique seed file(s) to process.")
    
    for filename in sorted(seed_filenames): # Process files in alphabetical order for consistency
        base_name = filename.replace('.md', '').replace('.', '_') # e.g., 'ai_common_sense'
        
        try:
            with open(os.path.join(submission_dir, filename), 'r') as f:
                payload_bytes = f.read().encode('utf-8')
        except Exception as e:
            print(f"Warning: Could not read {filename}. Skipping. Error: {e}")
            continue
            
        # Final data structure: [Hint Bytes] + [Payload Bytes] + [Terminator Byte]
        full_data = HINT_BYTES + payload_bytes + TERMINATOR_BYTES
        
        # Convert to a flat list of bits
        payload_bits = []
        for byte in full_data:
            for i in range(7, -1, -1):
                payload_bits.append((byte >> i) & 1)
                
        all_payload_data.append((base_name, payload_bits))
        print(f"  - Loaded '{filename}' ({len(payload_bytes)} bytes).")

    return all_payload_data

def embed_stego_lsb(clean_img_path, stego_path_target, payload_bits):
    """
    Implements the LSB steganography using bitwise operations, saving to a specified path.
    """
    
    # Image must be converted to RGBA before NumPy conversion to ensure 4 channels
    clean_img = Image.open(clean_img_path).convert('RGBA') 
    img_array = np.array(clean_img)
    color_channels = img_array.flatten()
    
    if len(payload_bits) > len(color_channels):
        print("ERROR: Payload size exceeds image embedding capacity. Skipping image.")
        return 

    # LSB Embedding
    embedded_channels = color_channels.copy()
    
    for i in range(len(payload_bits)):
        bit_to_embed = payload_bits[i]
        channel_val = embedded_channels[i]
        
        # Clear the LSB (Bitwise AND with 0xFE) and set the new bit (Bitwise OR)
        new_channel_val = (channel_val & 0xFE) | bit_to_embed
        embedded_channels[i] = new_channel_val

    # Reconstruct and Save the Image
    tainted_array = embedded_channels.reshape(img_array.shape).astype(np.uint8)
    
    # FIX: Remove mode='RGBA' parameter to avoid DeprecationWarning
    stego_img = Image.fromarray(tainted_array) 

    # Save the stego image as a lossless PNG
    stego_img.save(stego_path_target, IMAGE_FORMAT)
    
    print(f"Embedded LSB payload into: {stego_path_target}")


def generate_images(num_images_per_payload=NUM_IMAGES_PER_PAYLOAD, resolution=RESOLUTION):
    """Generates a batch of clean and stego images for every unique markdown payload."""
    
    clean_dir = "clean"
    stego_dir = "stego"
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    
    # Load all payloads
    all_payload_data = load_all_payloads()
    
    print("-" * 50)
    print(f"Starting image generation for {len(all_payload_data)} unique payload(s)...")

    total_images_generated = 0
    
    for payload_index, (base_name, payload_bits) in enumerate(all_payload_data):
        print(f"\nBatch {payload_index + 1}: Generating {num_images_per_payload} images using payload '{base_name}'...")
        
        for i in range(num_images_per_payload):
            # 1. Filename is based on the payload source (e.g., 'ai_common_sense_001.png')
            filename = f'{base_name}_{i:03d}.{IMAGE_FORMAT.lower()}'
            clean_path = os.path.join(clean_dir, filename)
            
            # 2. Generate Synthetic Clean Image (RGBA for LSB consistency)
            img_data_rgb = np.random.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
            alpha_channel = np.full((resolution, resolution, 1), 255, dtype=np.uint8)
            img_data_rgba = np.concatenate((img_data_rgb, alpha_channel), axis=2)
            
            # 3. Save Clean Image as PNG (Lossless)
            # FIX: Remove mode='RGBA' parameter to avoid DeprecationWarning
            Image.fromarray(img_data_rgba).save(clean_path, IMAGE_FORMAT)
            
            # 4. Generate Stego Image
            stego_path = os.path.join(stego_dir, filename)
            embed_stego_lsb(clean_path, stego_path, payload_bits)
            
            total_images_generated += 1
            
    print("-" * 50)
    print(f"Generation complete. Total pairs generated: {total_images_generated // 2}.")

if __name__ == "__main__":
    generate_images()
