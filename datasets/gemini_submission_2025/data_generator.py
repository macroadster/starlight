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
        # Using 0.2 bpnzac as a default for capacity check
        random_size = int((RESOLUTION * RESOLUTION * 4 * 0.2) / 8) 
        random_bits = [bit for byte in os.urandom(random_size) for bit in [(byte >> i) & 1 for i in range(7, -1, -1)]]
        return [("random_fallback", random_bits)]

    print(f"Found {len(seed_filenames)} unique seed file(s) to process.")
    
    for filename in sorted(seed_filenames): # Process files in alphabetical order for consistency
        base_name = filename.replace('.md', '').replace('.', '_') # e.g., 'ai_common_sense'
        
        try:
            with open(os.path.join(submission_dir, filename), 'r', encoding='utf-8') as f:
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
        print(f"  - Loaded '{filename}' ({len(payload_bytes)} bytes of source content). Total {len(payload_bits)} bits.")

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
        return False

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
    
    stego_img = Image.fromarray(tainted_array) 

    # Save the stego image as a lossless PNG
    stego_img.save(stego_path_target, IMAGE_FORMAT)
    
    return True # Return True on successful embed

# --- EXTRACTION HELPER FUNCTION ---

def extract_lsb(stego_img_path, num_bits):
    """
    Helper function to extract the first N LSB bits from an image.
    """
    stego_img = Image.open(stego_img_path).convert('RGBA')
    img_array = np.array(stego_img)
    color_channels = img_array.flatten()
    
    # Check if there are enough channels to extract the requested bits
    if num_bits > len(color_channels):
        raise ValueError(f"Not enough channels ({len(color_channels)}) to extract {num_bits} bits.")
    
    # Extract LSB (Bitwise AND with 0x01)
    extracted_bits = [(channel_val & 0x01) for channel_val in color_channels[:num_bits]]
    return extracted_bits

# --- VERIFICATION TEST FUNCTION ---

def test_lsb_steganography_on_image(stego_path_target, base_name, expected_payload_bits):
    """
    Runs a test on a generated stego image to ensure the embedded payload is intact.
    Prints a success/failure message for each verification.
    """
    num_test_bits = len(expected_payload_bits)
    
    # 1. Extract the payload from the generated image
    try:
        extracted_bits = extract_lsb(stego_path_target, num_test_bits)
    except Exception as e:
        print(f" [X] Verification FAILED for {os.path.basename(stego_path_target)}: Extraction error: {e}")
        return False

    # 2. Verify that the extracted bits match the original payload bits
    test_passed = extracted_bits == expected_payload_bits
    
    if test_passed:
        print(f" [✔] Verification SUCCESS for {os.path.basename(stego_path_target)}")
    else:
        mismatched_count = sum(1 for a, b in zip(extracted_bits, expected_payload_bits) if a != b)
        print(f" [X] Verification FAILED for {os.path.basename(stego_path_target)}: {mismatched_count} bits mismatched.")
        
    return test_passed


def generate_images(num_images_per_payload=NUM_IMAGES_PER_PAYLOAD, resolution=RESOLUTION):
    """Generates a batch of clean and stego images for every unique markdown payload."""
    
    clean_dir = "clean"
    stego_dir = "stego"
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    
    # Load all payloads
    all_payload_data = load_all_payloads()
    
    print("-" * 50)
    print(f"Starting image generation and **full verification** for {len(all_payload_data)} unique payload(s)...")

    total_images_generated = 0
    total_verifications_failed = 0
    
    for payload_index, (base_name, payload_bits) in enumerate(all_payload_data):
        print(f"\nBatch {payload_index + 1}: Generating and verifying {num_images_per_payload} images using payload '{base_name}'...")
        
        for i in range(num_images_per_payload):
            filename = f'{base_name}_{i:03d}.{IMAGE_FORMAT.lower()}'
            clean_path = os.path.join(clean_dir, filename)
            stego_path = os.path.join(stego_dir, filename)
            
            # 1. Generate Synthetic Clean Image
            img_data_rgb = np.random.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
            alpha_channel = np.full((resolution, resolution, 1), 255, dtype=np.uint8)
            img_data_rgba = np.concatenate((img_data_rgb, alpha_channel), axis=2)
            
            # 2. Save Clean Image
            Image.fromarray(img_data_rgba).save(clean_path, IMAGE_FORMAT)
            
            # 3. Generate Stego Image
            embed_result = embed_stego_lsb(clean_path, stego_path, payload_bits)
            
            if embed_result:
                # 4. Perform verification on the saved file
                verification_success = test_lsb_steganography_on_image(stego_path, base_name, payload_bits)
                if not verification_success:
                    total_verifications_failed += 1
                
                total_images_generated += 1
            
    print("-" * 50)
    print("Verification and Generation Summary:")
    print(f"Total pairs generated and verified: {total_images_generated}")
    if total_verifications_failed > 0:
        print(f"ATTENTION: Total verifications FAILED: {total_verifications_failed} 🚨")
    else:
        print("All generated images passed verification. Data integrity confirmed. ✅")
    print("-" * 50)

if __name__ == "__main__":
    generate_images()
