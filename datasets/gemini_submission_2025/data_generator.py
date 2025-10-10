import os
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
NUM_IMAGES_PER_PAYLOAD = 5      # Number of image pairs to generate for each .md file
RESOLUTION = 512
HINT_BYTES = b'0xAI42'           # Sequence to signal hidden data
TERMINATOR_BYTES = b'\x00'       # Byte to signal the end of the message

# --- PNG (LSB) CONFIGURATION ---
PNG_FORMAT = 'PNG'
PNG_ALGORITHM = 'lsb' # ADDED: Algorithm name for PNG LSB
# Steganography Method: Least Significant Bit (LSB) on RGBA channels.
# Image Format: PNG (Lossless).

# --- JPEG (EOI APPEND) CONFIGURATION ---
JPEG_FORMAT = 'JPEG'
JPEG_QUALITY = 90
JPEG_ALGORITHM = 'eoi' # ADDED: Algorithm name for JPEG EOI Append
EOI_MARKER = b'\xFF\xD9'         # JPEG End of Image (EOI) marker
# Steganography Method: Append payload data after the JPEG EOI marker.
# Image Format: JPEG (Quality 90).

# --- PAYLOAD LOADING ---

def load_all_payloads(submission_dir="."):
    """
    Identifies all markdown seed files and loads their content.
    Returns a list of (base_filename, full_byte_payload) tuples.
    """
    all_payload_data = []
    seed_filenames = [f for f in os.listdir(submission_dir) if f.endswith('.md')]

    if not seed_filenames:
        print("No markdown seed files found. Using random payload as fallback for 1 batch.")
        # Fallback to 2KB random data
        random_size = 2048
        full_data = HINT_BYTES + os.urandom(random_size) + TERMINATOR_BYTES
        return [("random_fallback", full_data)]

    print(f"Found {len(seed_filenames)} unique seed file(s) to process.")
    
    for filename in sorted(seed_filenames):
        base_name = filename.replace('.md', '').replace('.', '_')
        
        try:
            with open(os.path.join(submission_dir, filename), 'r', encoding='utf-8') as f:
                payload_bytes = f.read().encode('utf-8')
        except Exception as e:
            print(f"Warning: Could not read {filename}. Skipping. Error: {e}")
            continue
            
        # Final data structure: [Hint Bytes] + [Payload Bytes] + [Terminator Byte]
        full_data = HINT_BYTES + payload_bytes + TERMINATOR_BYTES
                
        all_payload_data.append((base_name, full_data))
        print(f"  - Loaded '{filename}' ({len(payload_bytes)} bytes of source content). Total {len(full_data)} bytes.")

    return all_payload_data

# --- PNG (LSB) IMPLEMENTATION ---

def get_payload_bits(full_byte_payload):
    """Converts a byte payload into a flat list of bits."""
    payload_bits = []
    for byte in full_byte_payload:
        for i in range(7, -1, -1):
            payload_bits.append((byte >> i) & 1)
    return payload_bits

def embed_stego_lsb(clean_img_path, stego_path_target, full_byte_payload):
    """LSB steganography for PNG (RGBA)."""
    
    payload_bits = get_payload_bits(full_byte_payload)
    
    # Image must be converted to RGBA before NumPy conversion to ensure 4 channels
    clean_img = Image.open(clean_img_path).convert('RGBA') 
    img_array = np.array(clean_img)
    color_channels = img_array.flatten()
    
    if len(payload_bits) > len(color_channels):
        print("ERROR: LSB Payload size exceeds image capacity (RGBA). Skipping.")
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
    stego_img.save(stego_path_target, PNG_FORMAT)
    
    return True

def extract_lsb(stego_img_path, num_bits):
    """Extract the first N LSB bits from a PNG (RGBA) image."""
    stego_img = Image.open(stego_img_path).convert('RGBA')
    img_array = np.array(stego_img)
    color_channels = img_array.flatten()
    
    if num_bits > len(color_channels):
        raise ValueError(f"Not enough channels ({len(color_channels)}) to extract {num_bits} bits.")
    
    # Extract LSB (Bitwise AND with 0x01)
    extracted_bits = [(channel_val & 0x01) for channel_val in color_channels[:num_bits]]
    return extracted_bits

def test_lsb_steganography_on_image(stego_path_target, expected_byte_payload):
    """Verifies LSB embedding."""
    expected_bits = get_payload_bits(expected_byte_payload)
    num_test_bits = len(expected_bits)
    
    try:
        extracted_bits = extract_lsb(stego_path_target, num_test_bits)
    except Exception as e:
        print(f" [X] LSB Verification FAILED: Extraction error: {e}")
        return False

    test_passed = extracted_bits == expected_bits
    
    if not test_passed:
        mismatched_count = sum(1 for a, b in zip(extracted_bits, expected_bits) if a != b)
        print(f" [X] LSB Verification FAILED: {mismatched_count} bits mismatched.")
        
    return test_passed

# --- JPEG (EOI APPEND) IMPLEMENTATION ---

def embed_stego_eoi_append(clean_img_path, stego_path_target, full_byte_payload):
    """EOI Append steganography for JPEG."""
    
    try:
        with open(clean_img_path, 'rb') as f:
            clean_data = f.read()
    except FileNotFoundError:
        print("ERROR: Clean image not found. Skipping.")
        return False
        
    eoi_pos = clean_data.rfind(EOI_MARKER)
    
    if eoi_pos == -1:
        print("ERROR: Could not find JPEG EOI marker (0xFFD9). Skipping.")
        return False

    # The new stego file is: [Data before EOI] + [EOI Marker] + [Payload]
    stego_data = clean_data[:eoi_pos + len(EOI_MARKER)] + full_byte_payload
    
    try:
        with open(stego_path_target, 'wb') as f:
            f.write(stego_data)
    except Exception as e:
        print(f"ERROR: Failed to save stego image. Error: {e}")
        return False
    
    return True

def extract_eoi_append(stego_img_path):
    """Extract the appended data from a JPEG file."""
    try:
        with open(stego_img_path, 'rb') as f:
            stego_data = f.read()
    except FileNotFoundError:
        return b''
    
    eoi_pos = stego_data.rfind(EOI_MARKER)
    
    if eoi_pos == -1:
        return b''
        
    # Appended data starts immediately after the EOI marker (length 2)
    appended_data = stego_data[eoi_pos + len(EOI_MARKER):]
    
    return appended_data

def test_eoi_append_steganography_on_image(stego_path_target, expected_byte_payload):
    """Verifies EOI Append embedding."""
    
    extracted_bytes = extract_eoi_append(stego_path_target)
    
    test_passed = extracted_bytes == expected_byte_payload
    
    if not test_passed:
        print(" [X] EOI Append Verification FAILED. Size mismatch or content changed.")
        
    return test_passed

# --- MAIN GENERATION LOOP ---

def generate_images(num_images_per_payload=NUM_IMAGES_PER_PAYLOAD, resolution=RESOLUTION):
    """Generates clean/stego pairs for both PNG (LSB) and JPEG (EOI Append)."""
    
    clean_dir = "clean"
    stego_dir = "stego"
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    
    all_payload_data = load_all_payloads()
    
    print("-" * 50)
    print(f"Starting generation for {len(all_payload_data)} unique payload(s)...")

    total_images_generated = 0
    total_verifications_failed = 0
    
    # Iterate over both formats for diversity. Added algorithm name to the iteration.
    for current_format, algorithm_name, quality, embed_func, test_func in [
        (PNG_FORMAT, PNG_ALGORITHM, None, embed_stego_lsb, test_lsb_steganography_on_image),
        (JPEG_FORMAT, JPEG_ALGORITHM, JPEG_QUALITY, embed_stego_eoi_append, test_eoi_append_steganography_on_image)
    ]:
        print(f"\n--- Generating {current_format} Images ({'Q'+str(quality) if quality else 'Lossless'})... ---")
        
        for payload_index, (base_name, full_byte_payload) in enumerate(all_payload_data):
            for i in range(num_images_per_payload):
                
                # --- NEW FILENAME GENERATION LOGIC ---
                # Format: {payload_name}_{algorithm}_{index}.{ext}
                
                payload_name = base_name.lower() # payload_name
                algorithm = algorithm_name.lower() # lsb or eoi
                index = f'{i:03d}' # zero-padded index
                ext = current_format.lower() # png or jpeg
                
                # Construct the full structured filename
                filename = f'{payload_name}_{algorithm}_{index}.{ext}'
                
                clean_path = os.path.join(clean_dir, filename)
                stego_path = os.path.join(stego_dir, filename)
                
                # 2. Generate Synthetic Clean Image
                img_data_rgb = np.random.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
                
                # Prepare image data based on format requirements
                if current_format == PNG_FORMAT:
                    # PNG/LSB requires RGBA for max capacity and lossless saving
                    alpha_channel = np.full((resolution, resolution, 1), 255, dtype=np.uint8)
                    img_data_rgba = np.concatenate((img_data_rgb, alpha_channel), axis=2)
                    Image.fromarray(img_data_rgba).save(clean_path, current_format)
                
                elif current_format == JPEG_FORMAT:
                    # JPEG uses RGB and fixed quality
                    Image.fromarray(img_data_rgb).save(clean_path, current_format, quality=quality)
                
                # 3. Generate Stego Image
                embed_result = embed_func(clean_path, stego_path, full_byte_payload)
                
                if embed_result:
                    # 4. Perform verification
                    verification_success = test_func(stego_path, full_byte_payload)
                    
                    if verification_success:
                        print(f" [✔] {current_format} SUCCESS: {filename}")
                    else:
                        print(f" [X] {current_format} FAILED: {filename} 🚨")
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
