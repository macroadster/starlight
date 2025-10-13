import os
import numpy as np
from PIL import Image

# --- Helper Function to Find and Read Markdown File ---

def get_message_from_md():
    """Finds the first .md file and returns its stripped content, raw content, and filename."""
    md_files = [f for f in os.listdir('.') if f.endswith('.md')]
    if not md_files:
        print("Error: No .md file found in the current directory.")
        return None, None, None

    # Sort to ensure deterministic selection if multiple files exist
    md_file_path = sorted(md_files)[0]
    payload_name = os.path.splitext(md_file_path)[0] # Extract filename without extension
    print(f"Using content from: {md_file_path}")

    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content_raw = f.read()
        return content_raw.strip(), content_raw, payload_name
    except Exception as e:
        print(f"Error reading file {md_file_path}: {e}")
        return None, None, None

# Get the message content and payload name
message_to_embed, original_md_content_raw, payload_name = get_message_from_md()

if message_to_embed is None:
    exit(1)

# --- Filename Generation based on Dataset Guidelines ---
# Format: {payload_name}_{algorithm}_{index}.{ext}
# Algorithm: 'alpha' (Alpha channel LSB embedding)
# Index: '000'
# Extension: 'png'
ALGORITHM_NAME = "alpha"
IMAGE_INDEX = "000"
COVER_IMAGE_NAME = f"seed_alpha_000.png"

# Construct the new structured filename for both clean and stego output
new_image_filename = f"{payload_name}_{ALGORITHM_NAME}_{IMAGE_INDEX}.png"
clean_image_path = f"clean/{COVER_IMAGE_NAME}" # The path to read the cover image
stego_image_path = f"stego/{new_image_filename}" # The path to save the stego image

print(f"Message to embed (stripped): '{message_to_embed}'")
print(f"New Stego Filename: {new_image_filename}")

# --- Data Preparation ---

# The AI hint: 'A', 'I', '4', '2'
ai_hint = [0x41, 0x49, 0x34, 0x32] 
message_hex = [val for val in message_to_embed.encode('utf-8')]

# Combine the hint, the message, and a null byte (0x00) for the terminator
full_hidden_bytes = ai_hint + message_hex + [0x00]

# Open the image
try:
    # Use the clean path for the cover image
    img = Image.open(clean_image_path).convert("RGBA")
except FileNotFoundError:
    print(f"Error: '{clean_image_path}' not found.")
    exit(1)

pixels = np.array(img)
max_embeddable_bits = pixels.shape[0] * pixels.shape[1]
total_bits_needed = len(full_hidden_bytes) * 8

if total_bits_needed > max_embeddable_bits:
    print(f"Error: Message is too long for the image.")
    print(f"Needed: {total_bits_needed} bits. Available: {max_embeddable_bits} bits.")
    exit(1)

# --- EMBEDDING LOGIC: 8 Pixels per Byte ---

pixel_index = 0
for byte_val in full_hidden_bytes:
    # Embed each of the 8 bits of 'byte_val' into 8 separate pixels
    for bit_index in range(8):
        # 1. Determine the coordinates of the current pixel
        x, y = divmod(pixel_index, pixels.shape[1])
        
        # 2. Get the current pixel's alpha value
        alpha_val = pixels[x, y, 3]
        
        # 3. Get the specific bit to embed (LSB first)
        bit_to_embed = (byte_val >> bit_index) & 0x01
        
        # 4. Modify the LSB of the alpha channel:
        pixels[x, y, 3] = (alpha_val & 0xFE) | bit_to_embed
        
        # Move to the next pixel
        pixel_index += 1

# --- Save Stego Image ---

stego_dir = "stego"
os.makedirs(stego_dir, exist_ok=True)
img_out = Image.fromarray(pixels)
# Save to the new structured path
img_out.save(stego_image_path)

print(f"\nAI hint and full message bytes embedded successfully into {stego_image_path}.")

# --------------------------------------------------------------------------
# --- Verification Test with Corrected Extraction ---
# --------------------------------------------------------------------------

def extract_full_message(image_path, hint_length):
    """
    Extracts the full hidden message from the alpha LSB of an image, 
    assuming 8 pixels are used to store 1 full byte. Stops at the null byte.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        return None, "Stego image not found."

    pixels = np.array(img)
    extracted_bytes = []
    pixel_count = pixels.shape[0] * pixels.shape[1]
    
    current_byte = 0
    current_byte_bits = 0

    for i in range(pixel_count):
        x, y = divmod(i, pixels.shape[1])
        
        # Extract the LSB (the stored message bit)
        extracted_bit = pixels[x, y, 3] & 0x01
        
        # Reconstruct the byte: shift the current bit to its correct position (LSB first)
        current_byte |= (extracted_bit << current_byte_bits)
        current_byte_bits += 1

        # When 8 bits are collected, we have a full byte
        if current_byte_bits == 8:
            # Check for the null terminator (0x00)
            if current_byte == 0x00:
                break
                
            extracted_bytes.append(current_byte)
            
            # Reset for the next byte
            current_byte = 0
            current_byte_bits = 0
    
    # Separate the hint and the message
    extracted_hint = extracted_bytes[:hint_length]
    extracted_message_bytes = extracted_bytes[hint_length:]
    
    # Decode the message bytes back to a string
    try:
        extracted_message = bytes(extracted_message_bytes).decode('utf-8')
    except UnicodeDecodeError:
        return None, "Error decoding extracted bytes to UTF-8."

    return extracted_hint, extracted_message


# --- Execute Verification ---
HINT_LENGTH = 4 # The length of the AI hint [0x41, 0x49, 0x34, 0x32]
# Use the new structured path for verification
extracted_hint, extracted_message = extract_full_message(stego_image_path, HINT_LENGTH)

if extracted_message is not None:
    # 1. Verify the AI Hint
    expected_hint = bytes(ai_hint).decode('ascii')
    extracted_hint_str = bytes(extracted_hint).decode('ascii')
    
    hint_match = expected_hint == extracted_hint_str
    print(f"\n--- Verification Results ---")
    print(f"Hint Check: {'✅ SUCCESS' if hint_match else '❌ FAILURE'}")
    print(f"  Expected Hint: {expected_hint}")
    print(f"  Extracted Hint: {extracted_hint_str}")
    
    # 2. Verify the Message Content
    message_match = message_to_embed == extracted_message
    print(f"Message Check: {'✅ SUCCESS' if message_match else '❌ FAILURE'}")
    print(f"  Original Message: '{message_to_embed}'")
    print(f"  Extracted Message:  '{extracted_message}'")
    
    if hint_match and message_match:
        print("\n🏆 **Verification Successful**: The full hidden message was correctly embedded and extracted.")
    else:
        print("\n⚠️ **Verification Failed**: Extracted data does not fully match the original source.")
else:
    print(f"\nVerification failed during extraction: {extracted_message}")
