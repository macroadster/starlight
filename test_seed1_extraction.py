#!/usr/bin/env python3
# test_seed1_extraction.py - Test extraction from seed1_lsb_004.png
import numpy as np
import cv2

def extract_lsb_correct(image_path, msg_len):
    """Extract message bits using correct LSB method"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_img = img_rgb.reshape(-1)
    
    bits = []
    for i in range(min(msg_len, len(flat_img))):
        bits.append(flat_img[i] & 1)
    
    return np.array(bits, dtype=int)

def bits_to_text(bits):
    """Convert binary bits to text"""
    text = ""
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte_bits = bits[i:i+8]
            byte_val = 0
            for bit in byte_bits:
                byte_val = (byte_val << 1) | bit
            text += chr(byte_val)
    return text

def main():
    # Read original message
    with open("datasets/sample_submission_2025/seed1.md", "r") as f:
        original_text = f.read()
    
    print("=== Original Message ===")
    print(f"Length: {len(original_text)} characters")
    print(f"First 100 chars: {repr(original_text[:100])}")
    
    # Extract from stego image
    stego_path = "datasets/sample_submission_2025/stego/seed1_lsb_004.png"
    print(f"\n=== Extracting from {stego_path} ===")
    
    # Calculate required bits (8 bits per character)
    required_bits = len(original_text) * 8
    
    extracted_bits = extract_lsb_correct(stego_path, required_bits)
    print(f"Extracted {len(extracted_bits)} bits")
    
    # Decode to text
    extracted_text = bits_to_text(extracted_bits)
    
    print(f"\n=== Extracted Message ===")
    print(f"Length: {len(extracted_text)} characters")
    print(f"First 100 chars: {repr(extracted_text[:100])}")
    
    # Compare
    if original_text == extracted_text:
        print("\n✅ SUCCESS! Perfect message recovery!")
        print("The hidden message has been completely extracted from the steganographic image.")
    else:
        print(f"\n❌ Messages differ")
        # Find first difference
        for i, (orig, ext) in enumerate(zip(original_text, extracted_text)):
            if orig != ext:
                print(f"First difference at position {i}: {repr(orig)} vs {repr(ext)}")
                break

if __name__ == "__main__":
    main()