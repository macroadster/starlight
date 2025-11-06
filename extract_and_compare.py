#!/usr/bin/env python3
# extract_and_compare.py - Extract message and compare with original text
import numpy as np
import cv2

def extract_lsb_simple(image_path, msg_len=100):
    """Extract message bits from image using LSB"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_img = img_rgb.reshape(-1)
    
    bits = []
    for i in range(min(msg_len, len(flat_img)//3)):
        bits.append(flat_img[i*3] & 1)
    
    return np.array(bits, dtype=int)

def text_to_bits(text):
    """Convert text to binary bits"""
    bits = []
    for char in text:
        # Convert each character to 8-bit binary
        binary = format(ord(char), '08b')
        for bit in binary:
            bits.append(int(bit))
    return bits

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
    print(f"Text length: {len(original_text)} characters")
    print(f"Text: {repr(original_text)}")
    
    # Convert original text to bits
    original_bits = text_to_bits(original_text)
    print(f"Original bits: {len(original_bits)} bits")
    print(f"First 50 bits: {original_bits[:50]}")
    
    # Extract from stego image
    stego_path = "datasets/sample_submission_2025/stego/seed1_lsb_004.png"
    print(f"\n=== Extracting from {stego_path} ===")
    
    extracted_bits = extract_lsb_simple(stego_path, len(original_bits))
    print(f"Extracted bits: {len(extracted_bits)} bits")
    print(f"First 50 bits: {extracted_bits[:50]}")
    
    # Compare bits
    if len(original_bits) == len(extracted_bits):
        matches = sum(1 for i in range(len(original_bits)) if original_bits[i] == extracted_bits[i])
        accuracy = matches / len(original_bits)
        print(f"\n=== Comparison ===")
        print(f"Bit accuracy: {accuracy:.4f} ({matches}/{len(original_bits)} bits match)")
        
        if accuracy == 1.0:
            print("✅ PERFECT MATCH! Message extracted correctly!")
        else:
            print(f"❌ Partial match. {len(original_bits) - matches} bits differ.")
    else:
        print(f"❌ Length mismatch: {len(original_bits)} vs {len(extracted_bits)} bits")
    
    # Try to decode extracted bits as text
    extracted_text = bits_to_text(extracted_bits)
    print(f"\n=== Decoded Text ===")
    print(f"Decoded: {repr(extracted_text)}")
    
    if original_text == extracted_text:
        print("✅ Texts match perfectly!")
    else:
        print("❌ Texts differ.")
        # Show first differing character
        for i, (orig, ext) in enumerate(zip(original_text, extracted_text)):
            if orig != ext:
                print(f"First difference at position {i}: '{orig}' vs '{ext}'")
                break

if __name__ == "__main__":
    main()