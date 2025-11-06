#!/usr/bin/env python3
# advanced_extract.py - Try different LSB extraction methods
import numpy as np
import cv2
import json

def extract_lsb_all_channels(image_path, msg_len=100):
    """Extract from all RGB channels sequentially"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_img = img_rgb.reshape(-1)
    
    bits = []
    for i in range(min(msg_len, len(flat_img))):
        bits.append(flat_img[i] & 1)
    
    return np.array(bits, dtype=int)

def extract_lsb_red_only(image_path, msg_len=100):
    """Extract from red channel only"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape
    red_channel = img_rgb[:,:,0].reshape(-1)
    
    bits = []
    for i in range(min(msg_len, len(red_channel))):
        bits.append(red_channel[i] & 1)
    
    return np.array(bits, dtype=int)

def extract_lsb_msb_first(image_path, msg_len=100):
    """Extract with MSB-first bit order within bytes"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_img = img_rgb.reshape(-1)
    
    bits = []
    for byte_idx in range(min(msg_len//8, len(flat_img))):
        byte_val = flat_img[byte_idx]
        # Extract bits MSB-first
        for bit_idx in range(7, -1, -1):
            if len(bits) < msg_len:
                bits.append((byte_val >> bit_idx) & 1)
    
    return np.array(bits, dtype=int)

def text_to_bits(text):
    """Convert text to binary bits"""
    bits = []
    for char in text:
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
    
    original_bits = text_to_bits(original_text)
    print(f"Original message: {len(original_text)} chars, {len(original_bits)} bits")
    
    stego_path = "datasets/sample_submission_2025/stego/seed1_lsb_004.png"
    
    # Read JSON metadata
    json_path = stego_path + ".json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n=== Metadata ===")
        print(f"Technique: {metadata.get('embedding', {}).get('technique')}")
        print(f"Bit order: {metadata.get('embedding', {}).get('bit_order')}")
        print(f"AI42: {metadata.get('embedding', {}).get('ai42')}")
    
    # Try different extraction methods
    methods = [
        ("Red channel only", extract_lsb_red_only),
        ("All channels sequential", extract_lsb_all_channels),
        ("MSB-first order", extract_lsb_msb_first)
    ]
    
    for method_name, extract_func in methods:
        print(f"\n=== {method_name} ===")
        try:
            extracted_bits = extract_func(stego_path, len(original_bits))
            
            # Calculate accuracy
            if len(original_bits) == len(extracted_bits):
                matches = sum(1 for i in range(len(original_bits)) if original_bits[i] == extracted_bits[i])
                accuracy = matches / len(original_bits)
                print(f"Bit accuracy: {accuracy:.4f}")
                
                if accuracy > 0.9:  # If good accuracy, show decoded text
                    decoded_text = bits_to_text(extracted_bits)
                    print(f"Decoded text: {repr(decoded_text[:100])}...")
                    
                    if original_text == decoded_text:
                        print("✅ PERFECT MATCH!")
                        return
            else:
                print(f"Length mismatch: {len(original_bits)} vs {len(extracted_bits)}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import os
    main()