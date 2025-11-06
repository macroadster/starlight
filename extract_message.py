#!/usr/bin/env python3
# extract_message.py - Extract message from specific stego image
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

def main():
    stego_path = "datasets/sample_submission_2025/stego/seed1_lsb_004.png"
    
    print(f"Extracting message from: {stego_path}")
    
    try:
        # Extract message
        extracted_bits = extract_lsb_simple(stego_path, 100)
        
        print(f"Extracted {len(extracted_bits)} bits:")
        print("Bits:", extracted_bits)
        
        # Convert to string if possible
        try:
            # Group bits into bytes
            bytes_list = []
            for i in range(0, len(extracted_bits), 8):
                if i + 8 <= len(extracted_bits):
                    byte_bits = extracted_bits[i:i+8]
                    byte_val = 0
                    for bit in byte_bits:
                        byte_val = (byte_val << 1) | bit
                    bytes_list.append(byte_val)
            
            # Try to decode as ASCII
            try:
                decoded = ''.join([chr(b) if 32 <= b <= 126 else f'\\x{b:02x}' for b in bytes_list])
                print(f"Decoded (ASCII attempt): {decoded}")
            except:
                print("Could not decode as ASCII")
                
        except Exception as e:
            print(f"Error converting bits to bytes: {e}")
        
        # Show statistics
        ones_count = np.sum(extracted_bits)
        zeros_count = len(extracted_bits) - ones_count
        print(f"Statistics: {ones_count} ones, {zeros_count} zeros ({ones_count/len(extracted_bits)*100:.1f}% ones)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()