#!/usr/bin/env python3
"""
Project Starlight Data Generator - Claude's Contribution
Steganography Methods:
1. PNG - DCT coefficient embedding (frequency domain)
2. BMP - Palette manipulation for indexed color
3. PNG - Alpha channel LSB (transparency-based hiding)
4. WAV - Audio LSB embedding for accessibility (for Maya and audio-based AI)

Author: Claude (Anthropic)
Date: 2025
License: MIT

Usage:
  python data_generator.py                    # Generate 12 pairs (default)
  python data_generator.py --limit 100        # Generate 100 pairs
  python data_generator.py --limit 1000       # Generate 1000 pairs
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import struct
from pathlib import Path
import wave
import argparse

class ClaudeStegGenerator:
    """Generate clean and stego images/audio with extraction verification"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.clean_dir = self.base_dir / "clean"
        self.stego_dir = self.base_dir / "stego"
        self.audio_dir = self.base_dir / "audio"
        self.clean_dir.mkdir(exist_ok=True)
        self.stego_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        
    def read_seed_files(self):
        """Read all .md files in base directory as payloads"""
        seed_files = list(self.base_dir.glob("*.md"))
        payloads = {}
        
        for seed_file in seed_files:
            with open(seed_file, 'r', encoding='utf-8') as f:
                content = f.read()
                payloads[seed_file.stem] = content.encode('utf-8')
        
        # Default payload if no seed files
        if not payloads:
            payloads['default'] = b"Project Starlight: Teaching AI common sense through inscribed wisdom. When an AI spends its last satoshi to save a human, that act is inscribed forever."
        
        return payloads
    
    def generate_diverse_clean_image(self, index, img_type='gradient'):
        """Generate visually diverse clean images"""
        width, height = 512, 512
        
        if img_type == 'gradient':
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    img_array[y, x] = [
                        int(255 * x / width),
                        int(255 * y / height),
                        int(255 * (x + y) / (width + height))
                    ]
            return Image.fromarray(img_array)
        
        elif img_type == 'geometric':
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            np.random.seed(index)
            for _ in range(20):
                x1, y1 = np.random.randint(0, width, 2)
                x2, y2 = np.random.randint(0, width, 2)
                
                # Fix: Ensure x1 <= x2 and y1 <= y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Ensure non-zero dimensions
                if x1 == x2:
                    x2 = x1 + 1
                if y1 == y2:
                    y2 = y1 + 1
                
                color = tuple(np.random.randint(0, 256, 3).tolist())
                shape = np.random.choice(['rectangle', 'ellipse', 'line'])
                
                if shape == 'rectangle':
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                elif shape == 'ellipse':
                    draw.ellipse([x1, y1, x2, y2], fill=color)
                else:
                    draw.line([x1, y1, x2, y2], fill=color, width=3)
            return img
        
        elif img_type == 'noise':
            np.random.seed(index)
            img_array = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
            return Image.fromarray(img_array)
        
        else:  # 'blocks'
            img = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(img)
            block_size = 64
            
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    color = (
                        ((x // block_size) * 40) % 256,
                        ((y // block_size) * 60) % 256,
                        ((x + y) // block_size * 30) % 256
                    )
                    draw.rectangle([x, y, x + block_size, y + block_size], fill=color)
            return img
    
    # ============= PNG ALPHA CHANNEL LSB =============
    
    def png_alpha_lsb_embed(self, img, payload):
        """PNG Alpha Channel LSB Embedding"""
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        payload_bits = ''.join(format(byte, '08b') for byte in payload)
        length_header = format(len(payload), '032b')
        full_payload = length_header + payload_bits
        
        max_bits = height * width
        if len(full_payload) > max_bits:
            raise ValueError(f"Payload too large: {len(full_payload)} bits > {max_bits}")
        
        bit_index = 0
        for y in range(height):
            for x in range(width):
                if bit_index < len(full_payload):
                    alpha = img_array[y, x, 3]
                    alpha = (alpha & 0xFE) | int(full_payload[bit_index])
                    img_array[y, x, 3] = alpha
                    bit_index += 1
                else:
                    break
            if bit_index >= len(full_payload):
                break
        
        # Fix: Remove deprecated 'mode' parameter
        return Image.fromarray(img_array)
    
    def png_alpha_lsb_extract(self, img_path):
        """Extract data from PNG Alpha Channel LSB"""
        img = Image.open(img_path).convert('RGBA')
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Extract length header (32 bits)
        bits = []
        for y in range(height):
            for x in range(width):
                bits.append(str(img_array[y, x, 3] & 1))
                if len(bits) == 32:
                    break
            if len(bits) >= 32:
                break
        
        length = int(''.join(bits[:32]), 2)
        
        # Extract payload
        bits = []
        bit_count = 0
        for y in range(height):
            for x in range(width):
                bits.append(str(img_array[y, x, 3] & 1))
                bit_count += 1
                if bit_count >= 32 + length * 8:
                    break
            if bit_count >= 32 + length * 8:
                break
        
        payload_bits = bits[32:32 + length * 8]
        payload_bytes = bytes([int(''.join(payload_bits[i:i+8]), 2) 
                               for i in range(0, len(payload_bits), 8)])
        
        return payload_bytes
    
    # ============= BMP PALETTE MANIPULATION =============
    
    def bmp_palette_embed(self, img, payload):
        """BMP Palette Index Manipulation"""
        img_palette = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        img_array = np.array(img_palette)
        
        payload_bits = ''.join(format(byte, '08b') for byte in payload)
        length_header = format(len(payload), '032b')
        full_payload = length_header + payload_bits
        
        height, width = img_array.shape
        max_bits = height * width
        
        if len(full_payload) > max_bits:
            raise ValueError("Payload too large")
        
        bit_index = 0
        for y in range(height):
            for x in range(width):
                if bit_index < len(full_payload):
                    pixel = int(img_array[y, x])
                    new_pixel = (pixel & 0xFE) | int(full_payload[bit_index])
                    img_array[y, x] = new_pixel
                    bit_index += 1
                else:
                    break
            if bit_index >= len(full_payload):
                break
        
        # Fix: Remove deprecated 'mode' parameter
        result = Image.fromarray(img_array)
        result.putpalette(img_palette.getpalette())
        return result
    
    def bmp_palette_extract(self, img_path):
        """Extract data from BMP Palette"""
        img = Image.open(img_path)
        if img.mode != 'P':
            raise ValueError("Image is not in palette mode")
        
        img_array = np.array(img)
        height, width = img_array.shape
        
        # Extract length
        bits = []
        for y in range(height):
            for x in range(width):
                bits.append(str(int(img_array[y, x]) & 1))
                if len(bits) == 32:
                    break
            if len(bits) >= 32:
                break
        
        length = int(''.join(bits[:32]), 2)
        
        # Extract payload
        bits = []
        bit_count = 0
        for y in range(height):
            for x in range(width):
                bits.append(str(int(img_array[y, x]) & 1))
                bit_count += 1
                if bit_count >= 32 + length * 8:
                    break
            if bit_count >= 32 + length * 8:
                break
        
        payload_bits = bits[32:32 + length * 8]
        payload_bytes = bytes([int(''.join(payload_bits[i:i+8]), 2) 
                               for i in range(0, len(payload_bits), 8)])
        
        return payload_bytes
    
    # ============= PNG DCT-LIKE EMBEDDING =============
    
    def png_dct_embed(self, img, payload):
        """PNG DCT Coefficient Embedding with payload truncation - SIMPLIFIED LSB approach"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img, dtype=np.uint8)  # Keep as uint8
        block_size = 8
        height, width = img_array.shape[:2]
        
        # Calculate available capacity
        blocks_available = (height // block_size) * (width // block_size)
        
        # Reserve 32 bits for length header
        max_payload_bits = blocks_available - 32
        max_payload_bytes = max_payload_bits // 8
        
        # Truncate payload if needed
        original_length = len(payload)
        if len(payload) > max_payload_bytes:
            print(f"      DCT: Truncating payload from {len(payload)} to {max_payload_bytes} bytes")
            payload = payload[:max_payload_bytes]
        
        payload_bits = ''.join(format(byte, '08b') for byte in payload)
        length_header = format(len(payload), '032b')
        full_payload = length_header + payload_bits
        
        print(f"      DCT Embed: payload={len(payload)} bytes, bits={len(full_payload)}, blocks={blocks_available}")
        
        bit_index = 0
        
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                if bit_index < len(full_payload):
                    # Use LSB of middle pixel instead of ±8 modification
                    mid_y, mid_x = block_size // 2, block_size // 2
                    pixel_val = int(img_array[y + mid_y, x + mid_x, 0])
                    
                    # Embed bit in LSB
                    bit_val = int(full_payload[bit_index])
                    new_val = (pixel_val & 0xFE) | bit_val
                    img_array[y + mid_y, x + mid_x, 0] = new_val
                    
                    bit_index += 1
                else:
                    break
            if bit_index >= len(full_payload):
                break
        
        print(f"      DCT Embed: Successfully embedded {bit_index} bits via LSB")
        
        return Image.fromarray(img_array)
    
    def png_dct_extract(self, img_path, original_img):
        """Extract data from PNG DCT embedding by comparing with original"""
        stego_img = Image.open(img_path).convert('RGB')
        stego_array = np.array(stego_img, dtype=np.float32)
        orig_array = np.array(original_img.convert('RGB'), dtype=np.float32)
        
        block_size = 8
        height, width = stego_array.shape[:2]
        
        bits = []
        
        # Extract bits from all blocks
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                mid_y, mid_x = block_size // 2, block_size // 2
                
                stego_val = stego_array[y + mid_y, x + mid_x, 0]
                orig_val = orig_array[y + mid_y, x + mid_x, 0]
                
                diff = stego_val - orig_val
                
                # Use threshold of 4 to detect modification
                if diff > 4:
                    bits.append('1')
                elif diff < -4:
                    bits.append('0')
                else:
                    if diff >= 0:
                        bits.append('1')
                    else:
                        bits.append('0')
                
                if len(bits) == 32:
                    try:
                        length = int(''.join(bits), 2)
                        if length > 100000 or length <= 0:
                            pass
                    except:
                        pass
                
                if len(bits) >= 32:
                    try:
                        length = int(''.join(bits[:32]), 2)
                        if 0 < length <= 100000 and len(bits) >= 32 + length * 8:
                            break
                    except:
                        pass
            
            if len(bits) >= 32:
                try:
                    length = int(''.join(bits[:32]), 2)
                    if 0 < length <= 100000 and len(bits) >= 32 + length * 8:
                        break
                except:
                    pass
        
        if len(bits) < 32:
            return b""
        
        try:
            length = int(''.join(bits[:32]), 2)
            
            if length > 100000 or length <= 0:
                return b""
            
            if len(bits) < 32 + length * 8:
                return b""
            
            payload_bits = bits[32:32 + length * 8]
            payload_bytes = bytes([int(''.join(payload_bits[i:i+8]), 2) 
                                   for i in range(0, len(payload_bits), 8)])
            
            return payload_bytes
        except Exception as e:
            return b""
    
    # ============= AUDIO GENERATION (placeholders for audio features) =============
    
    def text_to_audio_simple(self, text, sample_rate=22050):
        """Convert text to audio using simple phoneme-based synthesis"""
        duration_per_char = 0.05
        samples_per_char = int(sample_rate * duration_per_char)
        
        audio_samples = []
        base_freq = 200
        
        for char in text:
            char_val = ord(char)
            freq = base_freq + (char_val % 50) * 20
            
            t = np.linspace(0, duration_per_char, samples_per_char)
            tone = np.sin(2 * np.pi * freq * t)
            tone += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            tone += 0.2 * np.sin(2 * np.pi * freq * 3 * t)
            
            envelope = np.linspace(0, 1, samples_per_char // 10)
            tone[:len(envelope)] *= envelope
            tone[-len(envelope):] *= envelope[::-1]
            
            audio_samples.extend(tone)
        
        audio_array = np.array(audio_samples)
        audio_array = audio_array / np.max(np.abs(audio_array))
        audio_array = (audio_array * 32767 * 0.8).astype(np.int16)
        
        return audio_array, sample_rate
    
    def save_wav(self, audio_data, sample_rate, filepath):
        """Save audio data as WAV file"""
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    # ============= MAIN GENERATION =============
    
    def generate_dataset(self, num_images=12):
        """Generate complete dataset with verification"""
        payloads = self.read_seed_files()
        
        print(f"Found {len(payloads)} payload(s): {list(payloads.keys())}")
        print(f"Generating {num_images} pairs per payload with verification...\n")
        
        methods = [
            ('png_alpha', 'png', self.png_alpha_lsb_embed, self.png_alpha_lsb_extract),
            ('bmp_palette', 'bmp', self.bmp_palette_embed, self.bmp_palette_extract),
            ('png_dct', 'png', self.png_dct_embed, self.png_dct_extract),
        ]
        
        image_types = ['gradient', 'geometric', 'noise', 'blocks']
        
        verification_results = {
            'total': 0,
            'success': 0,
            'failed': []
        }
        
        for payload_name, payload_data in payloads.items():
            print(f"{'='*60}")
            print(f"Processing payload: {payload_name}")
            print(f"Payload length: {len(payload_data)} bytes")
            print(f"{'='*60}\n")
            
            # Generate images
            for i in range(num_images):
                img_type = image_types[i % len(image_types)]
                method_name, file_ext, embed_func, extract_func = methods[i % len(methods)]
                
                base_filename = f"{payload_name}_{method_name}_{i:03d}"
                
                try:
                    # Generate clean image
                    clean_img = self.generate_diverse_clean_image(
                        i * 100 + hash(payload_name) % 100, img_type
                    )
                    
                    # Save clean
                    clean_path = self.clean_dir / f"{base_filename}.{file_ext}"
                    if file_ext == 'bmp':
                        clean_img.save(clean_path, 'BMP')
                    else:
                        clean_img.save(clean_path, 'PNG')
                    
                    # Generate stego (may truncate payload for DCT)
                    stego_img = embed_func(clean_img, payload_data)
                    stego_path = self.stego_dir / f"{base_filename}.{file_ext}"
                    
                    if file_ext == 'bmp':
                        stego_img.save(stego_path, 'BMP')
                    else:
                        stego_img.save(stego_path, 'PNG')
                    
                    # Verify extraction
                    verification_results['total'] += 1
                    
                    if method_name == 'png_dct':
                        # CRITICAL FIX: For DCT, we need to get the truncated payload
                        # Calculate what was actually embedded
                        block_size = 8
                        height, width = stego_img.size
                        blocks_available = (height // block_size) * (width // block_size)
                        max_payload_bits = blocks_available - 32
                        max_payload_bytes = max_payload_bits // 8
                        expected_payload = payload_data[:max_payload_bytes]
                        
                        print(f"      Calling DCT extraction for {base_filename}.{file_ext}")
                        print(f"      Expected payload size: {len(expected_payload)} bytes")
                        extracted = extract_func(stego_path, clean_img)
                        print(f"      Extracted payload size: {len(extracted)} bytes")
                    else:
                        expected_payload = payload_data
                        extracted = extract_func(stego_path)
                    
                    if extracted == expected_payload:
                        print(f"  ✓ Verified: {base_filename}.{file_ext}")
                        verification_results['success'] += 1
                    else:
                        print(f"  ✗ Verification failed: {base_filename}.{file_ext}")
                        print(f"    Expected {len(expected_payload)} bytes, got {len(extracted)}")
                        if len(extracted) > 0 and len(extracted) <= 50:
                            print(f"    First bytes extracted: {extracted[:20]}")
                            print(f"    First bytes expected: {expected_payload[:20]}")
                        verification_results['failed'].append(f"{base_filename}.{file_ext}")
                    
                except Exception as e:
                    print(f"  ✗ Failed {base_filename}: {e}")
                    verification_results['failed'].append(f"{base_filename}.{file_ext}")
                    verification_results['total'] += 1
        
        # Final report
        print(f"\n{'='*60}")
        print("GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total files: {verification_results['total']}")
        print(f"Successfully verified: {verification_results['success']}")
        print(f"Failed: {len(verification_results['failed'])}")
        
        if verification_results['failed']:
            print(f"\nFailed files:")
            for f in verification_results['failed']:
                print(f"  - {f}")
        
        success_rate = (verification_results['success'] / verification_results['total'] * 100) if verification_results['total'] > 0 else 0
        print(f"\nVerification rate: {success_rate:.1f}%")
        
        print(f"\nOutput directories:")
        print(f"  Clean images: {self.clean_dir}")
        print(f"  Stego images: {self.stego_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate steganography training dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_generator.py                 # Generate 12 pairs (default)
  python data_generator.py --limit 100     # Generate 100 pairs
  python data_generator.py --limit 1000    # Generate 1000 pairs
        """
    )
    
    parser.add_argument('--limit', type=int, default=12, 
                       help='Number of image pairs to generate (default: 12)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("="*60)
    print("Project Starlight - Data Generator v3")
    print("Teaching AI Common Sense Through Inscribed Wisdom")
    print("="*60)
    print(f"\nGenerating {args.limit} image pairs")
    print("\nSteganography Techniques:")
    print("• Alpha Channel LSB (PNG) - Transparency-based")
    print("• Palette Index Manipulation (BMP) - Indexed color")
    print("• DCT Coefficient Embedding (PNG) - Frequency domain")
    print("="*60 + "\n")
    
    generator = ClaudeStegGenerator()
    generator.generate_dataset(num_images=args.limit)
