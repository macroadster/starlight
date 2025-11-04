#!/usr/bin/env python3
"""
Project Starlight Data Generator - Claude's Contribution v7
Steganography Methods:
1. PNG - Alpha channel LSB (transparency-based hiding) - AI42 protocol
2. BMP - Palette manipulation for indexed color - Human-compatible

Author: Claude (Anthropic)
Date: 2025
License: MIT

CHANGES IN v7:
✓ SDM removed (requires clean reference image - not blockchain compatible)
✓ Focused on self-contained methods (Alpha LSB, Palette)
✓ Alpha Protocol: AI42 prefix (AI-specific communication)
✓ Palette: No AI42 prefix (human-compatible steganography)
✓ All other v6 fixes retained
"""

import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import json


class ClaudeStegGenerator:
    """Generate clean and stego images with extraction verification"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.clean_dir = self.base_dir / "clean"
        self.stego_dir = self.base_dir / "stego"
        self.clean_dir.mkdir(exist_ok=True)
        self.stego_dir.mkdir(exist_ok=True)
        
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
            payloads['default'] = b"Project Starlight: Teaching AI common sense through inscribed wisdom."
        
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
                color = tuple(np.random.randint(0, 256, 3).tolist())
                shape = np.random.choice(['rectangle', 'ellipse', 'line'])
                
                # Ensure coordinates are in correct order (min, max)
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                
                # Skip if zero width or height
                if x_min == x_max or y_min == y_max:
                    continue
                
                if shape == 'rectangle':
                    draw.rectangle([x_min, y_min, x_max, y_max], fill=color)
                elif shape == 'ellipse':
                    draw.ellipse([x_min, y_min, x_max, y_max], fill=color)
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
    
    def validate_capacity(self, payload, capacity, method_name):
        """Validate that payload fits in container"""
        payload_size = len(payload)
        if payload_size > capacity:
            raise ValueError(
                f"{method_name}: Payload too large - {payload_size} bytes exceeds "
                f"capacity of {capacity} bytes"
            )
    
    # ============= PNG ALPHA CHANNEL LSB (AI42 PROTOCOL) =============

    def png_alpha_embed(self, img, payload):
        """PNG Alpha Channel LSB Embedding with AI42 prefix"""
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        img_array = np.array(img)
        height, width = img_array.shape[:2]

        # Validate capacity (AI42 prefix + payload + null terminator)
        prefix_bytes = 4  # AI42
        terminator_bytes = 1  # null
        max_bytes = (height * width) // 8 - prefix_bytes - terminator_bytes
        self.validate_capacity(payload, max_bytes, "PNG Alpha LSB")

        # LSB-first encoding with AI42 prefix (Alpha Protocol)
        ai42_prefix = b"AI42"
        prefix_bits = ''.join(format(byte, '08b')[::-1] for byte in ai42_prefix)
        payload_bits = ''.join(format(byte, '08b')[::-1] for byte in payload)
        terminator_bits = '00000000'  # null terminator
        full_payload = prefix_bits + payload_bits + terminator_bits
        
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
        
        return Image.fromarray(img_array)
    
    def png_alpha_extract(self, img_path):
        """Extract data from PNG Alpha Channel LSB (expects AI42 prefix)"""
        img = Image.open(img_path).convert('RGBA')
        img_array = np.array(img)
        height, width = img_array.shape[:2]

        # Collect all LSB bits
        all_bits = []
        for y in range(height):
            for x in range(width):
                all_bits.append(str(img_array[y, x, 3] & 1))

        # Check AI42 prefix
        if len(all_bits) < 32:
            return b""

        prefix_bits = ''.join(all_bits[:32])
        expected_prefix = ''.join(format(byte, '08b')[::-1] for byte in b"AI42")
        if prefix_bits != expected_prefix:
            return b""

        # Find null terminator
        payload_start = 32
        terminator_pos = -1
        for i in range(payload_start, len(all_bits) - 7, 8):
            byte_bits = ''.join(all_bits[i:i+8])
            if byte_bits == '00000000':
                terminator_pos = i
                break

        if terminator_pos == -1:
            return b""

        payload_bits = all_bits[payload_start:terminator_pos]
        if len(payload_bits) % 8 != 0:
            return b""

        # Convert to bytes (reverse each byte back to MSB)
        payload_bytes = bytes([int(''.join(payload_bits[j:j+8])[::-1], 2)
                               for j in range(0, len(payload_bits), 8)])
        return payload_bytes
    
    # ============= BMP PALETTE MANIPULATION (HUMAN-COMPATIBLE) =============
    
    def bmp_palette_embed(self, img, payload):
        """BMP Palette Index Manipulation (no AI42 - human-compatible)"""
        # Ensure RGB first for consistent conversion
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to palette mode
        try:
            img_palette = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        except Exception as e:
            raise ValueError(f"Failed to convert to palette mode: {e}")

        # Verify conversion
        if img_palette.mode != 'P':
            raise ValueError("Palette conversion did not produce P mode image")

        img_array = np.array(img_palette)
        height, width = img_array.shape

        # Validate capacity (payload + null terminator, no AI42 prefix for palette)
        terminator_bytes = 1  # null
        max_bytes = (height * width) // 8 - terminator_bytes
        self.validate_capacity(payload, max_bytes, "BMP Palette")

        # LSB-first encoding (no AI42 prefix - supports human palette steganography)
        payload_bits = ''.join(format(byte, '08b')[::-1] for byte in payload)
        terminator_bits = '00000000'  # null terminator
        full_payload = payload_bits + terminator_bits
        
        bit_index = 0
        for y in range(height):
            for x in range(width):
                if bit_index < len(full_payload):
                    pixel = int(img_array[y, x])
                    
                    # Ensure valid palette index
                    if pixel >= 256:
                        pixel = 255
                    
                    new_pixel = (pixel & 0xFE) | int(full_payload[bit_index])
                    img_array[y, x] = new_pixel
                    bit_index += 1
                else:
                    break
            if bit_index >= len(full_payload):
                break
        
        result = Image.fromarray(img_array)
        result.putpalette(img_palette.getpalette())
        return result
    
    def bmp_palette_extract(self, img_path):
        """Extract data from BMP Palette (no AI42 prefix expected)"""
        img = Image.open(img_path)
        if img.mode != 'P':
            raise ValueError("Image is not in palette mode")

        img_array = np.array(img)
        height, width = img_array.shape

        # Collect all LSB bits
        all_bits = []
        for y in range(height):
            for x in range(width):
                all_bits.append(str(int(img_array[y, x]) & 1))

        # No AI42 prefix for palette - find null terminator directly
        terminator_pos = -1
        for i in range(0, len(all_bits) - 7, 8):
            byte_bits = ''.join(all_bits[i:i+8])
            if byte_bits == '00000000':
                terminator_pos = i
                break

        if terminator_pos == -1:
            return b""

        payload_bits = all_bits[0:terminator_pos]
        if len(payload_bits) % 8 != 0:
            return b""

        # Convert to bytes (reverse each byte back to MSB)
        payload_bytes = bytes([int(''.join(payload_bits[j:j+8])[::-1], 2)
                               for j in range(0, len(payload_bits), 8)])
        return payload_bytes
    
    # ============= TESTING =============
    
    def test_methods(self):
        """Test each steganography method individually"""
        test_payload = b"Test message for debugging steganography methods!"
        
        print("="*60)
        print("TESTING INDIVIDUAL METHODS")
        print("="*60)
        print(f"Test payload: {len(test_payload)} bytes\n")
        
        results = {
            'passed': [],
            'failed': []
        }
        
        # Test 1: PNG Alpha LSB (AI42 Protocol)
        try:
            print("1. Testing PNG Alpha LSB (AI42 Protocol)...")
            img = self.generate_diverse_clean_image(0, 'gradient')
            stego = self.png_alpha_embed(img, test_payload)
            temp_path = self.stego_dir / "_test_alpha.png"
            stego.save(temp_path, 'PNG')
            extracted = self.png_alpha_extract(temp_path)
            temp_path.unlink()
            
            if extracted == test_payload:
                print("   ✓ PASSED\n")
                results['passed'].append('PNG Alpha LSB (AI42)')
            else:
                print(f"   ✗ FAILED: extracted {len(extracted)} bytes\n")
                results['failed'].append('PNG Alpha LSB (AI42)')
        except Exception as e:
            print(f"   ✗ ERROR: {e}\n")
            results['failed'].append('PNG Alpha LSB (AI42)')
        
        # Test 2: BMP Palette (Human-compatible)
        try:
            print("2. Testing BMP Palette (Human-compatible)...")
            img = self.generate_diverse_clean_image(1, 'blocks')
            stego = self.bmp_palette_embed(img, test_payload)
            temp_path = self.stego_dir / "_test_palette.bmp"
            stego.save(temp_path, 'BMP')
            extracted = self.bmp_palette_extract(temp_path)
            temp_path.unlink()
            
            if extracted == test_payload:
                print("   ✓ PASSED\n")
                results['passed'].append('BMP Palette')
            else:
                print(f"   ✗ FAILED: extracted {len(extracted)} bytes\n")
                results['failed'].append('BMP Palette')
        except Exception as e:
            print(f"   ✗ ERROR: {e}\n")
            results['failed'].append('BMP Palette')
        
        # Summary
        print("="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Passed: {len(results['passed'])}/2")
        print(f"Failed: {len(results['failed'])}/2")
        
        if results['passed']:
            print(f"\n✓ Passed methods:")
            for method in results['passed']:
                print(f"  - {method}")
        
        if results['failed']:
            print(f"\n✗ Failed methods:")
            for method in results['failed']:
                print(f"  - {method}")
        
        print("="*60 + "\n")
        
        return len(results['failed']) == 0
    
    # ============= DATASET GENERATION =============
    
    def generate_dataset(self, num_images=12):
        """Generate complete dataset with verification"""
        payloads = self.read_seed_files()
        
        print(f"Found {len(payloads)} payload(s): {list(payloads.keys())}")
        print(f"Generating {num_images} pairs per payload with verification...\n")
        
        methods = [
            ('alpha', 'png', self.png_alpha_embed, self.png_alpha_extract),
            ('palette', 'bmp', self.bmp_palette_embed, self.bmp_palette_extract),
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
            
            # Track sequential numbering per method
            method_counters = {method[0]: 0 for method in methods}
            
            # Generate images
            for i in range(num_images):
                img_type = image_types[i % len(image_types)]
                method_name, file_ext, embed_func, extract_func = methods[i % len(methods)]
                
                # Use sequential counter for this method
                method_idx = method_counters[method_name]
                method_counters[method_name] += 1
                
                base_filename = f"{payload_name}_{method_name}_{method_idx:03d}"
                
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
                    
                    # Generate stego
                    stego_img = embed_func(clean_img, payload_data)
                    stego_path = self.stego_dir / f"{base_filename}.{file_ext}"
                    
                    if file_ext == 'bmp':
                        stego_img.save(stego_path, 'BMP')
                    else:
                        stego_img.save(stego_path, 'PNG')

                    # --- Create JSON Sidecar ---
                    json_path = stego_path.with_suffix(stego_path.suffix + '.json')
                    embedding_data = {}
                    if method_name == 'alpha':
                        embedding_data = {"category": "pixel", "technique": "alpha", "ai42": True}
                    elif method_name == 'palette':
                        embedding_data = {"category": "pixel", "technique": "palette", "ai42": False, "bit_order": "lsb-first"}
                    
                    if embedding_data:
                        sidecar_content = {"embedding": embedding_data}
                        with open(json_path, 'w') as f:
                            json.dump(sidecar_content, f, indent=2)
                    
                    # Verify extraction
                    extracted = extract_func(stego_path)
                    
                    verification_results['total'] += 1
                    
                    if extracted == payload_data:
                        print(f"  ✓ Verified: {base_filename}.{file_ext}")
                        verification_results['success'] += 1
                    else:
                        print(f"  ✗ Verification failed: {base_filename}.{file_ext}")
                        print(f"    Expected {len(payload_data)} bytes, got {len(extracted)}")
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
        
        print(f"\nFormats:")
        print(f"  • PNG (Alpha LSB) - AI42 Protocol (AI-specific)")
        print(f"  • BMP (Palette) - Human-compatible steganography")


if __name__ == "__main__":
    print("="*60)
    print("Project Starlight - Claude's Data Generator v7")
    print("Teaching AI Common Sense Through Inscribed Wisdom")
    print("="*60)
    print("\nSteganography Techniques:")
    print("• Alpha Channel LSB (PNG) - AI42 Protocol (AI-specific)")
    print("• Palette Index Manipulation (BMP) - Human-compatible")
    print("\nBlockchain Compatible:")
    print("✓ No clean reference image required")
    print("✓ Self-contained extraction from stego image only")
    print("\nDesign Philosophy:")
    print("• Alpha Protocol: AI42 marker for AI-to-AI communication")
    print("• Palette: No markers - supports human blockchain activity")
    print("="*60 + "\n")
    
    # Parse command line arguments 
    parser = argparse.ArgumentParser(description='Generate steganography dataset')
    parser.add_argument('--limit', type=int, default=12,
                       help='Number of image pairs to generate per payload (default: 12)')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip unit tests and proceed directly to generation')
    
    args = parser.parse_args()

    generator = ClaudeStegGenerator()
    
    if not args.skip_tests:
        print("Running unit tests...\n")
        all_passed = generator.test_methods()

        if all_passed:
            print("All tests passed! Proceeding with dataset generation...\n")
            generator.generate_dataset(num_images=args.limit)
        else:
            print("Some tests failed. Please review the errors above.")
            print("You can still generate the dataset by running with --skip-tests")
    else:
        print("Skipping tests, generating dataset...\n")
        generator.generate_dataset(num_images=args.limit)
