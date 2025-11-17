#!/usr/bin/env python3
"""
Negative Examples Generator - Week 1, Day 3
Generates examples teaching what steganography is NOT

Run: python scripts/generate_negatives.py --output data/training/v3_repaired/negatives --count 1000
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import random
from typing import List, Tuple

class NegativeExampleGenerator:
    """Generate examples teaching what steganography is NOT"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def rgb_with_alpha_check(self, count: int = 100):
        """
        Generate RGB images that should NOT be detected as alpha steganography
        
        This teaches the model: RGB images cannot have alpha steganography
        """
        print(f"Generating {count} RGB images (no alpha)...")
        
        output_subdir = self.output_dir / 'rgb_no_alpha'
        output_subdir.mkdir(exist_ok=True)
        
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for i in range(count):
            size = random.choice(sizes)
            
            # Generate diverse RGB content
            img_type = i % 5
            
            if img_type == 0:
                # Solid colors
                color = tuple(random.randint(0, 255) for _ in range(3))
                img = Image.new('RGB', size, color)
            
            elif img_type == 1:
                # Random noise
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')
            
            elif img_type == 2:
                # Gradients
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                for y in range(size[1]):
                    for x in range(size[0]):
                        data[y, x] = [
                            int(255 * x / size[0]),
                            int(255 * y / size[1]),
                            128
                        ]
                img = Image.fromarray(data, 'RGB')
            
            elif img_type == 3:
                # Patterns
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                for y in range(size[1]):
                    for x in range(size[0]):
                        data[y, x] = [
                            (x * y) % 256,
                            (x + y) % 256,
                            (x - y) % 256
                        ]
                img = Image.fromarray(data, 'RGB')
            
            else:
                # Mixed regions
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                # Add some structure
                data[::10, :] = [255, 0, 0]  # Red lines
                data[:, ::10] = [0, 255, 0]  # Green lines
                img = Image.fromarray(data, 'RGB')
            
            img.save(output_subdir / f'rgb_no_alpha_{i:04d}.png')
        
        print(f"✅ Generated {count} RGB images in {output_subdir}")
    
    def uniform_alpha_images(self, count: int = 100):
        """
        Generate RGBA images with uniform alpha (no hidden data)
        
        This teaches: Uniform alpha channel = no steganography
        """
        print(f"Generating {count} RGBA images with uniform alpha...")
        
        output_subdir = self.output_dir / 'uniform_alpha'
        output_subdir.mkdir(exist_ok=True)
        
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for i in range(count):
            size = random.choice(sizes)
            
            # Generate RGB content (varied)
            rgb_data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            
            # Create RGBA with UNIFORM alpha
            alpha_values = [0, 128, 255]  # Common uniform alpha values
            alpha = alpha_values[i % len(alpha_values)]
            
            rgba_data = np.zeros((size[1], size[0], 4), dtype=np.uint8)
            rgba_data[:, :, :3] = rgb_data
            rgba_data[:, :, 3] = alpha  # Uniform alpha
            
            img = Image.fromarray(rgba_data, 'RGBA')
            img.save(output_subdir / f'uniform_alpha_{i:04d}_{alpha}.png')
        
        print(f"✅ Generated {count} uniform alpha images in {output_subdir}")
    
    def natural_lsb_noise(self, count: int = 100):
        """
        Generate clean images with natural LSB variation
        
        This teaches: Natural dithering/compression artifacts â‰  steganography
        """
        print(f"Generating {count} images with natural LSB noise...")
        
        output_subdir = self.output_dir / 'natural_noise'
        output_subdir.mkdir(exist_ok=True)
        
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for i in range(count):
            size = random.choice(sizes)
            
            noise_type = i % 4
            
            if noise_type == 0:
                # GIF dithering effect
                # Create color image, convert to palette with dithering
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')
                img = img.convert('P', dither=Image.FLOYDSTEINBERG, palette=Image.ADAPTIVE)
                img = img.convert('RGB')  # Convert back
            
            elif noise_type == 1:
                # JPEG compression artifacts (save/load cycle)
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')
                # Simulate JPEG compression
                temp_path = output_subdir / 'temp.jpg'
                img.save(temp_path, 'JPEG', quality=75)
                img = Image.open(temp_path)
                temp_path.unlink()
            
            elif noise_type == 2:
                # Posterization (reduces bit depth, creates LSB patterns)
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')
                # Posterize (reduce to fewer colors)
                img = img.quantize(colors=64)
                img = img.convert('RGB')
            
            else:
                # Natural photo-like noise
                # Smooth base + noise
                base = np.random.randint(100, 200, (size[1], size[0], 3), dtype=np.uint8)
                noise = np.random.normal(0, 5, (size[1], size[0], 3))
                data = np.clip(base + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(data, 'RGB')
            
            img.save(output_subdir / f'natural_noise_{i:04d}.png')
        
        print(f"✅ Generated {count} natural noise images in {output_subdir}")
    
    def repetitive_patterns(self, count: int = 100):
        """
        Generate images with repetitive patterns (not steganography)
        
        This teaches: Repetitive hex patterns â‰  hidden data
        """
        print(f"Generating {count} images with repetitive patterns...")
        
        output_subdir = self.output_dir / 'patterns'
        output_subdir.mkdir(exist_ok=True)
        
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for i in range(count):
            size = random.choice(sizes)
            
            pattern_type = i % 5
            
            if pattern_type == 0:
                # Solid colors (extreme case)
                colors = [
                    (255, 0, 0),    # Red
                    (0, 255, 0),    # Green
                    (0, 0, 255),    # Blue
                    (255, 255, 255), # White
                    (0, 0, 0),      # Black
                ]
                color = colors[i % len(colors)]
                data = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            
            elif pattern_type == 1:
                # Checkerboard
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                checker_size = 16
                for y in range(size[1]):
                    for x in range(size[0]):
                        if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                            data[y, x] = [255, 255, 255]
                        else:
                            data[y, x] = [0, 0, 0]
            
            elif pattern_type == 2:
                # Stripes
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                stripe_width = 10
                for y in range(size[1]):
                    for x in range(size[0]):
                        if (x // stripe_width) % 2 == 0:
                            data[y, x] = [255, 0, 0]
                        else:
                            data[y, x] = [0, 0, 255]
            
            elif pattern_type == 3:
                # Gradients
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                for y in range(size[1]):
                    for x in range(size[0]):
                        data[y, x] = [
                            int(255 * x / size[0]),
                            int(255 * y / size[1]),
                            int(255 * ((x + y) / (size[0] + size[1])))
                        ]
            
            else:
                # Concentric circles
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                center_x, center_y = size[0] // 2, size[1] // 2
                for y in range(size[1]):
                    for x in range(size[0]):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        value = int((np.sin(dist / 10) + 1) * 127.5)
                        data[y, x] = [value, value, value]
            
            img = Image.fromarray(data, 'RGB')
            img.save(output_subdir / f'pattern_{i:04d}.png')
        
        print(f"✅ Generated {count} pattern images in {output_subdir}")
    
    def special_case_negatives(self, count: int = 50):
        """
        Generate specific negatives for each special case rule
        
        This directly targets the special cases that models currently fail to learn
        """
        print(f"Generating {count} special case negatives...")
        
        output_subdir = self.output_dir / 'special_cases'
        output_subdir.mkdir(exist_ok=True)
        
        for i in range(count):
            case_type = i % 5
            size = (256, 256)
            
            if case_type == 0:
                # Small images (< 64x64)
                small_size = (random.randint(10, 63), random.randint(10, 63))
                data = np.random.randint(0, 256, (small_size[1], small_size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')
                img.save(output_subdir / f'small_image_{i:04d}.png')
            
            elif case_type == 1:
                # Palette images without sufficient colors
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')
                # Convert to palette with very few colors
                img = img.quantize(colors=8)
                img.save(output_subdir / f'few_colors_{i:04d}.png')
            
            elif case_type == 2:
                # High-frequency noise (looks suspicious but isn't stego)
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')
                img.save(output_subdir / f'high_freq_noise_{i:04d}.png')
            
            elif case_type == 3:
                # Specific problematic formats
                # GIF with transparency
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data, 'RGB')
                img = img.convert('P', palette=Image.ADAPTIVE)
                img.info['transparency'] = 0
                img.save(output_subdir / f'gif_transparent_{i:04d}.gif')
            
            else:
                # Monochrome/grayscale
                value = random.randint(0, 255)
                data = np.full((size[1], size[0]), value, dtype=np.uint8)
                img = Image.fromarray(data, 'L')
                img.save(output_subdir / f'grayscale_{i:04d}.png')
        
        print(f"✅ Generated {count} special case negatives in {output_subdir}")
    
    def generate_all(self, count_per_type: int = 100):
        """Generate all types of negative examples"""
        print("\n" + "="*70)
        print("GENERATING NEGATIVE EXAMPLES")
        print("="*70 + "\n")
        
        self.rgb_with_alpha_check(count_per_type)
        self.uniform_alpha_images(count_per_type)
        self.natural_lsb_noise(count_per_type)
        self.repetitive_patterns(count_per_type)
        self.special_case_negatives(count_per_type // 2)
        
        total = count_per_type * 4 + count_per_type // 2
        
        print("\n" + "="*70)
        print(f"✅ COMPLETE: Generated {total} negative examples")
        print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Generate negative examples for steganography detection training'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for negative examples'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of examples per category (default: 100)'
    )
    
    args = parser.parse_args()
    
    generator = NegativeExampleGenerator(Path(args.output))
    generator.generate_all(args.count)
    
    return 0

if __name__ == '__main__':
    exit(main())
