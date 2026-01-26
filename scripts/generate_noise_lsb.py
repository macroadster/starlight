#!/usr/bin/env python3
"""
Generate noise LSB category - 5th negative example type
"""

import numpy as np
from pathlib import Path
from PIL import Image
import random


def generate_noise_lsb(count: int = 1000):
    """Generate images with natural LSB noise patterns"""
    output_dir = Path("datasets/grok_submission_2025/training/v3_negatives/noise_lsb")
    output_dir.mkdir(exist_ok=True)

    print(f"Generating {count} images with natural LSB noise...")

    sizes = [(128, 128), (256, 256), (512, 512)]

    for i in range(count):
        size = random.choice(sizes)

        # Create base image with some structure
        base_data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)

        # Add LSB-specific noise patterns
        noise_type = i % 3

        if noise_type == 0:
            # LSB bit flipping noise
            for y in range(size[1]):
                for x in range(size[0]):
                    # Randomly flip LSB bits
                    if random.random() < 0.1:  # 10% chance
                        for c in range(3):
                            base_data[y, x, c] ^= 1  # Flip LSB

        elif noise_type == 1:
            # Natural LSB variation from compression
            base_data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            # Add subtle LSB patterns
            pattern = np.random.randint(0, 4, (size[1], size[0]), dtype=np.uint8)
            for c in range(3):
                base_data[:, :, c] = (base_data[:, :, c] & 0xFE) | (pattern & 0x01)

        else:
            # Environmental noise (camera sensor, transmission)
            base_data = np.random.randint(
                100, 200, (size[1], size[0], 3), dtype=np.uint8
            )
            noise = np.random.normal(0, 2, (size[1], size[0], 3))
            base_data = np.clip(base_data + noise, 0, 255).astype(np.uint8)
            # Add LSB variations
            lsb_noise = np.random.randint(0, 2, (size[1], size[0], 3), dtype=np.uint8)
            base_data = (base_data & 0xFE) | lsb_noise

        img = Image.fromarray(base_data)
        img.save(output_dir / f"lsb_noise_{i:04d}.png")

    print(f"✅ Generated {count} LSB noise images")


if __name__ == "__main__":
    generate_noise_lsb(1000)
