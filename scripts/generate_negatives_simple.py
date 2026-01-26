#!/usr/bin/env python3
"""
Simple Negative Examples Generator - Fast version without validation
Generates examples teaching what steganography is NOT

Run: python scripts/generate_negatives_simple.py --output datasets/grok_submission_2025/training/v3_negatives --count 1000
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import random
from typing import List, Tuple
import json
import datetime


class SimpleNegativeGenerator:
    """Generate examples teaching what steganography is NOT (no validation)"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def rgb_with_alpha_check(self, count: int = 100):
        """Generate RGB images that should NOT be detected as alpha steganography"""
        print(f"Generating {count} RGB images (no alpha)...")

        output_subdir = self.output_dir / "rgb_no_alpha"
        output_subdir.mkdir(exist_ok=True)

        sizes = [(128, 128), (256, 256), (512, 512)]

        for i in range(count):
            size = random.choice(sizes)

            # Generate diverse RGB content
            img_type = i % 5

            if img_type == 0:
                # Solid colors
                color = tuple(random.randint(0, 255) for _ in range(3))
                img = Image.new("RGB", size, color)

            elif img_type == 1:
                # Random noise
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data)

            elif img_type == 2:
                # Gradients
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                for y in range(size[1]):
                    for x in range(size[0]):
                        data[y, x] = [
                            int(255 * x / size[0]),
                            int(255 * y / size[1]),
                            128,
                        ]
                img = Image.fromarray(data)

            elif img_type == 3:
                # Patterns
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                for y in range(size[1]):
                    for x in range(size[0]):
                        data[y, x] = [(x * y) % 256, (x + y) % 256, (x - y) % 256]
                img = Image.fromarray(data)

            else:
                # Mixed regions
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                # Add some structure
                data[::10, :] = [255, 0, 0]  # Red lines
                data[:, ::10] = [0, 255, 0]  # Green lines
                img = Image.fromarray(data)

            img.save(output_subdir / f"rgb_no_alpha_{i:04d}.png")

        print(f"✅ Generated {count} RGB images in {output_subdir}")

    def uniform_alpha_images(self, count: int = 100):
        """Generate RGBA images with uniform alpha (no hidden data)"""
        print(f"Generating {count} RGBA images with uniform alpha...")

        output_subdir = self.output_dir / "uniform_alpha"
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

            img = Image.fromarray(rgba_data)
            img.save(output_subdir / f"uniform_alpha_{i:04d}_{alpha}.png")

        print(f"✅ Generated {count} uniform alpha images in {output_subdir}")

    def dithered_gif(self, count: int = 100):
        """Generate images with natural LSB variation"""
        print(f"Generating {count} images with natural LSB noise...")

        output_subdir = self.output_dir / "dithered_gif"
        output_subdir.mkdir(exist_ok=True)

        sizes = [(128, 128), (256, 256), (512, 512)]

        for i in range(count):
            size = random.choice(sizes)

            noise_type = i % 4

            if noise_type == 0:
                # GIF dithering effect
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data)
                img = img.convert(
                    "P",
                    dither=Image.Dither.FLOYDSTEINBERG,
                    palette=Image.Palette.ADAPTIVE,
                )
                img = img.convert("RGB")

            elif noise_type == 1:
                # JPEG compression artifacts
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data)
                # Simulate JPEG compression
                temp_path = output_subdir / "temp.jpg"
                img.save(temp_path, "JPEG", quality=75)
                img = Image.open(temp_path)
                temp_path.unlink()

            elif noise_type == 2:
                # Posterization
                data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                img = Image.fromarray(data)
                img = img.quantize(colors=64)
                img = img.convert("RGB")

            else:
                # Natural photo-like noise
                base = np.random.randint(
                    100, 200, (size[1], size[0], 3), dtype=np.uint8
                )
                noise = np.random.normal(0, 5, (size[1], size[0], 3))
                data = np.clip(base + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(data)

            img.save(output_subdir / f"natural_noise_{i:04d}.png")

        print(f"✅ Generated {count} natural noise images in {output_subdir}")

    def repetitive_hex(self, count: int = 100):
        """Generate images with repetitive patterns (not steganography)"""
        print(f"Generating {count} images with repetitive patterns...")

        output_subdir = self.output_dir / "repetitive_hex"
        output_subdir.mkdir(exist_ok=True)

        sizes = [(128, 128), (256, 256), (512, 512)]

        for i in range(count):
            size = random.choice(sizes)

            pattern_type = i % 5

            if pattern_type == 0:
                # Solid colors
                colors = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 255),
                    (0, 0, 0),
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
                            int(255 * ((x + y) / (size[0] + size[1]))),
                        ]

            else:
                # Concentric circles
                data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                center_x, center_y = size[0] // 2, size[1] // 2
                for y in range(size[1]):
                    for x in range(size[0]):
                        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        value = int((np.sin(dist / 10) + 1) * 127.5)
                        data[y, x] = [value, value, value]

            img = Image.fromarray(data)
            img.save(output_subdir / f"pattern_{i:04d}.png")

        print(f"✅ Generated {count} pattern images in {output_subdir}")

    def generate_all(self, count_per_type: int = 1000):
        """Generate all types of negative examples"""
        print("\n" + "=" * 70)
        print("GENERATING NEGATIVE EXAMPLES (FAST MODE - NO VALIDATION)")
        print("=" * 70 + "\n")

        self.rgb_with_alpha_check(count_per_type)
        self.uniform_alpha_images(count_per_type)
        self.dithered_gif(count_per_type)
        self.repetitive_hex(count_per_type)

        total = count_per_type * 4

        print("\n" + "=" * 70)
        print(f"✅ COMPLETE: Generated {total} negative examples")
        print("=" * 70)

        # Generate manifest
        constraints = {
            "rgb_no_alpha": "RGB images cannot have alpha steganography",
            "uniform_alpha": "Uniform alpha channel contains no hidden data",
            "dithered_gif": "GIF dithering is natural noise, not steganography",
            "repetitive_hex": "Repetitive hex patterns are visible, not hidden",
        }

        manifest_path = self.output_dir / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for image in sorted(self.output_dir.rglob("*")):
                if image.is_file() and image.suffix.lower() in [".png", ".gif"]:
                    method = image.parent.name
                    if method in constraints:
                        entry = {
                            "method": method,
                            "constraint": constraints[method],
                            "label": "clean",
                            "file_path": str(image.relative_to(self.output_dir)),
                        }
                        f.write(json.dumps(entry) + "\n")

        print(f"Manifest created: {manifest_path}")
        print(
            "Note: Validation skipped for speed. Run validation separately if needed."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate negative examples for steganography detection training (fast mode)"
    )
    parser.add_argument(
        "--output",
        default="datasets/grok_submission_2025/training/v3_negatives",
        help="Output directory for negative examples",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of examples per category (default: 100)",
    )

    args = parser.parse_args()

    generator = SimpleNegativeGenerator(Path(args.output))
    generator.generate_all(args.count)

    return 0


if __name__ == "__main__":
    exit(main())
