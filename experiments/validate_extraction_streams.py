import torch
import numpy as np
from PIL import Image
import argparse
import os
import sys

# Add the parent directory to the Python path to allow importing from 'scripts'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.starlight_utils import load_unified_input

def validate_stream(tensor, name):
    """Prints validation information for a single tensor stream."""
    print(f"--- {name} Stream ---")
    if tensor is None:
        print("  Tensor is None")
        return

    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")

    # Don't calculate stats for large tensors to avoid performance issues
    if tensor.numel() > 10000:
        print("  (Stats omitted for large tensor)")
    else:
        print(f"  Min: {tensor.min().item():.4f}")
        print(f"  Max: {tensor.max().item():.4f}")
        print(f"  Mean: {tensor.mean().item():.4f}")
    print("-" * (len(name) + 12))


def main():
    parser = argparse.ArgumentParser(description="Validate the 6 extraction streams of the Starlight pipeline.")
    parser.add_argument("image_paths", nargs="+", help="Paths to the image files to validate.")
    args = parser.parse_args()

    for image_path in args.image_paths:
        if not os.path.exists(image_path):
            print(f"Error: Image not found at '{image_path}'")
            continue

        print(f"\n{'='*60}")
        print(f"VALIDATING: {os.path.basename(image_path)}")
        print(f"{ '='*60}\n")

        try:
            img = Image.open(image_path)
            print(f"Image Mode: {img.mode}")
            print(f"Image Palette: {img.getpalette()}")

            # The new load_unified_input returns 8 tensors
            pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features = load_unified_input(image_path)

            validate_stream(pixel_tensor, "Pixel")
            validate_stream(meta, "Metadata")
            validate_stream(alpha, "Alpha")
            validate_stream(lsb, "LSB")
            validate_stream(palette, "Palette")
            validate_stream(palette_lsb, "Palette LSB")
            validate_stream(format_features, "Format Features")
            validate_stream(content_features, "Content Features")

        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}")

if __name__ == "__main__":
    main()
