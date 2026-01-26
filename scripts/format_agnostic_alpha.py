#!/usr/bin/env python3
"""
Format-Agnostic Alpha Detection
Modifies the model input processing to eliminate format bias for alpha detection.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def load_format_agnostic_input(path):
    """
    Format-agnostic version of load_unified_input that eliminates format bias.
    Treats PNG and WebP identically for alpha channel processing.
    """
    # Read file once to avoid multiple I/O operations
    with open(path, "rb") as f:
        raw_bytes = f.read()

    # Load image from bytes to avoid re-reading file
    from io import BytesIO

    img = Image.open(BytesIO(raw_bytes))

    # Store original format for debugging but don't use it for features
    original_format = img.format.lower() if img.format else "unknown"

    # Convert all images to consistent internal format for processing
    # This eliminates format-based bias
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Convert to RGB for pixel tensor (needed for both modes)
    rgb_img = img.convert("RGB")
    base_crop = transforms.CenterCrop((256, 256))(rgb_img)
    pixel_tensor = transforms.ToTensor()(base_crop)

    # Alpha channel extraction (format-agnostic)
    if "A" in img.getbands():
        alpha_crop = transforms.CenterCrop((256, 256))(img.getchannel("A"))
        alpha = transforms.ToTensor()(alpha_crop)
    else:
        alpha = torch.zeros(1, 256, 256, dtype=torch.float32)
        alpha_crop = None

    # Full mode with format-agnostic processing
    rgb_img = img.convert("RGB")

    # LSB extraction (format-agnostic)
    base_crop = transforms.CenterCrop((256, 256))(rgb_img)
    base_array = np.array(base_crop)
    lsb_array = (base_array & 1).astype(np.uint8)
    lsb = torch.from_numpy(lsb_array.astype(np.float32))

    # Alpha channel extraction (format-agnostic)
    if "A" in img.getbands():
        alpha_crop = transforms.CenterCrop((256, 256))(img.getchannel("A"))
        alpha = transforms.ToTensor()(alpha_crop)
    else:
        alpha = torch.zeros(1, 256, 256, dtype=torch.float32)

    # Palette extraction (format-agnostic)
    if img.mode == "P":
        palette_data = img.getpalette()
        if palette_data:
            palette_array = np.array(palette_data[:768], dtype=np.float32) / 255.0
            if len(palette_array) < 768:
                palette_array = np.pad(
                    palette_array, (0, 768 - len(palette_array)), "constant"
                )
            palette = torch.from_numpy(palette_array)
        else:
            palette = torch.zeros(768, dtype=torch.float32)
    else:
        palette = torch.zeros(768, dtype=torch.float32)

    # Palette LSB extraction (format-agnostic)
    if img.mode == "P":
        indexed_img = img.convert("L")
        indexed_array = np.array(transforms.CenterCrop((256, 256))(indexed_img))
        palette_lsb = torch.from_numpy(
            (indexed_array & 1).astype(np.float32)
        ).unsqueeze(0)
    else:
        palette_lsb = torch.zeros(1, 256, 256, dtype=torch.float32)

    # Metadata extraction (format-agnostic)
    try:
        from scripts.starlight_utils import extract_post_tail

        tail = extract_post_tail(raw_bytes, format_hint="auto")
        if len(tail) > 2048:
            tail = tail[:2048]
        meta_array = np.frombuffer(tail + b"\x00" * (2048 - len(tail)), dtype=np.uint8)
        meta = torch.from_numpy(meta_array.astype(np.float32)) / 255.0
    except:
        meta = torch.zeros(2048, dtype=torch.float32)

    # NEUTRAL format features - eliminate format bias
    format_features = torch.zeros(6, dtype=torch.float32)
    format_features[5] = 1.0  # Use "unknown" format for all images

    # Content features (format-agnostic)
    # LSB content analysis
    lsb_r, lsb_g, lsb_b = lsb_array[:, :, 0], lsb_array[:, :, 1], lsb_array[:, :, 2]
    lsb_bytes = np.packbits(np.stack([lsb_r, lsb_g, lsb_b], axis=0).flatten())

    try:
        from scripts.starlight_utils import _calculate_content_features

        lsb_content_features = _calculate_content_features(lsb_bytes)
    except:
        lsb_content_features = torch.zeros(3, dtype=torch.float32)

    # Alpha content analysis
    if "A" in img.getbands() and alpha_crop is not None:
        alpha_array = np.array(alpha_crop)
        alpha_bytes = alpha_array.flatten()

        try:
            from scripts.starlight_utils import _calculate_content_features

            alpha_content_features = _calculate_content_features(alpha_bytes)
        except:
            alpha_content_features = torch.zeros(3, dtype=torch.float32)
    else:
        alpha_content_features = torch.zeros(3, dtype=torch.float32)

    content_features = torch.cat([lsb_content_features, alpha_content_features])

    return (
        pixel_tensor,
        meta,
        alpha,
        lsb,
        palette,
        palette_lsb,
        format_features,
        content_features,
    )


def patch_starlight_utils():
    """
    Monkey-patch starlight_utils to use format-agnostic processing
    """
    import sys

    sys.path.append(".")
    import starlight_utils

    original_load = starlight_utils.load_unified_input
    starlight_utils.load_unified_input = load_format_agnostic_input
    print("Patched starlight_utils.load_unified_input with format-agnostic version")
    return original_load


def restore_starlight_utils(original_load):
    """
    Restore original starlight_utils function
    """
    import sys

    sys.path.append(".")
    import starlight_utils

    starlight_utils.load_unified_input = original_load
    print("Restored original starlight_utils.load_unified_input")


def test_format_agnostic_detection():
    """
    Test format-agnostic detection on problematic files
    """
    print("Testing format-agnostic alpha detection...")

    # Test files that were failing
    test_files = [
        "/Users/eric/sandbox/starlight/datasets/claude_submission_2025/stego/common_sense_in_uncommon_times_alpha_040.png",
        "/Users/eric/sandbox/starlight/datasets/gemini_submission_2025/stego/ai_common_sense_on_blockchain_alpha_004.png",
        "/Users/eric/sandbox/starlight/datasets/chatgpt_submission_2025/stego/sample_seed_alpha_119.webp",
    ]

    # Patch the utils
    original_load = patch_starlight_utils()

    try:
        import scanner
        import sys

        sys.path.append("./scripts")
        from starlight_utils import load_unified_input

        for test_file in test_files:
            if not os.path.exists(test_file):
                print(f"File not found: {test_file}")
                continue

            print(f"\nTesting: {test_file}")

            # Load with format-agnostic processing
            try:
                inputs = load_unified_input(test_file)
                print(f"  ✓ Loaded successfully with format-agnostic processing")
                print(f"  Alpha tensor shape: {inputs[2].shape}")
                print(f"  Format features: {inputs[6]}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

            # Test with scanner
            try:
                result = scanner._scan_logic(test_file, None)
                print(f"  Scanner result: {result}")
            except Exception as e:
                print(f"  ✗ Scanner error: {e}")

    finally:
        # Restore original
        restore_starlight_utils(original_load)


if __name__ == "__main__":
    import os

    test_format_agnostic_detection()
