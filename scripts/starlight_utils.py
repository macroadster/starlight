import torch
import numpy as np
from PIL import Image
import struct
import torchvision.transforms as transforms
import time
from pathlib import Path


def get_submission_dirs(base_path="datasets"):
    """
    Finds all submission dataset directories within the base path.
    A submission directory is any non-hidden subdirectory.
    """
    base = Path(base_path)
    if not base.is_dir():
        return []

    submission_dirs = []
    for item in base.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            submission_dirs.append(item)

    return sorted(submission_dirs)


def _find_gif_end_pos(raw_bytes):
    """
    Parses a GIF file to find the offset of the byte AFTER the trailer (';').
    Returns -1 if the file is not a valid GIF or is truncated.
    """
    if not (raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a")):
        return -1

    pos = 6  # Start after header

    try:
        # --- Logical Screen Descriptor (7 bytes) ---
        if pos + 7 > len(raw_bytes):
            return -1
        screen_descriptor = raw_bytes[pos : pos + 7]
        pos += 7

        # --- Global Color Table (if present) ---
        flags = screen_descriptor[4]
        if flags & 0x80:  # GCT flag is set
            color_table_size = 1 << ((flags & 0x07) + 1)
            if pos + 3 * color_table_size > len(raw_bytes):
                return -1
            pos += 3 * color_table_size

        # --- Main loop for blocks ---
        while pos < len(raw_bytes):
            block_type = raw_bytes[pos : pos + 1]
            if not block_type:
                break
            pos += 1

            if block_type == b";":  # Trailer
                return pos

            elif block_type == b"!":  # Extension Block
                if pos >= len(raw_bytes):
                    return -1
                pos += 1  # Skip extension label
                while True:
                    if pos >= len(raw_bytes):
                        return -1
                    block_size = raw_bytes[pos]
                    pos += 1
                    if block_size == 0:
                        break
                    if pos + block_size > len(raw_bytes):
                        return -1
                    pos += block_size

            elif block_type == b",":  # Image Descriptor Block
                if pos + 9 > len(raw_bytes):
                    return -1
                descriptor = raw_bytes[pos : pos + 9]
                pos += 9

                img_flags = descriptor[8]
                if img_flags & 0x80:
                    lct_size = 1 << ((img_flags & 0x07) + 1)
                    if pos + 3 * lct_size > len(raw_bytes):
                        return -1
                    pos += 3 * lct_size

                if pos >= len(raw_bytes):
                    return -1
                pos += 1  # Skip LZW Minimum Code Size

                while True:
                    if pos >= len(raw_bytes):
                        return -1
                    block_size = raw_bytes[pos]
                    pos += 1
                    if block_size == 0:
                        break
                    if pos + block_size > len(raw_bytes):
                        return -1
                    pos += block_size

            else:
                return -1  # Unknown block type, malformed GIF

        return -1  # Trailer not found

    except (IndexError, struct.error):
        return -1  # Truncated or malformed


def extract_post_tail(raw_bytes, format_hint="auto"):
    """
    Fast extraction of bytes after the official end of image data.
    Optimized for performance by early termination and simplified parsing.
    """
    # Quick check: if file is not much larger than expected, skip expensive parsing
    if len(raw_bytes) < 1000:  # Most clean images are small
        return b""

    tail = b""
    if format_hint == "jpeg":
        # Fast JPEG end marker search
        pos = raw_bytes.rfind(b"\xff\xd9")
        if pos != -1 and pos + 2 < len(raw_bytes):
            tail = raw_bytes[pos + 2 :]
    elif format_hint == "png":
        # Fast PNG IEND chunk search
        pos = raw_bytes.rfind(b"IEND")
        if pos != -1 and pos + 12 < len(raw_bytes):
            tail = raw_bytes[pos + 12 :]  # Fixed 12-byte offset for IEND chunk
    elif format_hint == "gif":
        # Simplified GIF parsing - just look for trailer
        pos = raw_bytes.rfind(b";")
        if pos != -1 and pos + 1 < len(raw_bytes):
            tail = raw_bytes[pos + 1 :]
    elif format_hint == "webp":
        # Fast WebP parsing
        if (
            len(raw_bytes) >= 12
            and raw_bytes.startswith(b"RIFF")
            and raw_bytes[8:12] == b"WEBP"
        ):
            try:
                riff_size = struct.unpack("<I", raw_bytes[4:8])[0]
                expected_size = 8 + riff_size
                if (
                    len(raw_bytes) > expected_size + 100
                ):  # Only extract if significant tail
                    tail = raw_bytes[expected_size:]
            except struct.error:
                pass  # Malformed header

    # Limit tail size to prevent memory issues with malformed files
    if len(tail) > 2048:
        tail = tail[:2048]

    return tail


def _calculate_content_features(data_bytes):
    """Fast calculation of content features from byte data."""
    if data_bytes.size == 0:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    total_chars = len(data_bytes)

    # Optimized unique character calculation
    if total_chars <= 256:
        unique_chars = np.unique(data_bytes)
        uniqueness_ratio = len(unique_chars) / total_chars
    else:
        # For larger data, sample for speed
        sample = data_bytes[:: max(1, total_chars // 1000)]
        unique_chars = np.unique(sample)
        uniqueness_ratio = min(
            1.0, len(unique_chars) / 256
        )  # Cap at 256 possible values

    # Vectorized printable character calculation
    printable_mask = (data_bytes >= 32) & (data_bytes <= 126)
    printable_char_ratio = np.sum(printable_mask) / total_chars

    # Fast most common character calculation
    if total_chars > 0:
        counts = np.bincount(data_bytes, minlength=256)
        most_common_char_ratio = np.max(counts) / total_chars
    else:
        most_common_char_ratio = 0.0

    return torch.tensor(
        [uniqueness_ratio, printable_char_ratio, most_common_char_ratio],
        dtype=torch.float32,
    )


def load_unified_input(path_or_bytes):
    # Read file or use provided bytes
    if isinstance(path_or_bytes, bytes):
        raw_bytes = path_or_bytes
    else:
        with open(path_or_bytes, "rb") as f:
            raw_bytes = f.read()

    # Load image from bytes to avoid re-reading file
    from io import BytesIO

    img = Image.open(BytesIO(raw_bytes))

    # --- Augmentation ---
    # Augmentations should be done on RGB version of image
    rgb_img = img.convert("RGB")

    # --- LSB Path (Pre-Augmentation) ---
    # Critical fix: LSB must be extracted from the original, un-augmented image data
    # to preserve steganographic signal.
    # Ensure LSB tensor is 3 channels, 256x256

    # First, crop the original RGB image to target size
    base_crop = transforms.CenterCrop((256, 256))(rgb_img)

    # Always extract LSB accurately - this is critical for detection accuracy
    base_array = np.array(base_crop)
    lsb_array = (base_array & 1).astype(np.uint8)
    lsb = torch.from_numpy(lsb_array.astype(np.float32))

    # Full mode: complete LSB extraction and analysis
    lsb_r, lsb_g, lsb_b = lsb_array[:, :, 0], lsb_array[:, :, 1], lsb_array[:, :, 2]
    lsb_bytes = np.packbits(np.stack([lsb_r, lsb_g, lsb_b], axis=0).flatten())
    lsb_content_features = _calculate_content_features(lsb_bytes)

    # --- Pixel Tensor (Post-Augmentation) ---
    # Now, apply augmentations for training robustness
    # For now, we use a standard crop, but this is where other transforms would go.
    aug_img = base_crop
    pixel_tensor = transforms.ToTensor()(aug_img)

    # --- Metadata Path ---
    # Full mode: complete metadata analysis
    exif = img.info.get("exif", b"")  # Use Pillow's built-in EXIF extraction

    # Always extract tail for EOI steganography (remove size limit)
    format_hint = img.format.lower() if img.format else "auto"
    tail = extract_post_tail(raw_bytes, format_hint)

    meta_bytes = np.frombuffer(exif + tail, dtype=np.uint8)[:2048]
    meta_bytes = np.pad(meta_bytes, (0, 2048 - len(meta_bytes)), "constant")
    meta = torch.from_numpy(meta_bytes.astype(np.float32) / 255.0)

    # --- Alpha Path ---
    # Ensure alpha tensor is 1 channel, 256x256
    has_alpha = 1.0 if img.mode == "RGBA" else 0.0
    alpha_std_dev = 0.0
    if img.mode == "RGBA":
        alpha_plane = np.array(img.split()[-1])
        alpha_std_dev = alpha_plane.std() / 255.0  # Normalize

        full_alpha = torch.from_numpy(alpha_plane.astype(np.float32) / 255.0).unsqueeze(
            0
        )
        lsb_alpha = torch.from_numpy((alpha_plane & 1).astype(np.float32)).unsqueeze(0)

        alpha = torch.cat([full_alpha, lsb_alpha], dim=0)
        alpha = transforms.CenterCrop((256, 256))(alpha)

        # Full mode: complete alpha analysis
        alpha_lsb = (alpha_plane & 1).astype(np.uint8)
        alpha_bytes = np.packbits(alpha_lsb.flatten())
        alpha_content_features = _calculate_content_features(alpha_bytes)
    else:
        alpha = torch.zeros(2, 256, 256)
        alpha_content_features = torch.zeros(3, dtype=torch.float32)

    # --- Palette Path ---
    # Ensure palette tensor is 768 elements
    if img.format == "GIF" and img.mode == "L":
        img = img.convert("P")

    if img.mode == "P":
        # Extract palette colors
        palette_bytes = np.array(img.getpalette(), dtype=np.uint8)
        palette_bytes = np.pad(palette_bytes, (0, 768 - len(palette_bytes)), "constant")
        palette = torch.from_numpy(palette_bytes.astype(np.float32) / 255.0)

        # Extract LSB patterns from palette indices (image pixel data)
        # Convert to 'L' mode to get the palette indices
        indexed_img = img.convert("L")
        indexed_array = np.array(transforms.CenterCrop((256, 256))(indexed_img))
        palette_lsb = torch.from_numpy(
            (indexed_array & 1).astype(np.float32)
        ).unsqueeze(0)
    else:
        palette_bytes = np.zeros(768, dtype=np.uint8)
        palette = torch.from_numpy(palette_bytes.astype(np.float32) / 255.0)
        palette_lsb = torch.zeros(
            1, 256, 256, dtype=torch.float32
        )  # Default for non-palette images

    # --- Format Features Path ---
    # Simplified format features to match the model's expectations.
    width, height = img.size
    has_alpha = 1.0 if img.mode == "RGBA" else 0.0
    alpha_std_dev = 0.0
    if has_alpha:
        alpha_plane = np.array(img.split()[-1])
        alpha_std_dev = alpha_plane.std() / 255.0  # Normalize

    format_features = torch.tensor(
        [
            has_alpha,
            alpha_std_dev,
            1.0 if img.mode == "P" else 0.0,  # is_palette
            1.0 if img.mode == "RGB" else 0.0,  # is_rgb
            float(width) / 256.0,  # width_norm
            float(height) / 256.0,  # height_norm
        ],
        dtype=torch.float32,
    )

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
