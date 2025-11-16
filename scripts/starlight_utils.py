import torch
import numpy as np
from PIL import Image
import struct
import torchvision.transforms as transforms

def _find_gif_end_pos(raw_bytes):
    """
    Parses a GIF file to find the offset of the byte AFTER the trailer (';').
    Returns -1 if the file is not a valid GIF or is truncated.
    """
    if not (raw_bytes.startswith(b'GIF87a') or raw_bytes.startswith(b'GIF89a')):
        return -1

    pos = 6  # Start after header

    try:
        # --- Logical Screen Descriptor (7 bytes) ---
        if pos + 7 > len(raw_bytes): return -1
        screen_descriptor = raw_bytes[pos:pos+7]
        pos += 7

        # --- Global Color Table (if present) ---
        flags = screen_descriptor[4]
        if flags & 0x80:  # GCT flag is set
            color_table_size = 1 << ((flags & 0x07) + 1)
            if pos + 3 * color_table_size > len(raw_bytes): return -1
            pos += 3 * color_table_size

        # --- Main loop for blocks ---
        while pos < len(raw_bytes):
            block_type = raw_bytes[pos:pos+1]
            if not block_type: break
            pos += 1

            if block_type == b';':  # Trailer
                return pos

            elif block_type == b'!':  # Extension Block
                if pos >= len(raw_bytes): return -1
                pos += 1  # Skip extension label
                while True:
                    if pos >= len(raw_bytes): return -1
                    block_size = raw_bytes[pos]
                    pos += 1
                    if block_size == 0:
                        break
                    if pos + block_size > len(raw_bytes): return -1
                    pos += block_size

            elif block_type == b',':  # Image Descriptor Block
                if pos + 9 > len(raw_bytes): return -1
                descriptor = raw_bytes[pos:pos+9]
                pos += 9

                img_flags = descriptor[8]
                if img_flags & 0x80:
                    lct_size = 1 << ((img_flags & 0x07) + 1)
                    if pos + 3 * lct_size > len(raw_bytes): return -1
                    pos += 3 * lct_size

                if pos >= len(raw_bytes): return -1
                pos += 1  # Skip LZW Minimum Code Size

                while True:
                    if pos >= len(raw_bytes): return -1
                    block_size = raw_bytes[pos]
                    pos += 1
                    if block_size == 0:
                        break
                    if pos + block_size > len(raw_bytes): return -1
                    pos += block_size
            
            else:
                return -1 # Unknown block type, malformed GIF
        
        return -1 # Trailer not found

    except (IndexError, struct.error):
        return -1 # Truncated or malformed

def extract_post_tail(raw_bytes, format_hint='auto'):
    tail = b""
    if format_hint == 'jpeg':
        pos = raw_bytes.rfind(b'\xFF\xD9')
        if pos != -1:
            tail = raw_bytes[pos + 2:]
    elif format_hint == 'png':
        pos = raw_bytes.rfind(b'IEND')
        if pos != -1:
            # The IEND chunk is 12 bytes total: 4b length (0), 4b type ('IEND'), 4b CRC.
            # The tail starts after the 4-byte CRC.
            tail = raw_bytes[pos + 8:]
    elif format_hint == 'gif':
        end_pos = _find_gif_end_pos(raw_bytes)
        if end_pos != -1 and end_pos < len(raw_bytes):
            tail = raw_bytes[end_pos:]
    elif format_hint == 'webp':
        if raw_bytes.startswith(b'RIFF') and len(raw_bytes) >= 12 and raw_bytes[8:12] == b'WEBP':
            try:
                riff_size = struct.unpack('<I', raw_bytes[4:8])[0]
                expected_size = 8 + riff_size
                if len(raw_bytes) > expected_size:
                    tail = raw_bytes[expected_size:]
            except struct.error:
                pass # Malformed header
    
    return tail

def _calculate_content_features(data_bytes):
    """Calculates features from a byte string to be used in a tensor."""
    if data_bytes.size == 0:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    total_chars = len(data_bytes)
    unique_chars, counts = np.unique(data_bytes, return_counts=True)
    
    uniqueness_ratio = len(unique_chars) / total_chars
    
    printable_chars = np.sum((data_bytes >= 32) & (data_bytes <= 126))
    printable_char_ratio = printable_chars / total_chars
    
    if len(counts) == 0:
        most_common_char_ratio = 0.0
    else:
        most_common_char_ratio = np.max(counts) / total_chars
        
    return torch.tensor([uniqueness_ratio, printable_char_ratio, most_common_char_ratio], dtype=torch.float32)

def load_unified_input(path):
    img = Image.open(path)
    
    # --- Augmentation ---
    # Augmentations should be done on the RGB version of the image
    rgb_img = img.convert('RGB')
    
    # --- LSB Path (Pre-Augmentation) ---
    # Critical fix: LSB must be extracted from the original, un-augmented image data
    # to preserve the steganographic signal.
    # Ensure LSB tensor is 3 channels, 256x256
    
    # First, crop the original RGB image to the target size
    base_crop = transforms.CenterCrop((256, 256))(rgb_img)
    
    lsb_r = (np.array(base_crop)[:, :, 0] & 1).astype(np.uint8)
    lsb_g = (np.array(base_crop)[:, :, 1] & 1).astype(np.uint8)
    lsb_b = (np.array(base_crop)[:, :, 2] & 1).astype(np.uint8)
    lsb_bytes = np.packbits(np.stack([lsb_r, lsb_g, lsb_b], axis=0).flatten())
    lsb_content_features = _calculate_content_features(lsb_bytes)
    
    lsb = torch.from_numpy(np.stack([lsb_r.astype(np.float32), lsb_g.astype(np.float32), lsb_b.astype(np.float32)], axis=0))

    # --- Pixel Tensor (Post-Augmentation) ---
    # Now, apply augmentations for training robustness
    # For now, we use a standard crop, but this is where other transforms would go.
    aug_img = base_crop
    pixel_tensor = transforms.ToTensor()(aug_img)


    # --- Metadata Path ---
    with open(path, 'rb') as f:
        raw = f.read()
    
    exif = img.info.get("exif", b"") # Use Pillow's built-in EXIF extraction
    
    format_hint = img.format.lower() if img.format else 'auto'
    tail = extract_post_tail(raw, format_hint)
    meta_bytes = np.frombuffer(exif + tail, dtype=np.uint8)[:2048]
    meta_bytes = np.pad(meta_bytes, (0, 2048 - len(meta_bytes)), 'constant')
    meta = torch.from_numpy(meta_bytes.astype(np.float32)/255.0)

    # --- Alpha Path ---
    # Ensure alpha tensor is 1 channel, 256x256
    has_alpha = 1.0 if img.mode == 'RGBA' else 0.0
    alpha_std_dev = 0.0
    if img.mode == 'RGBA':
        alpha_plane = np.array(img.split()[-1])
        alpha_std_dev = alpha_plane.std() / 255.0 # Normalize
        alpha_lsb = (alpha_plane & 1).astype(np.uint8)
        alpha_bytes = np.packbits(alpha_lsb.flatten())
        alpha_content_features = _calculate_content_features(alpha_bytes)
        
        alpha = torch.from_numpy((alpha_plane & 1).astype(np.float32)).unsqueeze(0)
        alpha = transforms.CenterCrop((256, 256))(alpha)
    else:
        alpha = torch.zeros(1, 256, 256)
        alpha_content_features = torch.zeros(3, dtype=torch.float32)


    # --- Palette Path ---
    # Ensure palette tensor is 768 elements
    if img.mode == 'P':
        palette_bytes = np.array(img.getpalette(), dtype=np.uint8)
        palette_bytes = np.pad(palette_bytes, (0, 768 - len(palette_bytes)), 'constant')
    else:
        palette_bytes = np.zeros(768, dtype=np.uint8)
    palette = torch.from_numpy(palette_bytes.astype(np.float32) / 255.0)

    # --- Format Features Path ---
    # Simplified format features to match the model's expectations.
    width, height = img.size
    format_features = torch.tensor([
        has_alpha,
        alpha_std_dev,
        1.0 if img.mode == 'P' else 0.0,    # is_palette
        1.0 if img.mode == 'RGB' else 0.0,   # is_rgb
        float(width) / 256.0,          # width_norm
        float(height) / 256.0            # height_norm
    ], dtype=torch.float32)
    
    content_features = torch.cat([lsb_content_features, alpha_content_features])

    return pixel_tensor, meta, alpha, lsb, palette, format_features, content_features
