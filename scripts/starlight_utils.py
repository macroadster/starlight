import torch
import numpy as np
from PIL import Image
import struct
import torchvision.transforms as transforms

def extract_post_tail(raw_bytes, format_hint='auto'):
    tails = {
        'jpeg': raw_bytes[raw_bytes.rfind(b'\xFF\xD9') + 2:] if raw_bytes.rfind(b'\xFF\xD9') != -1 else b"",
        'png': raw_bytes[raw_bytes.rfind(b'IEND') + 4:] if raw_bytes.rfind(b'IEND') != -1 else b"",
        'gif': raw_bytes[raw_bytes.rfind(b';') + 1:] if raw_bytes.rfind(b';') != -1 else b"",
        'webp': raw_bytes[raw_bytes.rfind(b'VP8X') + 10:] if raw_bytes.rfind(b'VP8X') != -1 else b"",
    }
    return tails.get(format_hint, b"")

def load_multi_input(path, transform=None):
    img = Image.open(path)
    
    # --- Augmentation ---
    # Augmentations should be done on the RGB version of the image
    rgb_img = img.convert('RGB')
    if transform:
        aug_img = transform(rgb_img)
    else:
        crop = transforms.CenterCrop((256, 256))
        aug_img = crop(rgb_img)

    # --- Metadata Path ---
    with open(path, 'rb') as f:
        raw = f.read()
    exif = b""
    pos = raw.find(b'\xFF\xE1')
    if pos != -1:
        length = struct.unpack('>H', raw[pos+2:pos+4])[0]
        exif = raw[pos+4:pos+4+length-2]
    format_hint = img.format.lower() if img.format else 'auto'
    tail = extract_post_tail(raw, format_hint)
    meta_bytes = np.frombuffer(exif + tail, dtype=np.uint8)[:1024]
    meta_bytes = np.pad(meta_bytes, (0, 1024 - len(meta_bytes)), 'constant')
    meta = torch.from_numpy(meta_bytes.astype(np.float32)/255.0)

    # --- Alpha Path ---
    # Ensure alpha tensor is 1 channel, 256x256
    if img.mode == 'RGBA':
        alpha_plane = np.array(img.split()[-1]).astype(np.float32) / 255.0
        alpha = torch.from_numpy(alpha_plane).unsqueeze(0) # Add channel dim
    else:
        alpha = torch.zeros(1, img.height, img.width) # Use original image dimensions
    alpha = transforms.CenterCrop((256, 256))(alpha) # Crop to 256x256

    # --- LSB Path ---
    # Ensure LSB tensor is 3 channels, 256x256
    lsb_r = (np.array(aug_img)[:, :, 0] & 1).astype(np.float32)
    lsb_g = (np.array(aug_img)[:, :, 1] & 1).astype(np.float32)
    lsb_b = (np.array(aug_img)[:, :, 2] & 1).astype(np.float32)
    lsb = torch.from_numpy(np.stack([lsb_r, lsb_g, lsb_b], axis=0))

    # --- Palette Path ---
    # Ensure palette tensor is 768 elements
    if img.mode == 'P':
        palette_bytes = np.array(img.getpalette(), dtype=np.uint8)
        palette_bytes = np.pad(palette_bytes, (0, 768 - len(palette_bytes)), 'constant')
    else:
        palette_bytes = np.zeros(768, dtype=np.uint8)
    palette = torch.from_numpy(palette_bytes.astype(np.float32) / 255.0)

    return meta, alpha, lsb, palette
