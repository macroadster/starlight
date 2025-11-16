import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import struct
import os
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import copy
import argparse
import random
import glob
import piexif
from scripts.starlight_utils import extract_post_tail
from scripts.starlight_utils import load_unified_input

class BalancedStegoDataset(Dataset):
    """
    Dataset that uses stego JSON files to find matching clean images.
    Creates perfect 1:1 clean:stego ratio by sampling stego images to match clean count.
    """

    def __init__(self, clean_dir_pattern, stego_dir_pattern, transform=None, balance_strategy='sample_stego'):
        self.transform = transform
        self.samples = []
        self.clean_files_used = set()
        self.balance_strategy = balance_strategy
        self.method_labels_list = []
        self.method_map = {"alpha": 0, "palette": 1, "lsb.rgb": 2, "exif": 3, "raw": 4}

        print(f"[BALANCED DATASET] Loading from clean pattern: {clean_dir_pattern}, stego pattern: {stego_dir_pattern}...")
        print(f"[BALANCED DATASET] Balance strategy: {balance_strategy}")

        stego_dirs = sorted(glob.glob(stego_dir_pattern))
        if not stego_dirs:
            print(f"Warning: No stego directories found for pattern: {stego_dir_pattern}")

        all_stego_samples = []
        clean_samples = []

        for stego_dir_str in stego_dirs:
            stego_dir = Path(stego_dir_str)
            # Derive clean directory from stego directory path
            clean_dir_str = stego_dir_str.replace('/stego', '/clean').replace('\\stego', '\\clean')
            clean_dir = Path(clean_dir_str)

            if not clean_dir.exists():
                print(f"Warning: Corresponding clean directory not found for {stego_dir_str}, expected at {clean_dir_str}")
                continue
            
            print(f"  - Processing stego dir: {stego_dir_str}")
            print(f"  - Using clean dir: {clean_dir_str}")

            for json_file in stego_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)

                    technique = metadata.get('embedding', {}).get('technique')
                    if technique not in self.method_map:
                        continue

                    clean_filename = metadata.get('clean_file')
                    if not clean_filename:
                        continue

                    clean_path = clean_dir / clean_filename
                    stego_path = json_file.with_suffix('')

                    if not clean_path.exists() or not stego_path.exists():
                        continue

                    method_id = self.method_map[technique]
                    bit_order = metadata.get('embedding', {}).get('bit_order', 'msb-first')
                    all_stego_samples.append({
                        'path': stego_path,
                        'stego_label': 1,
                        'method_label': method_id,
                        'bit_order': bit_order,
                        'type': 'stego'
                    })

                    if clean_filename not in self.clean_files_used:
                        clean_samples.append({
                            'path': clean_path,
                            'stego_label': 0,
                            'method_label': -1,
                            'bit_order': 'none',
                            'type': 'clean'
                        })
                        self.clean_files_used.add(clean_filename)

                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Balance the dataset
        if balance_strategy == 'sample_stego' or balance_strategy == 'undersample_stego':
            # Sample stego images to match clean count (undersample stego)
            clean_count = len(clean_samples)
            if len(all_stego_samples) > clean_count:
                # Randomly sample stego images to match clean count
                random.shuffle(all_stego_samples)
                sampled_stego = all_stego_samples[:clean_count]
                self.samples = clean_samples + sampled_stego
            else:
                # Use all stego and sample clean to match
                random.shuffle(clean_samples)
                sampled_clean = clean_samples[:len(all_stego_samples)]
                self.samples = sampled_clean + all_stego_samples

        elif balance_strategy == 'oversample_clean':
            # Oversample clean images to match stego count
            stego_count = len(all_stego_samples)
            if len(clean_samples) > 0:
                clean_multiplier = stego_count // len(clean_samples) + 1
                oversampled_clean = clean_samples * clean_multiplier
                # Trim to exact match
                oversampled_clean = oversampled_clean[:stego_count]
                self.samples = oversampled_clean + all_stego_samples
            else: # Handle case with no clean samples
                self.samples = all_stego_samples
        
        elif balance_strategy == 'use_all_stego':
            # Use all stego images and oversample clean to match
            stego_count = len(all_stego_samples)
            if len(clean_samples) > 0:
                clean_multiplier = stego_count // len(clean_samples) + 1
                oversampled_clean = clean_samples * clean_multiplier
                # Trim to exact match
                oversampled_clean = oversampled_clean[:stego_count]
                self.samples = oversampled_clean + all_stego_samples
            else: # Handle case with no clean samples
                self.samples = all_stego_samples
        
        elif balance_strategy == 'balanced_classes':
            # Ensure each stego class has equal representation
            # Group stego samples by method
            stego_by_method = {}
            for sample in all_stego_samples:
                method = sample['method_label']
                if method not in stego_by_method:
                    stego_by_method[method] = []
                stego_by_method[method].append(sample)
            
            # Find the smallest class size (excluding clean)
            min_class_size = min(len(samples) for samples in stego_by_method.values()) if stego_by_method else 0
            
            # Sample equally from each stego class
            balanced_stego = []
            for method, samples in stego_by_method.items():
                if len(samples) >= min_class_size:
                    random.shuffle(samples)
                    balanced_stego.extend(samples[:min_class_size])
            
            # Sample clean to match total stego
            total_stego = len(balanced_stego)
            if len(clean_samples) >= total_stego:
                random.shuffle(clean_samples)
                balanced_clean = clean_samples[:total_stego]
            else:
                # Oversample clean if needed
                clean_multiplier = total_stego // len(clean_samples) + 1
                balanced_clean = clean_samples * clean_multiplier
                balanced_clean = balanced_clean[:total_stego]
            
            self.samples = balanced_clean + balanced_stego


        # Collect method labels for class weights
        for sample in self.samples:
            if sample['type'] == 'stego':
                self.method_labels_list.append(sample['method_label'])

        print(f"[BALANCED DATASET] Loaded {len(self.samples)} total samples")
        print(f"[BALANCED DATASET] Available: {len(clean_samples)} clean, {len(all_stego_samples)} stego")

        # Print class distribution
        class_counts = {}
        type_counts = {'clean': 0, 'stego': 0}
        for sample in self.samples:
            if sample['type'] == 'stego':
                technique = list(self.method_map.keys())[list(self.method_map.values()).index(sample['method_label'])]
                class_counts[technique] = class_counts.get(technique, 0) + 1
            type_counts[sample['type']] += 1

        print(f"[BALANCED DATASET] Final class distribution: {class_counts}")
        print(f"[BALANCED DATASET] Final type distribution: {type_counts}")

        # Verify balance
        clean_count = type_counts['clean']
        stego_count = type_counts['stego']
        if clean_count == stego_count:
            print(f"[BALANCED DATASET] ✅ Perfect balance: {clean_count} clean, {stego_count} stego")
        else:
            print(f"[BALANCED DATASET] ⚠️  Imbalance: {clean_count} clean, {stego_count} stego")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']

        try:
            pixel_tensor, meta, alpha, lsb, palette, format_features, content_features = load_unified_input(str(img_path))
            # Add bit_order as one-hot encoding: [lsb-first, msb-first, none]
            bit_order = sample.get('bit_order', 'msb-first')
            if bit_order == 'lsb-first':
                bit_order_feat = torch.tensor([1.0, 0.0, 0.0])
            elif bit_order == 'msb-first':
                bit_order_feat = torch.tensor([0.0, 1.0, 0.0])
            else:  # clean images
                bit_order_feat = torch.tensor([0.0, 0.0, 1.0])
            
            return pixel_tensor, meta, alpha, lsb, palette, format_features, content_features, bit_order_feat, sample['stego_label'], sample['method_label']
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data
            dummy_pixel = torch.zeros(3, 256, 256)
            dummy_meta = torch.zeros(2048)
            dummy_alpha = torch.zeros(1, 256, 256)
            dummy_lsb = torch.zeros(3, 256, 256)
            dummy_palette = torch.zeros(768)
            dummy_format = torch.zeros(6)
            dummy_content = torch.zeros(6)
            dummy_bit_order = torch.tensor([0.0, 0.0, 1.0])  # clean default
            return dummy_pixel, dummy_meta, dummy_alpha, dummy_lsb, dummy_palette, dummy_format, dummy_content, dummy_bit_order, 0, -1

def extract_enhanced_metadata_features(image_path):
    """Extract enhanced metadata features for better EXIF detection across various formats."""

    with open(image_path, 'rb') as f:
        raw = f.read()

    img = Image.open(image_path)
    format_hint = img.format.lower() if img.format else 'unknown'

    # --- EXIF Extraction ---
    exif_bytes = b""
    # Prioritize EXIF data stored in img.info (e.g., by data_generator for PNG/WebP)
    if "exif" in img.info:
        exif_bytes = img.info["exif"]
    elif piexif is not None:
        try:
            # Try to load EXIF using piexif from raw bytes
            exif_dict = piexif.load(raw)
            if exif_dict:
                # Re-dump to bytes to get a consistent format for feature extraction
                exif_bytes = piexif.dump(exif_dict)
        except Exception:
            pass # Not all image types have EXIF or piexif might fail

    # --- EOI (Tail) Extraction ---
    tail = extract_post_tail(raw, format_hint)

    # Debug prints


    # Enhanced features (next 1024 bytes)
    enhanced_features = np.zeros(1024, dtype=np.float32)

    # EXIF size feature
    if exif_bytes:
        enhanced_features[0] = len(exif_bytes) / 65535.0 # Normalize by max EXIF size

    # JPEG marker analysis (next 100 bytes) - only for JPEG images
    if format_hint == 'jpeg':
        jpeg_markers = []
        for i in range(0, min(len(raw) - 1, 10000), 2):  # Check first 10KB
            if raw[i] == 0xFF and i + 1 < len(raw):
                marker = raw[i+1]
                if marker != 0x00:
                    jpeg_markers.append(marker)

        # Store marker histogram (50 bytes)
        marker_hist = np.zeros(50, dtype=np.float32)
        for marker in jpeg_markers[:100]:  # First 100 markers
            idx = marker % 50
            if idx < 50:
                marker_hist[idx] = marker_hist[idx] + 1

        enhanced_features[60:110] = marker_hist

    # Tail analysis (next 100 bytes)
    if len(tail) > 0:
        hist = np.histogram(bytearray(tail[:1000]), bins=256)[0]
        # Normalize histogram to get probabilities for entropy calculation
        hist_probs = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        tail_entropy = -np.sum(hist_probs * np.log2(hist_probs + 1e-10))
        enhanced_features[110] = tail_entropy
        enhanced_features[111] = len(tail)

        # Store first 88 bytes of tail data
        tail_bytes = np.frombuffer(tail[:88], dtype=np.uint8).astype(np.float32)
        enhanced_features[112:112+len(tail_bytes)] = tail_bytes

    # EXIF content analysis (next 200 bytes)
    exif_content_features = np.zeros(200, dtype=np.float32)
    if exif_bytes:
        # EXIF header check (first 6 bytes should be "Exif\x00\x00")
        if len(exif_bytes) >= 6:
            exif_header = exif_bytes[:6]
            exif_content_features[0:6] = np.frombuffer(exif_header, dtype=np.uint8).astype(np.float32)

        # EXIF data entropy
        if len(exif_bytes) > 6:
            exif_data = exif_bytes[6:]
            hist = np.histogram(bytearray(exif_data), bins=256)[0]
            exif_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            exif_content_features[6] = exif_entropy

        # EXIF length ratio (relative to total file)
        exif_ratio = len(exif_bytes) / len(raw) if len(raw) > 0 else 0
        exif_content_features[7] = exif_ratio * 255

        # Check for valid TIFF header in EXIF (big-endian or little-endian)
        if len(exif_bytes) >= 12:
            tiff_header = exif_bytes[6:12]
            # "II" (little-endian) or "MM" (big-endian)
            if tiff_header[:2] in [b'II', b'MM']:
                exif_content_features[12] = 1  # Valid TIFF header
            else:
                exif_content_features[12] = 0  # Invalid


    enhanced_features[200:400] = exif_content_features

    # Location-based features (next 50 bytes)
    location_features = np.zeros(50, dtype=np.float32)
    if exif_bytes:
        # EXIF presence
        location_features[0] = 1.0

    # Tail position relative to file size
    if len(tail) > 0:
        tail_start_pos = len(raw) - len(tail)
        rel_tail_pos = tail_start_pos / len(raw)
        location_features[1] = rel_tail_pos * 255 # Using index 1 now
        location_features[2] = 1 if rel_tail_pos > 0.75 else 0 # Using index 2 now

    # JPEG structure analysis (only for JPEG images)
    if format_hint == 'jpeg':
        soi_pos = raw.find(b'\xFF\xD8')
        eoi_pos = raw.rfind(b'\xFF\xD9')
        if soi_pos != -1 and eoi_pos != -1:
            # Distance between SOI and EOI
            image_size = eoi_pos - soi_pos + 2
            rel_image_size = image_size / len(raw)
            location_features[3] = rel_image_size * 255 # Using index 3 now

            # Data after EOI (tail) ratio
            tail_ratio = len(tail) / len(raw)
            location_features[4] = tail_ratio * 255 # Using index 4 now

    enhanced_features[400:450] = location_features

    # Combine basic and enhanced features
    basic_bytes = np.frombuffer(exif_bytes + tail, dtype=np.uint8)[:1024]
    basic_bytes = np.pad(basic_bytes, (0, 1024 - len(basic_bytes)), 'constant')

    # Clip enhanced features to 0-255 and normalize
    enhanced_features = np.clip(enhanced_features, 0, 255) / 255.0

    # Return both basic and enhanced features
    return basic_bytes.astype(np.float32) / 255.0, enhanced_features

def load_enhanced_multi_input(path, transform=None):
    """Enhanced version of load_multi_input with better metadata features"""
    img = Image.open(path)

    # Enhanced metadata features
    basic_meta, enhanced_meta = extract_enhanced_metadata_features(path)

    # Combine metadata features (basic + enhanced)
    meta = torch.cat([torch.from_numpy(basic_meta), torch.from_numpy(enhanced_meta)])

    # Alpha path - handle BEFORE any transformations to preserve steganography
    if img.mode == 'RGBA':
        alpha_pil = img.split()[-1]  # Extract alpha channel directly
        # For alpha steganography preservation, use Resize instead of CenterCrop when needed
        if transform:
            # Check if transform contains CenterCrop and image is not square
            if hasattr(transform, 'transforms'):
                has_center_crop = any(isinstance(t, transforms.CenterCrop) for t in transform.transforms)
            else:
                has_center_crop = isinstance(transform, transforms.CenterCrop)
            
            if has_center_crop and (img.width != img.height):
                # Use Resize to preserve alpha steganography for non-square images
                alpha_aug = transforms.Resize((256, 256))(alpha_pil)
            else:
                # Apply same transform to alpha
                alpha_aug = transform(alpha_pil)
        else:
            # Default handling - use Resize instead of CenterCrop for non-square images
            if img.width != img.height:
                alpha_aug = transforms.Resize((256, 256))(alpha_pil)
            else:
                crop = transforms.CenterCrop((256, 256))
                alpha_aug = crop(alpha_pil)
        
        alpha = torch.from_numpy(np.array(alpha_aug).astype(np.float32) / 255.0).unsqueeze(0)
    else:
        alpha = torch.zeros(1, 256, 256) # Ensure it's 256x256

    # Extract LSB BEFORE augmentation to preserve steganography signals
    rgb_img_original = img.convert('RGB')
    lsb_r = (np.array(rgb_img_original)[:, :, 0] & 1).astype(np.float32)
    lsb_g = (np.array(rgb_img_original)[:, :, 1] & 1).astype(np.float32)
    lsb_b = (np.array(rgb_img_original)[:, :, 2] & 1).astype(np.float32)
    lsb_original = torch.from_numpy(np.stack([lsb_r, lsb_g, lsb_b], axis=0))
    
    # Resize LSB to match expected dimensions (256x256) using nearest neighbor to preserve binary values
    from torchvision.transforms import functional as F_transforms
    lsb = F_transforms.resize(lsb_original, [256, 256], interpolation=F_transforms.InterpolationMode.NEAREST)

    # Augmentation for RGB channels (applied to RGB image for metadata/other features)
    if transform:
        aug_img = transform(rgb_img_original)
    else:
        # Default to CenterCrop if no transform is provided
        crop = transforms.CenterCrop((256, 256))
        aug_img = crop(rgb_img_original)

    # Palette path - extract both palette colors AND pixel index LSB patterns for palette steganography
    if img.mode == 'P':
        # Extract palette colors (traditional approach)
        palette_colors = np.array(img.getpalette(), dtype=np.uint8)
        palette_colors = np.pad(palette_colors, (0, 768 - len(palette_colors)), 'constant')
        
        # Extract LSB patterns from pixel indices (actual palette steganography)
        pixel_indices = np.array(img.getdata())
        
        # Create LSB features from pixel indices
        lsb_features = np.zeros(256, dtype=np.uint8)  # Reduced size for efficiency
        
        total_pixels = len(pixel_indices)
        
        if total_pixels > 0:
            # Extract LSB patterns from pixel indices
            pixel_lsb = pixel_indices & 1
            
            # Sample LSB values from different regions (store as raw bytes, not packed)
            region_size = min(64, total_pixels // 4)  # Sample up to 64 pixels per region
            
            # Top-left region - store raw LSB values
            if region_size > 0:
                lsb_features[0:region_size] = pixel_lsb[:region_size].astype(np.uint8) * 255
            
            # Center region  
            center_start = total_pixels // 2 - region_size // 2
            if center_start + region_size <= total_pixels:
                lsb_features[64:64+region_size] = pixel_lsb[center_start:center_start + region_size].astype(np.uint8) * 255
            
            # Bottom-right region
            if region_size > 0:
                start_idx = 128
                lsb_features[start_idx:start_idx+region_size] = pixel_lsb[-region_size:].astype(np.uint8) * 255
            
            # Overall LSB statistics
            lsb_features[192] = int(np.mean(pixel_lsb) * 255)  # Average LSB value
            lsb_features[193] = int(np.std(pixel_lsb) * 255)   # LSB variation
            lsb_features[194] = int((pixel_lsb.sum() % 256))    # LSB sum modulo
            
            # Additional features: transitions and patterns
            if len(pixel_lsb) > 1:
                transitions = np.sum(pixel_lsb[1:] != pixel_lsb[:-1])  # Count LSB transitions
                lsb_features[195] = min(255, transitions)  # Cap at 255
                
                # Count consecutive same bits
                max_consecutive = 0
                current_consecutive = 1
                for i in range(1, len(pixel_lsb)):
                    if pixel_lsb[i] == pixel_lsb[i-1]:
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 1
                lsb_features[196] = min(255, max_consecutive)
            
        # Combine palette colors and LSB features
        palette_bytes = np.concatenate([palette_colors[:512], lsb_features])  # 512 + 256 = 768
        
    else:
        # For non-palette images, create minimal palette data
        palette_bytes = np.zeros(768, dtype=np.uint8)
        
        # Only extract palette if there's evidence of palette-based steganography
        if hasattr(img, 'info') and 'palette' in img.info:
            try:
                embedded_palette = img.info['palette']
                if isinstance(embedded_palette, (bytes, bytearray)) and len(embedded_palette) > 0:
                    palette_data = np.frombuffer(embedded_palette[:768], dtype=np.uint8)
                    palette_bytes[:len(palette_data)] = palette_data
            except (ValueError, TypeError):
                pass  # Keep zero palette if extraction fails
    
    palette = torch.from_numpy(palette_bytes.astype(np.float32) / 255.0)

    return meta, alpha, lsb, palette

class BalancedStarlightDetector(nn.Module):
    """Balanced model with weighted metadata processing to reduce EXIF dominance"""

    def __init__(self, meta_weight=0.1):
        super(BalancedStarlightDetector, self).__init__()

        self.meta_weight = meta_weight  # Weight to reduce metadata dominance

        # Metadata stream (now 2048 features instead of 1024)
        self.meta_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )

        # Alpha stream
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )

        # LSB stream
        self.lsb_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )

        # Palette stream
        self.palette_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # Format features stream
        self.format_features_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Content features stream
        self.content_features_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Fusion and classification
        self.fusion_dim = 128 * 16 + 64 * 8 * 8 + 64 * 8 * 8 + 64 + 16 + 16 + 3  # +3 for bit_order features
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128 + 64)  # 128 for embedding, 64 for classification
        )

        # Heads
        self.stego_head = nn.Linear(64, 1)
        self.method_head = nn.Linear(64, 5)  # alpha, palette, lsb.rgb, exif, raw
        self.embedding_head = nn.Linear(64, 64)

    def forward(self, meta, alpha, lsb, palette, format_features, content_features, bit_order):
        # Metadata stream with weighting
        meta = meta.unsqueeze(1)  # Add channel dimension
        meta = self.meta_conv(meta)
        meta = meta.view(meta.size(0), -1)
        meta = meta * self.meta_weight  # Apply weighting to reduce dominance

        # Alpha stream
        alpha = self.alpha_conv(alpha)
        alpha = alpha.view(alpha.size(0), -1)

        # LSB stream
        lsb = self.lsb_conv(lsb)
        lsb = lsb.view(lsb.size(0), -1)

        # Palette stream
        palette = self.palette_fc(palette)

        # Format features stream
        format_features = self.format_features_fc(format_features)

        # Content features stream
        content_features = self.content_features_fc(content_features)

        # Fusion with bit_order features
        fused = torch.cat([meta, alpha, lsb, palette, format_features, content_features, bit_order], dim=1)
        fused = self.fusion(fused)

        # Split into embedding and classification features
        embedding = fused[:, :128]
        cls_features = fused[:, 128:]

        # Outputs
        stego_logits = self.stego_head(cls_features)
        method_logits = self.method_head(cls_features)
        embedding = self.embedding_head(cls_features)

        # Get method predictions
        method_probs = F.softmax(method_logits, dim=1)
        method_id = torch.argmax(method_probs, dim=1)

        return stego_logits, method_logits, method_id, method_probs, embedding

def compute_class_weights(method_labels):
    """Compute class weights to balance training"""
    # Count samples per class
    class_counts = Counter(method_labels)
    total_samples = len(method_labels)

    # Compute inverse frequency weights
    class_weights = {}
    for class_id, count in class_counts.items():
        weight = total_samples / (len(class_counts) * count)
        class_weights[class_id] = weight

    # Convert to tensor
    weights = torch.zeros(5)  # 5 classes
    for class_id, weight in class_weights.items():
        weights[class_id] = weight

    return weights

def train_model(train_clean_dir, train_stego_dir, val_clean_dir=None, val_stego_dir=None, epochs=10, batch_size=8, lr=1e-4, out_path="models/detector_generalized.onnx"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    # --- Data Loading and Splitting ---
    # Use separate datasets: training from submissions, validation from val set
    print("Loading training data from submissions and validation data from val set...")

    # Load training samples from all submission datasets with balanced classes
    train_source_dataset = BalancedStegoDataset(train_clean_dir, train_stego_dir, transform=None, balance_strategy='balanced_classes')
    
    # Load validation samples from val dataset only - use all stego images
    val_source_dataset = BalancedStegoDataset(val_clean_dir, val_stego_dir, transform=None, balance_strategy='use_all_stego')

    # Create training dataset with augmentations
    train_dataset = copy.deepcopy(train_source_dataset)
    train_dataset.transform = train_transform

    # Create validation dataset without augmentations
    val_dataset = copy.deepcopy(val_source_dataset)
    val_dataset.transform = val_transform
    
    method_labels_list = train_dataset.method_labels_list

    print(f"\nDataset split into {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    print(f"Training method distribution: {Counter(method_labels_list)}")
    print(f"Validation method distribution: {Counter(val_dataset.method_labels_list)}")

    # --- Dataloaders and Class Weights ---
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights
    class_weights = compute_class_weights(method_labels_list)
    print(f"Class weights: {class_weights}")

    # Model
    model = BalancedStarlightDetector(meta_weight=0.3).to(device)  # Reduce metadata dominance
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss functions with class weighting
    stego_criterion = nn.BCEWithLogitsLoss()
    method_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        stego_correct = 0
        method_correct = 0
        stego_total = 0
        method_total = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            pixel_tensor, meta, alpha, lsb, palette, format_features, content_features, bit_order_feat, stego_labels, method_labels = batch
            pixel_tensor = pixel_tensor.to(device)
            meta = meta.to(device)
            alpha = alpha.to(device)
            lsb = lsb.to(device)
            palette = palette.to(device)
            format_features = format_features.to(device)
            content_features = content_features.to(device)
            bit_order_feat = bit_order_feat.to(device)
            stego_labels = stego_labels.float().to(device)
            method_labels = method_labels.long().to(device)

            optimizer.zero_grad()

            # Forward pass
            stego_logits, method_logits, method_ids, method_probs, embeddings = model(meta, alpha, lsb, palette, format_features, content_features, bit_order_feat)

            # Compute losses
            stego_loss = stego_criterion(stego_logits.view(-1), stego_labels)

            # Only compute method loss for stego samples
            stego_mask = stego_labels > 0.5
            if stego_mask.sum() > 0:
                method_loss = method_criterion(method_logits[stego_mask], method_labels[stego_mask])
            else:
                method_loss = torch.tensor(0.0).to(device)

            # Dynamic loss weighting - reduce method loss importance to avoid overfitting
            total_loss_batch = stego_loss + 0.01 * method_loss #+ 0.05 * method_loss  # Further reduced

            # Backward pass
            total_loss_batch.backward()
            optimizer.step()

            # Statistics
            total_loss += total_loss_batch.item()
            stego_pred = (torch.sigmoid(stego_logits) > 0.5).float()
            stego_correct += (stego_pred.squeeze() == stego_labels).sum().item()
            stego_total += stego_labels.size(0)

            if stego_mask.sum() > 0:
                method_correct += (method_ids[stego_mask] == method_labels[stego_mask]).sum().item()
                method_total += stego_mask.sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_stego_correct = 0
        val_method_correct = 0
        val_stego_total = 0
        val_method_total = 0
        val_method_correct_per_class = Counter()
        val_method_total_per_class = Counter()
        id_to_method = {v: k for k, v in val_dataset.method_map.items()}

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                pixel_tensor, meta, alpha, lsb, palette, format_features, content_features, bit_order_feat, stego_labels, method_labels = batch
                pixel_tensor = pixel_tensor.to(device)
                meta = meta.to(device)
                alpha = alpha.to(device)
                lsb = lsb.to(device)
                palette = palette.to(device)
                format_features = format_features.to(device)
                content_features = content_features.to(device)
                bit_order_feat = bit_order_feat.to(device)
                stego_labels = stego_labels.float().to(device)
                method_labels = method_labels.long().to(device)

                stego_logits, method_logits, method_ids, method_probs, embeddings = model(meta, alpha, lsb, palette, format_features, content_features, bit_order_feat)
                stego_loss = stego_criterion(stego_logits.view(-1), stego_labels)
                stego_mask = stego_labels > 0.5
                if stego_mask.sum() > 0:
                    method_loss = method_criterion(method_logits[stego_mask], method_labels[stego_mask])
                else:
                    method_loss = torch.tensor(0.0).to(device)

                val_loss += (stego_loss + 0.01 * method_loss).item()
                stego_pred = (torch.sigmoid(stego_logits) > 0.5).float()
                val_stego_correct += (stego_pred.squeeze() == stego_labels).sum().item()
                val_stego_total += stego_labels.size(0)

                if stego_mask.sum() > 0:
                    val_method_correct += (method_ids[stego_mask] == method_labels[stego_mask]).sum().item()
                    val_method_total += stego_mask.sum().item()

                    preds = method_ids[stego_mask]
                    labels = method_labels[stego_mask]
                    for i in range(len(labels)):
                        label = labels[i].item()
                        pred = preds[i].item()
                        val_method_total_per_class[label] += 1
                        if pred == label:
                            val_method_correct_per_class[label] += 1

        # Print statistics
        avg_loss = total_loss / len(train_dataloader)
        stego_acc = stego_correct / stego_total if stego_total > 0 else 0
        method_acc = method_correct / method_total if method_total > 0 else 0

        avg_val_loss = val_loss / len(val_dataloader)
        val_stego_acc = val_stego_correct / val_stego_total if val_stego_total > 0 else 0
        val_method_acc = val_method_correct / val_method_total if val_method_total > 0 else 0

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Stego Acc={stego_acc:.3f}, Method Acc={method_acc:.3f}")
        print(f"Val Loss={avg_val_loss:.4f}, Val Stego Acc={val_stego_acc:.3f}, Val Method Acc={val_method_acc:.3f}")

        # Print per-class accuracy
        print("  Validation Method Accuracy per class:")
        for method_id, total in sorted(val_method_total_per_class.items()):
            correct = val_method_correct_per_class.get(method_id, 0)
            acc = correct / total if total > 0 else 0
            method_name = id_to_method.get(method_id, "Unknown")
            print(f"    - {method_name}: {acc:.3f} ({correct}/{total})")

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_path.replace('.onnx', '.pth'))
            print(f"Validation loss improved. Saved best model to {out_path.replace('.onnx', '.pth')}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement in validation loss for {patience} epochs. Stopping early.")
                break



    # Export to ONNX
    print("Exporting to ONNX...")
    model.load_state_dict(torch.load(out_path.replace('.onnx', '.pth')))
    model.eval()

    # Create dummy input
    dummy_meta = torch.randn(1, 2048)
    dummy_alpha = torch.randn(1, 1, 256, 256)
    dummy_lsb = torch.randn(1, 3, 256, 256)
    dummy_palette = torch.randn(1, 768)
    dummy_format_features = torch.randn(1, 6)
    dummy_content_features = torch.randn(1, 6)
    dummy_bit_order = torch.tensor([[0.0, 1.0, 0.0]])  # msb-first default

    dummy_meta = dummy_meta.to(device)
    dummy_alpha = dummy_alpha.to(device)
    dummy_lsb = dummy_lsb.to(device)
    dummy_palette = dummy_palette.to(device)
    dummy_format_features = dummy_format_features.to(device)
    dummy_content_features = dummy_content_features.to(device)
    dummy_bit_order = dummy_bit_order.to(device)

    torch.onnx.export(
        model,
        (dummy_meta, dummy_alpha, dummy_lsb, dummy_palette, dummy_format_features, dummy_content_features, dummy_bit_order),
        out_path,
        input_names=['meta', 'alpha', 'lsb', 'palette', 'format_features', 'content_features', 'bit_order'],
        output_names=['stego_logits', 'method_logits', 'method_id', 'method_probs', 'embedding'],
        dynamic_axes={
            'meta': {0: 'batch'},
            'alpha': {0: 'batch'},
            'lsb': {0: 'batch'},
            'palette': {0: 'batch'},
            'format_features': {0: 'batch'},
            'content_features': {0: 'batch'},
            'bit_order': {0: 'batch'},
            'stego_logits': {0: 'batch'},
            'method_logits': {0: 'batch'},
            'method_id': {0: 'batch'},
            'method_probs': {0: 'batch'},
            'embedding': {0: 'batch'}
        },
        opset_version=11
    )

    print(f"Generalized model exported to {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_clean_dir", default="datasets/*_submission_*/clean")
    parser.add_argument("--train_stego_dir", default="datasets/*_submission_*/stego")
    parser.add_argument("--val_clean_dir", default="datasets/val/clean")
    parser.add_argument("--val_stego_dir", default="datasets/val/stego")
    parser.add_argument("--epochs", type=int, default=50) # Increased default epochs
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="models/detector_balanced.onnx")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    train_model(args.train_clean_dir, args.train_stego_dir, args.val_clean_dir, args.val_stego_dir, args.epochs, args.batch_size, args.lr, args.out)


