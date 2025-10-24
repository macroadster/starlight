#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import os
import math
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.stats import entropy

# Define transforms with RGBA support
transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
])

def extract_features(path, img):
    """
    Enhanced feature extraction with better palette and LSB detection.
    Total features: 24 (upgraded from 15)
    
    This function is used by both trainer.py and starlight_extractor.py
    """
    width, height = img.size
    area = width * height if width * height > 0 else 1.0
    
    original_mode = img.mode
    original_format = img.format
    
    # Basic features (5)
    file_size = os.path.getsize(path) / 1024.0
    file_size_norm = file_size / area
    
    exif_bytes = img.info.get('exif')
    exif_present = 1.0 if exif_bytes else 0.0
    exif_length = len(exif_bytes) if exif_bytes else 0.0
    exif_length_norm = min(exif_length / area, 1.0)
    
    comment_length = 0.0
    exif_entropy_val = 0.0
    if exif_bytes:
        try:
            exif_dict = img.getexif()
            tag_values = []
            for tag_id, value in exif_dict.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'UserComment' and isinstance(value, bytes):
                    comment_length = min(len(value) / area, 1.0)
                if isinstance(value, (bytes, str)):
                    tag_values.append(value if isinstance(value, bytes) else value.encode('utf-8'))
            if tag_values:
                lengths = [len(v) for v in tag_values]
                max_len = max(lengths or [1])
                hist = np.histogram(lengths, bins=10, range=(0, max_len))[0]
                exif_entropy_val = entropy(hist + 1e-10) / area if any(hist) else 0.0
        except:
            pass
    
    # ENHANCED PALETTE FEATURES (8 features instead of 3)
    palette_present = 1.0 if original_mode == 'P' else 0.0
    palette_length = 0.0
    palette_entropy_val = 0.0
    palette_lsb_bias = 0.0  # NEW: LSB bias in palette indices
    palette_color_variance = 0.0  # NEW: Variance in palette colors
    palette_usage_entropy = 0.0  # NEW: How uniformly palette indices are used
    palette_sequential_bias = 0.0  # NEW: Sequential pattern detection
    palette_lsb_chi2 = 0.0  # NEW: Chi-square test on LSBs
    
    if palette_present:
        try:
            palette = img.getpalette()
            if palette:
                palette_length = len(palette) / 3
                
                # Palette color variance (how diverse are the colors?)
                palette_colors = np.array(palette).reshape(-1, 3)
                palette_color_variance = np.var(palette_colors) / 65025.0
                
                # Get pixel indices
                img_array = np.array(img)
                indices = img_array.flatten()
                
                # Usage entropy (how uniformly are indices distributed?)
                index_counts = np.bincount(indices, minlength=256)
                palette_usage_entropy = entropy(index_counts + 1)
                
                # LSB bias (are LSBs of indices random or biased?)
                lsbs = indices & 1
                lsb_count = np.sum(lsbs)
                expected_lsb = len(lsbs) / 2
                palette_lsb_bias = abs(lsb_count - expected_lsb) / len(lsbs)
                
                # Sequential bias (do indices follow patterns?)
                if len(indices) > 1:
                    diffs = np.abs(np.diff(indices.astype(int)))
                    # Stego often has small sequential changes
                    small_changes = np.sum(diffs <= 2) / len(diffs)
                    palette_sequential_bias = small_changes
                
                # Chi-square test on LSB distribution
                lsb_expected = len(lsbs) / 2
                lsb_observed = [lsb_count, len(lsbs) - lsb_count]
                lsb_expected_arr = [lsb_expected, lsb_expected]
                if lsb_expected > 0:
                    chi2_stat = sum((o - e)**2 / e for o, e in zip(lsb_observed, lsb_expected_arr))
                    palette_lsb_chi2 = min(chi2_stat / len(lsbs), 1.0)
                
        except Exception:
            pass
    
    # EOI features (only for JPEG) - IMPROVED: Filter out legitimate post-EOI data
    eof_length_norm = 0.0
    if original_format == 'JPEG':
        try:
            with open(path, 'rb') as f:
                data = f.read()
            if data.startswith(b'\xff\xd8'):
                eoi_pos = data.rfind(b'\xff\xd9')
                if eoi_pos >= 0:
                    payload = data[eoi_pos + 2:]
                    
                    # Check if payload looks like legitimate JPEG metadata vs steganography
                    is_suspicious = False
                    
                    if len(payload) > 0:
                        # Check for stego markers
                        if payload.startswith(b'0xAI42') or payload.startswith(b'AI42'):
                            is_suspicious = True
                        # Check for high entropy (random data suggests stego)
                        elif len(payload) > 100:
                            # Calculate byte entropy
                            byte_counts = np.bincount(np.frombuffer(payload[:min(1000, len(payload))], dtype=np.uint8), minlength=256)
                            payload_entropy = entropy(byte_counts + 1)
                            # High entropy (>7.0) suggests encrypted/random data
                            if payload_entropy > 7.0:
                                is_suspicious = True
                        # Check for unusual length (very long trailing data)
                        elif len(payload) > 10000:  # >10KB is suspicious
                            is_suspicious = True
                        # Check if it's NOT a known format
                        # Legitimate data often starts with known markers:
                        # - FF D8 (JPEG thumbnail)
                        # - Exif, JFIF, ICC, XMP markers
                        # - Padding (00 00 00...)
                        elif not (payload.startswith(b'\xff\xd8') or  # JPEG thumbnail
                                  payload.startswith(b'Exif') or
                                  payload.startswith(b'ICC_PROFILE') or
                                  payload.startswith(b'<?xpacket') or
                                  payload.startswith(b'http://ns.adobe.com') or
                                  all(b == 0 for b in payload[:min(20, len(payload))])):  # Padding
                            # Unknown format - potentially suspicious
                            if len(payload) > 50:  # Only flag if substantial
                                is_suspicious = True
                    
                    # Only set feature if suspicious
                    if is_suspicious:
                        eof_length = len(payload)
                        eof_length_norm = min(eof_length / area, 1.0)
        except:
            pass
    
    # Alpha channel features (only for real alpha)
    has_real_alpha = original_mode in ('RGBA', 'LA', 'PA')
    has_alpha = 1.0 if has_real_alpha else 0.0
    alpha_variance = 0.0
    alpha_mean = 0.5
    alpha_unique_ratio = 0.0
    alpha_lsb_bias = 0.0  # NEW: LSB bias in alpha channel
    
    if has_real_alpha and original_mode == 'RGBA':
        try:
            img_array = np.array(img)
            if img_array.shape[-1] == 4:
                alpha_channel = img_array[:, :, 3].astype(float)
                unique_alphas = np.unique(alpha_channel)
                
                # Skip if fake alpha (all 255)
                if not (len(unique_alphas) == 1 and unique_alphas[0] == 255):
                    alpha_variance = np.var(alpha_channel) / 65025.0
                    alpha_mean = np.mean(alpha_channel) / 255.0
                    alpha_unique_ratio = len(unique_alphas) / min(alpha_channel.size, 256)
                    
                    # LSB bias in alpha channel
                    alpha_lsbs = alpha_channel.astype(int) & 1
                    lsb_count = np.sum(alpha_lsbs)
                    expected = alpha_lsbs.size / 2
                    alpha_lsb_bias = abs(lsb_count - expected) / alpha_lsbs.size
                else:
                    has_alpha = 0.0
        except:
            pass
    
    # ENHANCED LSB DETECTION FEATURES (3 features)
    rgb_lsb_bias = 0.0  # NEW: LSB bias across RGB channels
    rgb_lsb_chi2 = 0.0  # NEW: Chi-square on RGB LSBs
    rgb_correlation = 0.0  # NEW: Inter-channel correlation
    
    try:
        if original_mode in ('RGB', 'RGBA'):
            img_array = np.array(img)
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                rgb = img_array[:, :, :3]
                
                # LSB bias across all RGB channels
                lsbs = rgb.astype(int) & 1
                lsb_flat = lsbs.flatten()
                lsb_count = np.sum(lsb_flat)
                expected = len(lsb_flat) / 2
                rgb_lsb_bias = abs(lsb_count - expected) / len(lsb_flat)
                
                # Chi-square test
                observed = [lsb_count, len(lsb_flat) - lsb_count]
                expected_arr = [expected, expected]
                if expected > 0:
                    chi2_stat = sum((o - e)**2 / e for o, e in zip(observed, expected_arr))
                    rgb_lsb_chi2 = min(chi2_stat / len(lsb_flat), 1.0)
                
                # Inter-channel correlation (stego often reduces this)
                r_flat = rgb[:, :, 0].flatten()
                g_flat = rgb[:, :, 1].flatten()
                b_flat = rgb[:, :, 2].flatten()
                
                # Sample for efficiency (use every 100th pixel if large)
                if len(r_flat) > 10000:
                    step = len(r_flat) // 10000
                    r_flat = r_flat[::step]
                    g_flat = g_flat[::step]
                    b_flat = b_flat[::step]
                
                if len(r_flat) > 1:
                    rg_corr = np.corrcoef(r_flat, g_flat)[0, 1]
                    rb_corr = np.corrcoef(r_flat, b_flat)[0, 1]
                    gb_corr = np.corrcoef(g_flat, b_flat)[0, 1]
                    # Handle NaN from corrcoef
                    rg_corr = 0.0 if np.isnan(rg_corr) else rg_corr
                    rb_corr = 0.0 if np.isnan(rb_corr) else rb_corr
                    gb_corr = 0.0 if np.isnan(gb_corr) else gb_corr
                    rgb_correlation = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
                
    except:
        pass
    
    # Format indicators
    is_jpeg = 1.0 if original_format == 'JPEG' else 0.0
    is_png = 1.0 if original_format == 'PNG' else 0.0
    
    # Return 24 features (upgraded from 15)
    return np.array([
        # Basic (5)
        file_size_norm, exif_present, exif_length_norm, comment_length, exif_entropy_val,
        # Palette (8) - ENHANCED
        palette_present, palette_length, palette_entropy_val, palette_lsb_bias, 
        palette_color_variance, palette_usage_entropy, palette_sequential_bias, palette_lsb_chi2,
        # EOI (1)
        eof_length_norm,
        # Alpha (5) - ENHANCED
        has_alpha, alpha_variance, alpha_mean, alpha_unique_ratio, alpha_lsb_bias,
        # LSB/RGB (3) - NEW
        rgb_lsb_bias, rgb_lsb_chi2, rgb_correlation,
        # Format (2)
        is_jpeg, is_png
    ], dtype=np.float32)

# TWO-STAGE ARCHITECTURE: Anomaly Detection + Type Classification
class StarlightTwoStage(nn.Module):
    def __init__(self, num_stego_classes=6, feature_dim=24):
        super(StarlightTwoStage, self).__init__()
        
        # ============== STAGE 1: ANOMALY DETECTOR ==============
        # Statistical Feature Analyzer (primary for anomaly detection)
        self.anomaly_sf_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Pixel-level anomaly detection (lightweight)
        self.anomaly_pixel_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Anomaly score fusion
        self.anomaly_fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single normality score
            nn.Sigmoid()
        )
        
        # ============== STAGE 2: STEGO TYPE CLASSIFIER ==============
        
        # SRM filter for stego type detection
        kernel = torch.tensor([[[ -1.,  2., -1.],
                                [  2., -4.,  2.],
                                [ -1.,  2., -1.]]], dtype=torch.float32)
        kernel_rgba = kernel.repeat(4, 1, 1, 1)
        self.srm_conv = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=False)
        self.srm_conv.weight = nn.Parameter(kernel_rgba, requires_grad=False)
        
        # Pixel Domain branch for type classification
        resnet = models.resnet18(weights=None)
        self.rgba_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            nn.init.kaiming_normal_(self.rgba_conv1.weight, mode='fan_out', nonlinearity='relu')
        resnet.conv1 = self.rgba_conv1
        self.pd_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # SF branch for type classification (with enhanced features)
        self.type_sf_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Frequency Domain branch
        self.fd_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Type classification fusion
        self.type_fusion = nn.Sequential(
            nn.Linear(512 + 256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_stego_classes)  # 6 stego types (no clean)
        )

    def forward(self, image, features, return_stage1=False):
        # ============== STAGE 1: Anomaly Detection ==============
        sf_anomaly = self.anomaly_sf_branch(features)
        pixel_anomaly = self.anomaly_pixel_branch(image)
        
        anomaly_features = torch.cat([sf_anomaly, pixel_anomaly], dim=1)
        normality_score = self.anomaly_fusion(anomaly_features)  # [batch, 1], 1=normal, 0=anomalous
        
        if return_stage1:
            return normality_score
        
        # ============== STAGE 2: Stego Type Classification ==============
        filtered = self.srm_conv(image)
        pd_feat = self.pd_backbone(filtered).flatten(1)
        
        sf_feat = self.type_sf_branch(features)
        
        image_rgb = image[:, :3, :, :]
        dct_img = self.dct2d(image_rgb)
        fd_feat = self.fd_cnn(dct_img)
        
        concatenated = torch.cat([pd_feat, sf_feat, fd_feat], dim=1)
        stego_type_logits = self.type_fusion(concatenated)  # [batch, 6]
        
        # Combine as: [clean_logit, stego_type_logits]
        temperature = 0.7
        eps = 1e-7
        # Logit for normality_score = log(p/(1-p))
        normality_logit = torch.log(normality_score + eps) - torch.log(1 - normality_score + eps)
        clean_logit = normality_logit * temperature
        all_logits = torch.cat([clean_logit, stego_type_logits], dim=1)  # [batch, 7]
        
        return all_logits, normality_score, stego_type_logits
    
    def dct2d(self, x):
        def dct1d(y):
            N = y.size(-1)
            even = y[..., ::2]
            odd = y[..., 1::2].flip(-1)
            v = torch.cat([even, odd], dim=-1)
            Vc = torch.fft.fft(v, dim=-1)
            k = torch.arange(N, dtype=x.dtype, device=x.device) * (math.pi / (2 * N))
            W_r = torch.cos(k)
            W_i = torch.sin(k)
            V = 2 * (Vc.real * W_r - Vc.imag * W_i)
            return V
        dct_col = dct1d(x)
        dct_col = dct_col.transpose(2, 3)
        dct_row = dct1d(dct_col)
        dct_row = dct_row.transpose(2, 3)
        return dct_row
