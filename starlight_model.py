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
    Extracts 15 statistical/structural features from the image file.
    This consolidated function is used by the Dataset and the extractor/scanner.
    """
    width, height = img.size
    area = width * height if width * height > 0 else 1.0
    
    file_size = os.path.getsize(path) / 1024.0
    file_size_norm = file_size / area
    
    exif_bytes = img.info.get('exif')
    exif_present = 1.0 if exif_bytes else 0.0
    exif_length = len(exif_bytes) if exif_bytes else 0.0
    exif_length_norm = min(exif_length / area, 1.0)
    
    comment_length = 0.0
    exif_entropy = 0.0
    if exif_bytes:
        try:
            exif_dict = img.getexif()
            tag_values = []
            for tag_id, value in exif_dict.items():
                tag = TAGS.get(tag_id, tag_id)
                # Feature: UserComment length
                if tag == 'UserComment' and isinstance(value, bytes):
                    comment_length = min(len(value) / area, 1.0)
                if isinstance(value, (bytes, str)):
                    tag_values.append(value if isinstance(value, bytes) else value.encode('utf-8'))
            # Feature: EXIF entropy
            if tag_values:
                lengths = [len(v) for v in tag_values]
                max_len = max(lengths or [1])
                hist = np.histogram(lengths, bins=10, range=(0, max_len))[0]
                exif_entropy = entropy(hist + 1e-10) / area if any(hist) else 0.0
        except:
            comment_length = 0.0
            exif_entropy = 0.0
    
    # Features: Palette presence, length, and entropy
    palette_present = 1.0 if img.mode == 'P' else 0.0
    palette = img.getpalette()
    palette_length = len(palette) / 3 if palette else 0.0
    if palette_present:
        hist = img.histogram()
        palette_entropy_value = entropy([h + 1 for h in hist if h > 0]) if any(hist) else 0.0
    else:
        palette_entropy_value = 0.0
    
    # Feature: EOF length (for JPEG EOI)
    # Re-reading file is necessary to get correct EOF length regardless of PIL info
    with open(path, 'rb') as f:
        data = f.read()
    if img.format == 'JPEG':
        eoi_pos = data.rfind(b'\xff\xd9')
        eof_length = len(data) - (eoi_pos + 2) if eoi_pos >= 0 else 0.0
    else:
        eof_length = 0.0
    eof_length_norm = min(eof_length / area, 1.0)
    
    # Features: Alpha channel metrics
    has_alpha = 1.0 if img.mode in ('RGBA', 'LA', 'PA') else 0.0
    alpha_variance = 0.0
    alpha_mean = 0.5 # Default to 0.5 for non-RGBA
    alpha_unique_ratio = 0.0
    
    if has_alpha and img.mode == 'RGBA':
        img_array = np.array(img)
        if img_array.shape[-1] == 4:
            alpha_channel = img_array[:, :, 3].astype(float)
            alpha_variance = np.var(alpha_channel) / 65025.0
            alpha_mean = np.mean(alpha_channel) / 255.0
            unique_alphas = len(np.unique(alpha_channel))
            total_pixels = alpha_channel.size
            alpha_unique_ratio = unique_alphas / min(total_pixels, 256)
    
    # Features: File format indicators
    is_jpeg = 1.0 if img.format == 'JPEG' else 0.0
    is_png = 1.0 if img.format == 'PNG' else 0.0
    
    return np.array([
        file_size_norm, exif_present, exif_length_norm, comment_length, 
        exif_entropy, palette_present, palette_length, palette_entropy_value, 
        eof_length_norm, has_alpha, alpha_variance, alpha_mean, alpha_unique_ratio, 
        is_jpeg, is_png # Total 15 features
    ], dtype=np.float32)

# TWO-STAGE ARCHITECTURE: Anomaly Detection + Type Classification
class StarlightTwoStage(nn.Module):
    def __init__(self, num_stego_classes=6, feature_dim=15):
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
        
        # SF branch for type classification
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
        temperature = 0.5
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
