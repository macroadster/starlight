#!/usr/bin/env python3
"""
Steganography Scanner and Extractor

Uses the Universal 6-class Stego Detector model to identify steganography,
then extracts hidden messages using starlight_extractor utilities.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Import extraction functions
from starlight_extractor import (
    extract_alpha, extract_palette, 
    extract_lsb, extract_exif, extract_eoi
)

try:
    import piexif
except ImportError:
    piexif = None

# --- CONFIGURATION ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ALGO_TO_ID = {
    "alpha": 0, "palette": 1, "lsb": 2,
    "exif": 3, "eoi": 4, "clean": 5,
}
ID_TO_ALGO = {v: k for k, v in ALGO_TO_ID.items()}
NUM_CLASSES = 6

# --- FEATURE EXTRACTION ---
def get_eoi_payload_size(filepath):
    filepath_str = str(filepath)
    if not filepath_str.lower().endswith(('.jpg', '.jpeg')):
        return 0
    try:
        with open(filepath_str, 'rb') as f:
            data = f.read()
        eoi_pos = data.rfind(b'\xff\xd9')
        if eoi_pos > 0:
            return len(data) - (eoi_pos + 2)
    except Exception:
        return 0
    return 0

def get_exif_features(img, filepath):
    exif_present = 0.0
    exif_len = 0.0
    filepath_str = str(filepath)
    exif_bytes = img.info.get('exif')
    if exif_bytes:
        exif_present = 1.0
        exif_len = len(exif_bytes)
    elif piexif and filepath_str.lower().endswith(('.jpg', '.jpeg')):
        try:
            exif_dict = piexif.load(filepath_str)
            if exif_dict and any(val for val in exif_dict.values() if val):
                exif_present = 1.0
                exif_len = len(piexif.dump(exif_dict))
        except Exception:
            pass
    return torch.tensor([exif_present, exif_len / 1000.0], dtype=torch.float)

# --- MODEL ARCHITECTURE ---
class LSBDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.lsb_conv = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, rgb):
        lsb = (rgb * 255).long() & 1
        return self.lsb_conv(lsb.float()).flatten(1)

class PaletteIndexDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, indices):
        # indices are normalized to [0,1]. We need to scale them back to [0, 255] to get LSB.
        lsb = (indices * 255).long() & 1
        return self.conv(lsb.float()).flatten(1)

class AlphaDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, alpha):
        alpha_lsb = (alpha * 255).long() & 1
        
        # Detect AI42 marker (MSB-first: 01000001 01001001 00110100 00110010)
        batch_size, _, h, w = alpha.shape
        marker_feature = torch.zeros(batch_size, 1).to(alpha.device)
        
        if h >= 4 and w >= 8:
            alpha_int = (alpha[:, 0, :4, :8] * 255).long()
            bits = alpha_int & 1
            # Check for AI42: A=65, I=73, 4=52, 2=50 in MSB-first order
            byte0 = (bits[:, 0, 0] << 7) | (bits[:, 0, 1] << 6) | (bits[:, 0, 2] << 5) | (bits[:, 0, 3] << 4) | \
                    (bits[:, 0, 4] << 3) | (bits[:, 0, 5] << 2) | (bits[:, 0, 6] << 1) | bits[:, 0, 7]
            byte1 = (bits[:, 1, 0] << 7) | (bits[:, 1, 1] << 6) | (bits[:, 1, 2] << 5) | (bits[:, 1, 3] << 4) | \
                    (bits[:, 1, 4] << 3) | (bits[:, 1, 5] << 2) | (bits[:, 1, 6] << 1) | bits[:, 1, 7]
            byte2 = (bits[:, 2, 0] << 7) | (bits[:, 2, 1] << 6) | (bits[:, 2, 2] << 5) | (bits[:, 2, 3] << 4) | \
                    (bits[:, 2, 4] << 3) | (bits[:, 2, 5] << 2) | (bits[:, 2, 6] << 1) | bits[:, 2, 7]
            byte3 = (bits[:, 3, 0] << 7) | (bits[:, 3, 1] << 6) | (bits[:, 3, 2] << 5) | (bits[:, 3, 3] << 4) | \
                    (bits[:, 3, 4] << 3) | (bits[:, 3, 5] << 2) | (bits[:, 3, 6] << 1) | bits[:, 3, 7]
            
            marker_match = (byte0 == 65) & (byte1 == 73) & (byte2 == 52) & (byte3 == 50)
            marker_feature = marker_match.float().unsqueeze(1)
        
        conv_features = self.alpha_conv(alpha_lsb.float()).flatten(1)
        return torch.cat([conv_features, marker_feature], dim=1)

class PaletteDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, dim, 3, 1, 1), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, dim, 3, 1, 1), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, palette):
        x = palette.permute(0, 2, 1)  # (batch, 3, 256)
        return self.conv(x).flatten(1)

class ExifEoiDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU()
        )
    def forward(self, exif, eoi):
        return self.fc(torch.cat([exif, eoi], dim=1))

class UniversalStegoDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dim=64):
        super().__init__()
        self.lsb = LSBDetector(dim)
        self.alpha = AlphaDetector(dim)
        self.meta = ExifEoiDetector(dim)
        self.palette = PaletteDetector(dim)
        self.palette_index = PaletteIndexDetector(dim)
        self.rgb_base = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        fusion_dim = dim + (dim + 1) + dim + dim + dim + dim # lsb, alpha, meta, rgb, palette, palette_index
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, alpha, exif, eoi, palette, indices):
        f_lsb = self.lsb(rgb)
        f_alpha = self.alpha(alpha)
        f_meta = self.meta(exif, eoi)
        f_palette = self.palette(palette)
        f_rgb = self.rgb_base(rgb).flatten(1)
        f_palette_index = self.palette_index(indices)
        
        features = torch.cat([f_lsb, f_alpha, f_meta, f_rgb, f_palette, f_palette_index], dim=1)
        return self.classifier(features)

# --- IMAGE PROCESSING ---
def preprocess_image(img_path):
    """Preprocess image for model inference."""
    try:
        from torchvision import transforms
        img = Image.open(img_path)
        exif_features = get_exif_features(img, img_path)
        eoi_features = torch.tensor([1.0 if get_eoi_payload_size(img_path) > 0 else 0.0], dtype=torch.float)

        # Common transforms
        resize_transform = transforms.Resize((224, 224), antialias=True)
        tensor_transform = transforms.ToTensor()

        # Initialize all tensors to zeros
        rgb_tensor = torch.zeros(3, 224, 224)
        alpha_tensor = torch.zeros(1, 224, 224)
        palette_tensor = torch.zeros(256, 3)
        indices_tensor = torch.zeros(1, 224, 224)

        if img.mode == 'P':
            # Extract palette
            palette_data = img.getpalette()
            if palette_data:
                palette_padded = (palette_data + [0] * (768 - len(palette_data)))[:768]
                palette_array = np.array(palette_padded).reshape(256, 3) / 255.0
                palette_tensor = torch.from_numpy(palette_array).float()
            
            # Extract indices, treat as a single-channel image
            indices_img = Image.fromarray(np.array(img))
            indices_resized_img = resize_transform(indices_img)
            indices_tensor = tensor_transform(indices_resized_img) # This will normalize to [0,1]

        elif img.mode == 'RGBA':
            rgb_pil = img.convert('RGB')
            rgb_tensor = tensor_transform(resize_transform(rgb_pil))
            
            # Extract alpha channel
            alpha_np = np.array(img)[:, :, 3]
            alpha_img = Image.fromarray(alpha_np)
            alpha_resized_img = resize_transform(alpha_img)
            alpha_tensor = tensor_transform(alpha_resized_img)

        else: # Grayscale, RGB, etc.
            rgb_pil = img.convert('RGB')
            rgb_tensor = tensor_transform(resize_transform(rgb_pil))

        return rgb_tensor, alpha_tensor, exif_features, eoi_features, palette_tensor, indices_tensor
    except Exception as e:
        print(f"Error preprocessing {img_path}: {e}")
        return None

# --- SCANNER ---
class StegoScanner:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = UniversalStegoDetector().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        
    def detect(self, img_path):
        """Detect steganography in an image."""
        preprocessed = preprocess_image(img_path)
        if preprocessed is None:
            return None

        rgb, alpha, exif, eoi, palette, indices = preprocessed

        with torch.no_grad():
            rgb = rgb.unsqueeze(0).to(device)
            alpha = alpha.unsqueeze(0).to(device)
            exif = exif.unsqueeze(0).to(device)
            eoi = eoi.unsqueeze(0).to(device)
            palette = palette.unsqueeze(0).to(device)
            indices = indices.unsqueeze(0).to(device)

            outputs = self.model(rgb, alpha, exif, eoi, palette, indices)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

            predicted_class = ID_TO_ALGO[int(predicted.item())]
            confidence_score = confidence.item()

            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
            top3_predictions = [
                (ID_TO_ALGO[int(idx.item())], prob.item())
                for idx, prob in zip(top3_indices[0], top3_probs[0])
            ]

            return {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'top3_predictions': top3_predictions,
                'is_stego': predicted_class != 'clean' and confidence_score > self.confidence_threshold
            }
    
    def extract(self, img_path, predicted_class):
        """Extract hidden message based on predicted class."""
        if predicted_class == 'clean':
            return None
        
        # Convert Path to string for extraction functions
        img_path_str = str(img_path)
        
        extraction_map = {
            'alpha': lambda: extract_alpha(img_path_str),
            'palette': lambda: extract_palette(img_path_str),
            'lsb': lambda: extract_lsb(img_path_str),
            'exif': lambda: extract_exif(img_path_str),
            'eoi': lambda: extract_eoi(img_path_str)
        }
        
        extractor = extraction_map.get(predicted_class)
        if extractor:
            try:
                message, _ = extractor()
                return message
            except Exception as e:
                print(f"\n  [ERROR] Extraction failed for {Path(img_path_str).name}: {e}")
                return None
        return None
    
    def scan_file(self, img_path, extract_messages=True):
        """Scan a single file."""
        result = {
            'file': str(img_path),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        detection = self.detect(img_path)
        if detection is None:
            result['status'] = 'error'
            result['error'] = 'Failed to process image'
            return result
        
        result.update(detection)
        
        if extract_messages and detection['is_stego']:
            message = self.extract(img_path, detection['predicted_class'])
            result['extracted_message'] = message if message else "No message extracted"
        
        return result
    
    def scan_directory(self, directory, extract_messages=True, recursive=True,
                      output_file=None, detail=False):
        """Scan all images in a directory."""
        directory = Path(directory)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
        
        # Find all image files
        if recursive:
            image_files = [f for f in directory.rglob('*') if f.suffix.lower() in image_extensions]
        else:
            image_files = [f for f in directory.glob('*') if f.suffix.lower() in image_extensions]
        
        print(f"\n[SCANNER] Found {len(image_files)} images to scan")
        
        results = []
        stego_detections = []  # Store detections for later display
        
        # Scan all images with progress bar
        for img_path in tqdm(image_files, desc="Scanning images"):
            result = self.scan_file(img_path, extract_messages)
            results.append(result)
            
            if result.get('is_stego', False):
                stego_detections.append((img_path, result))
        
        # Display all detections after progress bar completes if detail
        if detail and stego_detections:
            print(f"\n{'='*60}")
            print(f"DETECTIONS FOUND")
            print(f"{'='*60}")
            for img_path, result in stego_detections:
                print(f"\n[FOUND] {img_path.name}")
                print(f"  Path: {img_path}")
                print(f"  Type: {result['predicted_class']} (confidence: {result['confidence']:.2%})")
                if extract_messages and result.get('extracted_message'):
                    msg = result['extracted_message']
                    preview = msg[:100] + '...' if len(msg) > 100 else msg
                    print(f"  Message: {preview}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SCAN COMPLETE")
        print(f"{'='*60}")
        print(f"Total images scanned: {len(image_files)}")
        print(f"Steganography detected: {len(stego_detections)}")
        print(f"Clean images: {len(image_files) - len(stego_detections)}")
        
        # Type breakdown
        type_counts = {}
        for result in results:
            if result.get('is_stego', False):
                stego_type = result['predicted_class']
                type_counts[stego_type] = type_counts.get(stego_type, 0) + 1
        
        if type_counts:
            print(f"\nSteganography types found:")
            for stego_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {stego_type}: {count}")
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Steganography Scanner and Extractor")
    parser.add_argument('target', help='Image file or directory to scan')
    parser.add_argument('--model', default="models/starlight.pth", help='Path to trained model (.pth file)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--no-extract', action='store_true',
                       help='Only detect, do not extract messages')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Recursively scan subdirectories')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--detail', action='store_true',
                       help='Display detailed detection results during directory scan')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Check if target exists
    if not os.path.exists(args.target):
        print(f"Error: Target not found: {args.target}")
        sys.exit(1)
    
    print(f"[INIT] Loading model from {args.model}")
    print(f"[INIT] Device: {device}")
    print(f"[INIT] Confidence threshold: {args.threshold}")
    
    scanner = StegoScanner(args.model, args.threshold)
    
    target_path = Path(args.target)
    
    if target_path.is_file():
        print(f"\n[SCANNING] Single file: {args.target}")
        result = scanner.scan_file(target_path, extract_messages=not args.no_extract)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"File: {result['file']}")
        print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.2%})")
        print(f"Is Stego: {result['is_stego']}")
        print(f"\nTop 3 Predictions:")
        for pred_class, prob in result['top3_predictions']:
            print(f"  {pred_class}: {prob:.2%}")
        
        if not args.no_extract and result.get('extracted_message'):
            print(f"\nExtracted Message:")
            print(f"{result['extracted_message']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    elif target_path.is_dir():
        print(f"\n[SCANNING] Directory: {args.target}")
        results = scanner.scan_directory(
            args.target,
            extract_messages=not args.no_extract,
            recursive=args.recursive,
            output_file=args.output,
            detail=args.detail
        )
    
    else:
        print(f"Error: Invalid target (not a file or directory)")
        sys.exit(1)

if __name__ == "__main__":
    main()
