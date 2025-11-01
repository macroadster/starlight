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
        self.lsb_conv = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU())
    def forward(self, rgb):
        lsb = (rgb * 255).long() & 1
        return F.adaptive_avg_pool2d(self.lsb_conv(lsb.float()), 1).flatten(1)

class AlphaLSBDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.alpha_conv = nn.Sequential(nn.Conv2d(1, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU())
    def forward(self, alpha):
        alpha_lsb = (alpha * 255).long() & 1
        return F.adaptive_avg_pool2d(self.alpha_conv(alpha_lsb.float()), 1).flatten(1)

class ExifEoiDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(3, dim), nn.ReLU())
    def forward(self, exif, eoi):
        return self.fc(torch.cat([exif, eoi], dim=1))

class UniversalStegoDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dim=64):
        super().__init__()
        self.lsb = LSBDetector(dim)
        self.alpha = AlphaLSBDetector(dim)
        self.meta = ExifEoiDetector(dim)
        self.rgb_base = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), 
                                     nn.ReLU(), nn.AdaptiveAvgPool2d(1))

        fusion_dim = dim + dim + dim + dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, alpha, exif, eoi):
        f_lsb = self.lsb(rgb)
        f_alpha = self.alpha(alpha)
        f_meta = self.meta(exif, eoi)
        f_rgb = self.rgb_base(rgb).flatten(1)
        
        features = torch.cat([f_lsb, f_alpha, f_meta, f_rgb], dim=1)
        return self.classifier(features)

# --- IMAGE PROCESSING ---
def preprocess_image(img_path):
    """Preprocess image for model inference."""
    try:
        img = Image.open(img_path)
        exif_features = get_exif_features(img, img_path)
        eoi_features = torch.tensor([get_eoi_payload_size(img_path) / 1000.0], dtype=torch.float)

        if img.mode == 'RGBA':
            rgb, alpha_channel = img.convert('RGB'), np.array(img)[:, :, 3] / 255.0
            alpha = torch.from_numpy(alpha_channel).float().unsqueeze(0)
        else:
            rgb, alpha = img.convert('RGB'), torch.zeros(1, img.size[1], img.size[0])

        # Resize to 224x224
        from torchvision import transforms
        resize = transforms.Resize((224, 224), antialias=True)
        rgb_tensor = transforms.ToTensor()(resize(rgb))
        alpha_tensor = F.interpolate(alpha.unsqueeze(0), size=(224, 224), 
                                     mode='bilinear', align_corners=False).squeeze(0)

        return rgb_tensor, alpha_tensor, exif_features, eoi_features
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
        
        rgb, alpha, exif, eoi = preprocessed
        
        with torch.no_grad():
            rgb = rgb.unsqueeze(0).to(device)
            alpha = alpha.unsqueeze(0).to(device)
            exif = exif.unsqueeze(0).to(device)
            eoi = eoi.unsqueeze(0).to(device)
            
            outputs = self.model(rgb, alpha, exif, eoi)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            predicted_class = ID_TO_ALGO[predicted.item()]
            confidence_score = confidence.item()
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
            top3_predictions = [
                (ID_TO_ALGO[idx.item()], prob.item()) 
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
                      output_file=None):
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
        
        # Display all detections after progress bar completes
        if stego_detections:
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
            output_file=args.output
        )
    
    else:
        print(f"Error: Invalid target (not a file or directory)")
        sys.exit(1)

if __name__ == "__main__":
    main()
