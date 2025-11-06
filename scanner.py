#!/usr/bin/env python3
"""
Steganography Scanner and Extractor

Uses the aggregated Starlight ensemble model to identify steganography,
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

# Import ensemble model
from aggregate_models import SuperStarlightDetector, create_ensemble

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
        
        # Define a fixed high-pass filter kernel (from SRM)
        srm_kernel = [[-1, 2, -2, 2, -1],
                      [2, -6, 8, -6, 2],
                      [-2, 8, -12, 8, -2],
                      [2, -6, 8, -6, 2],
                      [-1, 2, -2, 2, -1]]
        srm_kernel = torch.tensor(srm_kernel, dtype=torch.float32) / 12.0
        srm_kernel = srm_kernel.view(1, 1, 5, 5)
        
        # Create a non-trainable conv layer to apply this filter to each RGB channel
        self.srm_filter = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False, groups=3)
        self.srm_filter.weight.data = torch.cat([srm_kernel] * 3, dim=0)
        self.srm_filter.weight.requires_grad = False

        # The rest of the network is trainable and processes the noise residuals
        self.trainable_conv = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1), # Takes the 3-channel residual as input
            nn.BatchNorm2d(dim), 
            nn.Tanh(),
            nn.Conv2d(dim, dim, 3, 1, 1), 
            nn.BatchNorm2d(dim), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, rgb):
        # De-normalize the image tensor from [0, 1] to [0, 255] before filtering
        rgb_255 = rgb * 255.0

        # Pass the image through the fixed high-pass filter to get noise residuals
        residuals = self.srm_filter(rgb_255)
        
        # Pass residuals to the trainable part of the network
        return self.trainable_conv(residuals).flatten(1)

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
        
        # Split the classifier into a feature fusion part and a final classification layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward_features(self, rgb, alpha, exif, eoi, palette, indices):
        f_lsb = self.lsb(rgb)
        f_alpha = self.alpha(alpha)
        f_meta = self.meta(exif, eoi)
        f_palette = self.palette(palette)
        f_rgb = self.rgb_base(rgb).flatten(1)
        f_palette_index = self.palette_index(indices)
        
        combined_features = torch.cat([f_lsb, f_alpha, f_meta, f_rgb, f_palette, f_palette_index], dim=1)
        
        # Return the final feature vector before classification
        return self.feature_fusion(combined_features)

    def forward(self, rgb, alpha, exif, eoi, palette, indices):
        features = self.forward_features(rgb, alpha, exif, eoi, palette, indices)
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
    def __init__(self, model_path=None, margin=2.0, clean_ref_path=None, use_ensemble=True):
        if use_ensemble:
            print("[INIT] Using aggregated ensemble model")
            self.detector = create_ensemble()
            self.use_ensemble = True
        else:
            print("[INIT] Using single PyTorch model")
            self.feature_extractor = UniversalStegoDetector().to(device)
            if model_path:
                self.feature_extractor.load_state_dict(torch.load(model_path, map_location=device))
            self.feature_extractor.eval()
            self.margin = margin
            self.use_ensemble = False
            
            # Establish a clean reference feature vector
            self.clean_reference_features = self._get_clean_reference_features(clean_ref_path)
        
    def _get_clean_reference_features(self, clean_ref_path):
        if clean_ref_path:
            # Use a specific clean image if provided
            print(f"[INIT] Using {clean_ref_path} as clean reference.")
            preprocessed = preprocess_image(clean_ref_path)
            if preprocessed is None:
                raise ValueError(f"Failed to preprocess clean reference image: {clean_ref_path}")
            tensors = tuple(t.unsqueeze(0).to(device) for t in preprocessed)
            with torch.no_grad():
                return self.feature_extractor.forward_features(*tensors)
        else:
            # Fallback: Use a zero tensor as a generic 'clean' baseline
            # This is a placeholder and might not be optimal. A real clean image is preferred.
            print("[INIT] No clean reference path provided. Using zero tensor as clean baseline.")
            dummy_rgb = torch.zeros(1, 3, 224, 224).to(device)
            dummy_alpha = torch.zeros(1, 1, 224, 224).to(device)
            dummy_exif = torch.zeros(1, 2).to(device)
            dummy_eoi = torch.zeros(1, 1).to(device)
            dummy_palette = torch.zeros(1, 256, 3).to(device)
            dummy_indices = torch.zeros(1, 1, 224, 224).to(device)
            with torch.no_grad():
                return self.feature_extractor.forward_features(dummy_rgb, dummy_alpha, dummy_exif, dummy_eoi, dummy_palette, dummy_indices)

    def detect(self, img_path):
        """Detect steganography in an image using ensemble or single model."""
        if self.use_ensemble:
            # Use ensemble model
            result = self.detector.predict(img_path)
            if 'error' in result:
                return None
                
            return {
                'predicted_class': 'stego' if result['predicted'] else 'clean',
                'confidence': abs(result['ensemble_probability'] - 0.5) * 2,  # Convert to confidence
                'ensemble_probability': result['ensemble_probability'],
                'stego_type': result['stego_type'],
                'is_stego': result['predicted'],
                'individual_results': result.get('individual_results', [])
            }
        else:
            # Use single PyTorch model
            preprocessed = preprocess_image(img_path)
            if preprocessed is None:
                return None

            tensors = tuple(t.unsqueeze(0).to(device) for t in preprocessed)

            with torch.no_grad():
                img_features = self.feature_extractor.forward_features(*tensors)
                distance = F.pairwise_distance(img_features, self.clean_reference_features)
                distance_val = distance.item()

                # Classify based on distance to clean reference
                is_stego = distance_val > self.margin
                predicted_class = 'stego' if is_stego else 'clean'
                
                # Confidence can be derived from how far it is from the margin
                if is_stego:
                    confidence = min(1.0, max(0.0, (distance_val - self.margin) / self.margin))
                else:
                    confidence = min(1.0, max(0.0, (self.margin - distance_val) / self.margin))

                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'distance_to_clean': distance_val,
                    'is_stego': is_stego
                }
    
    def extract(self, img_path, predicted_class):
        """Extract hidden message if detected as stego. Tries all extractors."""
        if predicted_class == 'clean':
            return None
        
        # If detected as stego, try all extractors as we don't know the specific type
        img_path_str = str(img_path)
        extraction_map = {
            'alpha': lambda: extract_alpha(img_path_str),
            'palette': lambda: extract_palette(img_path_str),
            'lsb': lambda: extract_lsb(img_path_str),
            'exif': lambda: extract_exif(img_path_str),
            'eoi': lambda: extract_eoi(img_path_str)
        }
        
        extracted_messages = {}
        for algo, extractor_func in extraction_map.items():
            try:
                message, _ = extractor_func()
                if message:
                    extracted_messages[algo] = message
            except Exception:
                pass # Ignore extraction errors for other types
        
        if extracted_messages:
            # Return the first non-empty message found, or a summary
            return extracted_messages
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
        
        # Merge detection results
        result.update(detection)
        
        if extract_messages and detection['is_stego']:
            message = self.extract(img_path, detection['predicted_class'])
            result['extracted_message'] = message if message else {}
        
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
                if 'stego_type' in result:
                    print(f"  Stego Type: {result['stego_type']}")
                if 'ensemble_probability' in result:
                    print(f"  Ensemble Probability: {result['ensemble_probability']:.3f}")
                    
                if extract_messages and result.get('extracted_message'):
                    msg_dict = result['extracted_message']
                    if isinstance(msg_dict, dict):
                        for algo, msg in msg_dict.items():
                            preview = msg[:100] + '...' if len(msg) > 100 else msg
                            print(f"  Message ({algo}): {preview}")
                    else:
                        preview = str(msg_dict)[:100] + '...' if len(str(msg_dict)) > 100 else str(msg_dict)
                        print(f"  Message: {preview}")
                else:
                    print(f"  Message: No message extracted")
        
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
    parser.add_argument('--margin', type=float, default=0.8,
                       help='Distance threshold for detection (from training log, best at 0.8)')
    parser.add_argument('--clean_ref', type=str, default=None,
                       help='Path to a known clean image to use as reference for detection. Defaults to a zero tensor if not provided.')
    parser.add_argument('--no-extract', action='store_true',
                       help='Only detect, do not extract messages')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Recursively scan subdirectories')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--detail', action='store_true',
                       help='Display detailed detection results during directory scan')
    parser.add_argument('--single-model', action='store_true',
                       help='Use single PyTorch model instead of ensemble')
    
    args = parser.parse_args()
    
    # Check if target exists
    if not os.path.exists(args.target):
        print(f"Error: Target not found: {args.target}")
        sys.exit(1)
    
    use_ensemble = not args.single_model
    
    if use_ensemble:
        print("[INIT] Using aggregated ensemble model")
    else:
        # Check if model exists for single model mode
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)
        print(f"[INIT] Loading model from {args.model}")
        print(f"[INIT] Device: {device}")
        print(f"[INIT] Detection margin: {args.margin}")
    
    scanner = StegoScanner(
        model_path=args.model if not use_ensemble else None,
        margin=args.margin, 
        clean_ref_path=args.clean_ref,
        use_ensemble=use_ensemble
    )
    
    target_path = Path(args.target)
    
    if target_path.is_file():
        print(f"\n[SCANNING] Single file: {args.target}")
        result = scanner.scan_file(target_path, extract_messages=not args.no_extract)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"File: {result['file']}")
        print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.2%})")
        if 'distance_to_clean' in result:
            print(f"Distance: {result['distance_to_clean']:.2f}")
        if 'ensemble_probability' in result:
            print(f"Ensemble Probability: {result['ensemble_probability']:.3f}")
        if 'stego_type' in result:
            print(f"Stego Type: {result['stego_type']}")
        print(f"Is Stego: {result['is_stego']}")
        
        if not args.no_extract and result.get('extracted_message'):
            print(f"\nExtracted Messages (attempted all types):")
            msg_dict = result['extracted_message']
            if isinstance(msg_dict, dict):
                for algo, msg in msg_dict.items():
                    preview = msg[:100] + '...' if len(msg) > 100 else msg
                    print(f"  - {algo}: {preview}")
            else:
                preview = str(msg_dict)[:100] + '...' if len(str(msg_dict)) > 100 else str(msg_dict)
                print(f"  - Message: {preview}")
        
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
