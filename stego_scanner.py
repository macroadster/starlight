#!/usr/bin/env python3
# =============================================================
# Project Starlight - Steganography Scanner
# Detecting Hidden Data and Identifying Algorithms in Images
#
# This script uses the Trainer Model architecture to predict the 
# presence and type of steganography in a folder of images.
# =============================================================

import os
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from PIL.ExifTags import TAGS
import math

# --- 1. CONFIGURATION AND MODEL DEFINITION ---

# Class configuration to match trainer.py's class_map
ALGO_NAMES: Dict[int, str] = {
    0: 'clean',
    1: 'alpha',
    2: 'palette',
    3: 'dct',
    4: 'lsb',
    5: 'eoi',
    6: 'exif',
}
NUM_CLASSES = len(ALGO_NAMES)  # 7 classes

# Image Transform to match trainer.py
StegoTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(path: str, img: Image.Image) -> np.ndarray:
    """Extract 13 features to match trainer.py's CustomDataset.extract_features."""
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
                if tag == 'UserComment' and isinstance(value, bytes):
                    comment_length = min(len(value) / area, 1.0)
                if isinstance(value, (bytes, str)):
                    tag_values.append(value if isinstance(value, bytes) else value.encode('utf-8'))
            if tag_values:
                lengths = [len(v) for v in tag_values]
                hist = np.histogram(lengths, bins=10, range=(0, max(lengths or [1])))[0]
                exif_entropy = entropy(hist + 1e-10) / area if any(hist) else 0.0
        except:
            comment_length = 0.0
            exif_entropy = 0.0
    
    palette_present = 1.0 if img.mode == 'P' else 0.0
    palette = img.getpalette()
    palette_length = len(palette) / 3 if palette else 0.0
    if palette_present:
        hist = img.histogram()
        palette_entropy_value = entropy([h + 1 for h in hist if h > 0]) if any(hist) else 0.0
    else:
        palette_entropy_value = 0.0
    
    with open(path, 'rb') as f:
        data = f.read()
    if img.format == 'JPEG':
        eoi_pos = data.rfind(b'\xff\xd9')
        eof_length = len(data) - (eoi_pos + 2) if eoi_pos >= 0 else 0.0
    else:
        eof_length = 0.0
    eof_length_norm = min(eof_length / area, 1.0)
    
    has_alpha = 1.0 if img.mode in ('RGBA', 'LA', 'PA') else 0.0
    alpha_variance = 0.0
    if has_alpha and img.mode == 'RGBA':
        img_array = np.array(img)
        if img_array.shape[-1] == 4:
            alpha_channel = img_array[:, :, 3].astype(float)
            alpha_variance = np.var(alpha_channel) / 65025.0
    
    is_jpeg = 1.0 if img.format == 'JPEG' else 0.0
    is_png = 1.0 if img.format == 'PNG' else 0.0
    
    return np.array([
        file_size_norm, exif_present, exif_length_norm, comment_length,
        exif_entropy, palette_present, palette_length, palette_entropy_value,
        eof_length_norm, has_alpha, alpha_variance, is_jpeg, is_png
    ], dtype=np.float32)

class StarlightTrainerModel(nn.Module):
    """Model structure aligned with trainer.py's Starlight model."""
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=13):
        super(StarlightTrainerModel, self).__init__()
        
        # SRM Convolution
        kernel = torch.tensor([[[ -1.,  2., -1.],
                                [  2., -4.,  2.],
                                [ -1.,  2., -1.]]]).repeat(3, 1, 1, 1)
        self.srm_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        self.srm_conv.weight = nn.Parameter(kernel, requires_grad=False)
        
        # Pixel Domain Backbone (ResNet-18)
        resnet = models.resnet18(weights=None)
        self.pd_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Spatial Feature MLP
        self.sf_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Frequency Domain CNN
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
        
        # Gating Networks
        self.gate_pd = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.gate_fd = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Fusion Block (512 + 256 + 256 = 1024 input features)
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Spatial Features Path
        sf_feat = self.sf_mlp(features)
        
        # Compute Gates
        gate_pd = self.gate_pd(features)
        gate_fd = self.gate_fd(features)
        
        # Pixel Domain Path
        filtered = self.srm_conv(image)
        pd_feat = self.pd_backbone(filtered).flatten(1)
        pd_feat_gated = pd_feat * gate_pd
        
        # Frequency Domain Path
        dct_img = self.dct2d(image)
        fd_feat = self.fd_cnn(dct_img)
        fd_feat_gated = fd_feat * gate_fd
        
        # Concatenate and Fuse
        concatenated = torch.cat([pd_feat_gated, sf_feat, fd_feat_gated], dim=1)
        out = self.fusion(concatenated)
        
        if self.training:
            return out, gate_pd, gate_fd
        return out
    
    def dct2d(self, x: torch.Tensor) -> torch.Tensor:
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

class ImageDataset(Dataset):
    """Dataset for loading images and extracting features."""
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            features = extract_features(str(img_path), image)
        except Exception:
            # Handle unreadable/corrupt files
            return torch.zeros(3, 224, 224), torch.zeros(13, dtype=torch.float32), str(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(features, dtype=torch.float32), str(img_path)

# --- 2. SCANNER CLASS ---

class StegoScanner:
    def __init__(self, model_path: str, device: str = 'auto', batch_size: int = 16):
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.batch_size = batch_size
        self.model = StarlightTrainerModel(num_classes=NUM_CLASSES, feature_dim=13).to(self.device)
        self.load_model(model_path)
        self.model.eval()
        self.transform = StegoTransform
        self.algo_names = ALGO_NAMES
        self.num_classes = len(ALGO_NAMES)

    def load_model(self, model_path: str):
        """Loads the pre-trained model weights."""
        print(f"Loading model from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            print("Model loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {e}. Ensure the model structure matches the trained model.")

    def scan_folder(self, folder_path: str, threshold: float = 0.5, output_json: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scans all supported images in a folder and returns results."""
        image_paths = list(Path(folder_path).glob('**/*'))
        image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']]
        
        if not image_paths:
            print(f"No supported images found in: {folder_path}")
            return []

        dataset = ImageDataset(image_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        results: List[Dict[str, Any]] = []
        
        with torch.no_grad():
            for i, (images, features, paths) in enumerate(dataloader):
                images = images.to(self.device)
                features = features.to(self.device)
                
                # Forward pass
                outputs = self.model(images, features)
                probabilities = F.softmax(outputs, dim=1)
                
                # Process batch results
                for prob, path in zip(probabilities, paths):
                    prob_clean = prob[0].item()
                    prob_stego_only = prob[1:]
                    prob_max_stego, pred_stego_idx = torch.max(prob_stego_only, dim=0)
                    
                    overall_max_prob, overall_max_idx = torch.max(prob, dim=0)
                    predicted_label = self.algo_names[overall_max_idx.item()]
                    confidence = overall_max_prob.item()
                    is_flagged = (prob_clean < threshold) and (overall_max_idx.item() != 0)
                    
                    result = {
                        'path': path,
                        'is_stego': is_flagged,
                        'prediction': predicted_label,
                        'confidence': round(confidence, 4),
                        'clean_confidence': round(prob_clean, 4),
                        'probabilities': {self.algo_names[i]: round(p.item(), 4) for i, p in enumerate(prob)},
                    }
                    results.append(result)
                    
                print(f"Processed batch {i+1} of {len(dataloader)}", end='\r')

        # Sort results by stego confidence
        def get_stego_conf(r):
            stego_probs = [v for k, v in r['probabilities'].items() if k != 'clean']
            return max(stego_probs) if stego_probs else 0.0

        results.sort(key=get_stego_conf, reverse=True)
        
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved to {output_json}")

        return results

    def print_results(self, results: List[Dict[str, Any]], threshold: float, show_all: bool, top: Optional[int]):
        """Prints a human-readable summary of the scan results."""
        flagged_count = sum(1 for r in results if r['is_stego'])
        
        print("\n\n--- SCAN SUMMARY ---")
        print(f"Total Images Scanned: {len(results)}")
        print(f"Images Flagged as Stego: {flagged_count} (Clean Prob. Threshold: {threshold:.2f})")
        print("--------------------")

        display_results = [r for r in results if r['is_stego']] if not show_all else results
        
        if top is not None:
            display_results = display_results[:top]

        if not display_results:
            print("No images flagged for steganography above the confidence threshold.")
            return

        print(f"\n--- Detailed Results ({len(display_results)} images displayed) ---")
        
        for i, res in enumerate(display_results):
            status = "✅ CLEAN" if res['prediction'] == 'clean' else "🚨 STEGO"
            stego_probs = [(name, prob) for name, prob in res['probabilities'].items() if name != 'clean']
            stego_probs.sort(key=lambda x: x[1], reverse=True)
            top_stego_algo = stego_probs[0][0] if stego_probs else 'N/A'
            top_stego_conf = stego_probs[0][1] if stego_probs else 0.0

            print(f"\n[{i+1}] {Path(res['path']).name}")
            print(f"  Status: {status}")
            print(f"  Overall Prediction: {res['prediction'].upper()} (Conf: {res['confidence']:.4f})")
            print(f"  Most Likely Stego Type: {top_stego_algo.upper()} (Conf: {top_stego_conf:.4f})")

# --- 3. MAIN EXECUTION ---

def main():
    """Main function to parse arguments and run the Stego Scanner."""
    parser = argparse.ArgumentParser(
        description="Project Starlight Steganography Scanner: Detects and classifies hidden data algorithms in images."
    )
    
    parser.add_argument('input', type=str, help="Path to the folder containing images to scan.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained Starlight model weights file (.pth).")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help="Device to use for inference ('cpu', 'cuda', 'mps', or 'auto'). Default: auto")
    parser.add_argument('--threshold', type=float, default=0.8,
                       help="Probability threshold for the 'clean' class. If clean_prob < threshold, it's flagged as stego.")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Batch size for parallel processing during inference.")
    parser.add_argument('--output', type=str, default=None,
                       help="Save results to JSON file.")
    parser.add_argument('--show-all', action='store_true',
                       help="Show results for all images, not just flagged ones.")
    parser.add_argument('--top', type=int, default=None,
                       help="Show only top N images by stego probability.")
    
    args = parser.parse_args()

    print("="*80)
    print("Project Starlight - Steganography Scanner")
    print("Unified Detection & Algorithm Classification")
    print("="*80)

    try:
        scanner = StegoScanner(
            model_path=args.model,
            device=args.device,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"\n❌ Error initializing scanner: {e}")
        if "Model file not found" in str(e):
            print("\nACTION REQUIRED: Please ensure you provide the correct path to your *trained model weights* file (e.g., --model best_model.pth).")
        return 1

    print(f"Using device: {scanner.device.type}")
    print(f"Model: StarlightTrainerModel ({scanner.num_classes} classes)")
    print(f"Model classes: {', '.join(scanner.algo_names.values())}")
    print(f"Clean Probability Threshold: {args.threshold}")
    
    try:
        results = scanner.scan_folder(
            args.input, 
            threshold=args.threshold,
            output_json=args.output
        )
    except Exception as e:
        print(f"\n❌ Error during scanning: {e}")
        return 1
    
    scanner.print_results(
        results, 
        threshold=args.threshold,
        show_all=args.show_all,
        top=args.top
    )
    
    return 0

if __name__ == "__main__":
    main()
