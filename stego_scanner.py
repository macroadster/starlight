#!/usr/bin/env python3
"""
Steganography Scanner - Directory Inference Tool
Scans directories for images containing hidden data
Part of Project Starlight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import csv

# ============= MODEL ARCHITECTURES (must match training) =============

class ResidualBlock(nn.Module):
    """Residual block for deeper network"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StegoDetectorResNet(nn.Module):
    """ResNet-style architecture for steganalysis"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class StegoDetectorCNN(nn.Module):
    """CNN architecture optimized for steganalysis"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============= SCANNER CLASS =============

class StegoScanner:
    """Scan directories for steganography"""
    
    def __init__(self, model_path: str, model_type: str = 'resnet', 
                 device: Optional[str] = None, batch_size: int = 16):
        """
        Initialize the scanner
        
        Args:
            model_path: Path to trained model weights
            model_type: 'resnet' or 'cnn'
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Number of images to process at once
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model
        if model_type.lower() == 'resnet':
            self.model = StegoDetectorResNet(num_classes=2)
        elif model_type.lower() == 'cnn':
            self.model = StegoDetectorCNN(num_classes=2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.batch_size = batch_size
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded: {model_type}")
    
    def find_images(self, directory: str, recursive: bool = True) -> List[Path]:
        """Find all images in directory"""
        directory = Path(directory)
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.gif'}
        images = []
        
        if recursive:
            for ext in image_extensions:
                images.extend(directory.rglob(f'*{ext}'))
        else:
            for ext in image_extensions:
                images.extend(directory.glob(f'*{ext}'))
        
        return sorted(images)
    
    def scan_image(self, image_path: Path) -> Dict:
        """Scan a single image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred_class = output.argmax(1).item()
                confidence = probs[0][pred_class].item()
            
            return {
                'path': str(image_path),
                'filename': image_path.name,
                'status': 'STEGO' if pred_class == 1 else 'CLEAN',
                'confidence': confidence,
                'stego_prob': probs[0][1].item(),
                'clean_prob': probs[0][0].item(),
                'error': None
            }
        
        except Exception as e:
            return {
                'path': str(image_path),
                'filename': image_path.name,
                'status': 'ERROR',
                'confidence': 0.0,
                'stego_prob': 0.0,
                'clean_prob': 0.0,
                'error': str(e)
            }
    
    def scan_batch(self, image_paths: List[Path]) -> List[Dict]:
        """Scan multiple images in batch"""
        results = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_tensors = []
            valid_paths = []
            
            # Load and preprocess images
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    results.append({
                        'path': str(path),
                        'filename': path.name,
                        'status': 'ERROR',
                        'confidence': 0.0,
                        'stego_prob': 0.0,
                        'clean_prob': 0.0,
                        'error': str(e)
                    })
            
            if not batch_tensors:
                continue
            
            # Process batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                pred_classes = outputs.argmax(1)
            
            # Collect results
            for j, path in enumerate(valid_paths):
                pred_class = pred_classes[j].item()
                confidence = probs[j][pred_class].item()
                
                results.append({
                    'path': str(path),
                    'filename': path.name,
                    'status': 'STEGO' if pred_class == 1 else 'CLEAN',
                    'confidence': confidence,
                    'stego_prob': probs[j][1].item(),
                    'clean_prob': probs[j][0].item(),
                    'error': None
                })
        
        return results
    
    def scan_directory(self, directory: str, recursive: bool = True, 
                      threshold: float = 0.5, output_format: str = 'console') -> Dict:
        """
        Scan entire directory for steganography
        
        Args:
            directory: Directory to scan
            recursive: Scan subdirectories
            threshold: Confidence threshold for flagging (0.0-1.0)
            output_format: 'console', 'json', or 'csv'
        
        Returns:
            Dictionary with scan results and statistics
        """
        print(f"\n{'='*60}")
        print(f"Scanning: {directory}")
        print(f"Recursive: {recursive}")
        print(f"Threshold: {threshold}")
        print(f"{'='*60}\n")
        
        # Find all images
        images = self.find_images(directory, recursive)
        
        if not images:
            print("No images found!")
            return {'images': [], 'stats': {}}
        
        print(f"Found {len(images)} images")
        print("Analyzing...\n")
        
        # Scan images
        results = []
        for img_path in tqdm(images, desc='Scanning'):
            result = self.scan_image(img_path)
            results.append(result)
        
        # Calculate statistics
        stego_count = sum(1 for r in results if r['status'] == 'STEGO' and r['confidence'] >= threshold)
        clean_count = sum(1 for r in results if r['status'] == 'CLEAN' and r['confidence'] >= threshold)
        error_count = sum(1 for r in results if r['status'] == 'ERROR')
        uncertain_count = len(results) - stego_count - clean_count - error_count
        
        stats = {
            'total_images': len(results),
            'stego_detected': stego_count,
            'clean': clean_count,
            'uncertain': uncertain_count,
            'errors': error_count,
            'threshold': threshold,
            'scan_time': datetime.now().isoformat()
        }
        
        # Display results
        self._display_results(results, stats, threshold, output_format)
        
        return {'images': results, 'stats': stats}
    
    def _display_results(self, results: List[Dict], stats: Dict, 
                        threshold: float, output_format: str):
        """Display or save results"""
        
        print(f"\n{'='*60}")
        print("SCAN RESULTS")
        print(f"{'='*60}")
        print(f"Total Images:      {stats['total_images']}")
        print(f"Stego Detected:    {stats['stego_detected']} ({stats['stego_detected']/stats['total_images']*100:.1f}%)")
        print(f"Clean:             {stats['clean']} ({stats['clean']/stats['total_images']*100:.1f}%)")
        print(f"Uncertain:         {stats['uncertain']} ({stats['uncertain']/stats['total_images']*100:.1f}%)")
        print(f"Errors:            {stats['errors']}")
        print(f"{'='*60}\n")
        
        # Show flagged images
        flagged = [r for r in results if r['status'] == 'STEGO' and r['confidence'] >= threshold]
        
        if flagged:
            print(f"🚨 FLAGGED IMAGES (threshold: {threshold}):")
            for r in sorted(flagged, key=lambda x: x['confidence'], reverse=True):
                print(f"  [{r['confidence']:.2%}] {r['filename']}")
                print(f"      Path: {r['path']}")
        else:
            print("✓ No suspicious images detected")
        
        # Save outputs
        if output_format == 'json':
            output_file = f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({'results': results, 'stats': stats}, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")
        
        elif output_format == 'csv':
            output_file = f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'path', 'status', 'confidence', 'stego_prob', 'clean_prob', 'error'])
                writer.writeheader()
                writer.writerows(results)
            print(f"\n✓ Results saved to: {output_file}")


# ============= COMMAND LINE INTERFACE =============

def main():
    parser = argparse.ArgumentParser(
        description='Steganography Scanner - Detect hidden data in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan current directory
  python stego_scanner.py scan . --model best_stego_detector.pth
  
  # Scan with high sensitivity
  python stego_scanner.py scan /path/to/images --threshold 0.3 --recursive
  
  # Export results to CSV
  python stego_scanner.py scan /path/to/images --output csv
  
  # Scan single image
  python stego_scanner.py scan image.png --model best_stego_detector.pth
        """
    )
    
    parser.add_argument('command', choices=['scan'], help='Command to run')
    parser.add_argument('path', help='Directory or image to scan')
    parser.add_argument('--model', default='best_stego_detector.pth', help='Path to model weights')
    parser.add_argument('--model-type', default='resnet', choices=['resnet', 'cnn'], help='Model architecture')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold (0.0-1.0)')
    parser.add_argument('--recursive', action='store_true', help='Scan subdirectories')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--output', choices=['console', 'json', 'csv'], default='console', help='Output format')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None, help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = StegoScanner(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Check if path is file or directory
    path = Path(args.path)
    
    if path.is_file():
        # Scan single image
        print(f"\nScanning single image: {path}")
        result = scanner.scan_image(path)
        print(f"\nResult: {result['status']} (confidence: {result['confidence']:.2%})")
        print(f"  Stego probability: {result['stego_prob']:.2%}")
        print(f"  Clean probability: {result['clean_prob']:.2%}")
    
    elif path.is_dir():
        # Scan directory
        scanner.scan_directory(
            directory=str(path),
            recursive=args.recursive,
            threshold=args.threshold,
            output_format=args.output
        )
    
    else:
        print(f"Error: Path not found: {path}")
        return 1
    
    return 0


if __name__ == "__main__":
    print("="*60)
    print("Project Starlight - Steganography Scanner")
    print("Detecting Hidden Data in Images")
    print("="*60)
    exit(main())
