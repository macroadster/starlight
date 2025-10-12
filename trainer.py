#!/usr/bin/env python3
"""
Improved Steganography Detection Model for Project Starlight
Key improvements:
- Specialized preprocessing for steganalysis
- High-pass filtering to detect subtle changes
- Better data augmentation
- Adjusted learning rate and optimization
- Global features for JPEG metadata
- Palette usage anomaly detection (uniformity as stego indicator)

Usage:
  python trainer.py                    # Train with all data
  python trainer.py --limit 1000       # Train with 1000 pairs max
  python trainer.py --epochs 50        # Train for 50 epochs
  python trainer.py --lr 0.0003        # Custom learning rate
  python trainer.py --batch-size 16    # Custom batch size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import json
from tqdm import tqdm
import random
import argparse
import os

# ============= SPECIALIZED PREPROCESSING =============

class StegoPreprocessing:
    """Preprocessing optimized for steganography detection"""
    
    @staticmethod
    def high_pass_filter(img_array):
        """Apply high-pass filter to emphasize subtle changes"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)
        
        filtered = np.zeros_like(img_array, dtype=np.float32)
        
        for c in range(3):
            channel = img_array[:, :, c].astype(np.float32)
            # Apply convolution
            from scipy.ndimage import convolve
            filtered[:, :, c] = convolve(channel, kernel, mode='constant')
        
        return filtered
    
    @staticmethod
    def get_residual(img_array):
        """Get residual (difference from smoothed version)"""
        from scipy.ndimage import gaussian_filter
        
        residual = np.zeros_like(img_array, dtype=np.float32)
        
        for c in range(3):
            channel = img_array[:, :, c].astype(np.float32)
            smoothed = gaussian_filter(channel, sigma=1.0)
            residual[:, :, c] = channel - smoothed
        
        return residual

def compute_entropy(data) -> float:
    """Compute Shannon entropy"""
    if len(data) == 0:
        return 0.0
    if isinstance(data, bytes):
        hist = np.bincount(np.frombuffer(data, np.uint8), minlength=256)
    else:
        hist = np.bincount(data.ravel().astype(np.int32), minlength=256)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0

class StegoTransform:
    """Custom transform for steganalysis"""
    
    def __init__(self, size=256, augment=True):
        self.size = size
        self.augment = augment
        self.preprocessing = StegoPreprocessing()
    
    def __call__(self, img: Image.Image, file_path: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Preserve alpha channel if present, and handle palette-indexed images
        has_alpha = img.mode in ('RGBA', 'LA') or 'transparency' in img.info
        is_palette = img.mode in ('P', 'PA')
        
        palette_uniformity = 0.0
        
        if is_palette:
            # GIF/BMP palette mode - critical for palette steganography detection!
            # Store original palette indices before conversion
            palette_indices = np.array(img, dtype=np.float32)
            
            # Get the palette (handle different PIL palette formats)
            if img.palette:
                try:
                    # Get palette data - returns (mode, data)
                    palette_mode, palette_data = img.palette.getdata()
                    # Convert bytes to numpy array and reshape
                    palette = np.frombuffer(palette_data, dtype=np.uint8).astype(np.float32)
                    # Reshape to (num_colors, 3) for RGB or (num_colors, 4) for RGBA
                    if palette_mode == 'RGB':
                        palette = palette.reshape(-1, 3)
                    elif palette_mode == 'RGBA':
                        palette = palette.reshape(-1, 4)
                    else:
                        palette = None
                except Exception as e:
                    # If palette extraction fails, just set to None
                    palette = None
            else:
                palette = None
            
            # Compute palette index uniformity (stego indicator)
            if palette_indices is not None and len(palette_indices.ravel()) > 0:
                # Count usage of each palette index
                index_counts = np.bincount(palette_indices.ravel().astype(np.int32), minlength=256)
                index_counts = index_counts[index_counts > 0]  # Only used colors
                
                if len(index_counts) > 1:
                    # Compute chi-square statistic vs uniform distribution
                    # Clean images: clustered usage (high chi-square)
                    # Stego images: uniform usage (low chi-square) due to LSB manipulation
                    
                    total_pixels = len(palette_indices.ravel())
                    expected_count = total_pixels / len(index_counts)
                    chi_square = np.sum((index_counts - expected_count) ** 2 / (expected_count + 1e-10))
                    
                    # Normalize chi-square to [0, 1]
                    # Theoretical max chi-square for maximally non-uniform distribution
                    max_chi = total_pixels * len(index_counts)
                    
                    # Uniformity score: 0 = clustered (natural), 1 = uniform (suspicious)
                    palette_uniformity = 1.0 - min(chi_square / max_chi, 1.0)
            
            # Convert to RGB for standard processing
            if 'transparency' in img.info or img.mode == 'PA':
                img_rgb = img.convert('RGBA')
                has_alpha = True
            else:
                img_rgb = img.convert('RGB')
                has_alpha = False
        else:
            palette_indices = None
            palette = None
            
            if has_alpha:
                img_rgb = img.convert('RGBA')
            else:
                img_rgb = img.convert('RGB')
        
        # Resize
        img_rgb = img_rgb.resize((self.size, self.size), Image.BILINEAR)
        
        # Resize palette indices if they exist
        if palette_indices is not None:
            from PIL import Image as PILImage
            palette_img = PILImage.fromarray(palette_indices.astype(np.uint8))
            palette_img = palette_img.resize((self.size, self.size), Image.NEAREST)
            palette_indices = np.array(palette_img, dtype=np.float32)
        
        # Convert to array
        img_array = np.array(img_rgb, dtype=np.float32)
        
        # Data augmentation (only for training)
        if self.augment and random.random() > 0.5:
            if random.random() > 0.5:
                img_array = np.fliplr(img_array)
            if random.random() > 0.5:
                k = random.randint(1, 3)
                img_array = np.rot90(img_array, k)
        
        # Handle RGB vs RGBA vs Palette-indexed (spatial channels)
        if img_array.shape[2] == 4:
            rgb_array = img_array[:, :, :3]
            alpha_array = img_array[:, :, 3:4]
            
            hp_filtered_rgb = self.preprocessing.high_pass_filter(rgb_array)
            residual_rgb = self.preprocessing.get_residual(rgb_array)
            alpha_hp = self.preprocessing.high_pass_filter(np.repeat(alpha_array, 3, axis=2))[:, :, 0:1]
            alpha_residual = self.preprocessing.get_residual(np.repeat(alpha_array, 3, axis=2))[:, :, 0:1]
            
            rgb_normalized = rgb_array / 255.0
            alpha_normalized = alpha_array / 255.0
            hp_normalized_rgb = np.clip((hp_filtered_rgb - hp_filtered_rgb.mean()) / (hp_filtered_rgb.std() + 1e-8), -3, 3) / 3.0
            res_normalized_rgb = np.clip((residual_rgb - residual_rgb.mean()) / (residual_rgb.std() + 1e-8), -3, 3) / 3.0
            hp_normalized_alpha = np.clip((alpha_hp - alpha_hp.mean()) / (alpha_hp.std() + 1e-8), -3, 3) / 3.0
            res_normalized_alpha = np.clip((alpha_residual - alpha_residual.mean()) / (alpha_residual.std() + 1e-8), -3, 3) / 3.0
            
            if palette_indices is not None:
                palette_expanded = np.expand_dims(palette_indices, axis=2)
                palette_hp = self.preprocessing.high_pass_filter(np.repeat(palette_expanded, 3, axis=2))[:, :, 0:1]
                palette_residual = self.preprocessing.get_residual(np.repeat(palette_expanded, 3, axis=2))[:, :, 0:1]
                palette_normalized = palette_expanded / 255.0
                palette_hp_norm = np.clip((palette_hp - palette_hp.mean()) / (palette_hp.std() + 1e-8), -3, 3) / 3.0
                palette_res_norm = np.clip((palette_residual - palette_residual.mean()) / (palette_residual.std() + 1e-8), -3, 3) / 3.0
                
                combined = np.concatenate([
                    rgb_normalized, alpha_normalized, palette_normalized,
                    hp_normalized_rgb, hp_normalized_alpha, palette_hp_norm,
                    res_normalized_rgb, res_normalized_alpha, palette_res_norm
                ], axis=2)  # 15 channels
            else:
                combined = np.concatenate([
                    rgb_normalized, alpha_normalized,
                    hp_normalized_rgb, hp_normalized_alpha,
                    res_normalized_rgb, res_normalized_alpha
                ], axis=2)  # 12 channels
        else:
            hp_filtered = self.preprocessing.high_pass_filter(img_array)
            residual = self.preprocessing.get_residual(img_array)
            img_normalized = img_array / 255.0
            hp_normalized = np.clip((hp_filtered - hp_filtered.mean()) / (hp_filtered.std() + 1e-8), -3, 3) / 3.0
            res_normalized = np.clip((residual - residual.mean()) / (residual.std() + 1e-8), -3, 3) / 3.0
            
            if palette_indices is not None:
                palette_expanded = np.expand_dims(palette_indices, axis=2)
                palette_hp = self.preprocessing.high_pass_filter(np.repeat(palette_expanded, 3, axis=2))[:, :, 0:1]
                palette_residual = self.preprocessing.get_residual(np.repeat(palette_expanded, 3, axis=2))[:, :, 0:1]
                palette_normalized = palette_expanded / 255.0
                palette_hp_norm = np.clip((palette_hp - palette_hp.mean()) / (palette_hp.std() + 1e-8), -3, 3) / 3.0
                palette_res_norm = np.clip((palette_residual - palette_residual.mean()) / (palette_residual.std() + 1e-8), -3, 3) / 3.0
                
                combined = np.concatenate([
                    img_normalized, palette_normalized,
                    hp_normalized, palette_hp_norm,
                    res_normalized, palette_res_norm
                ], axis=2)  # 12 channels
            else:
                combined = np.concatenate([img_normalized, hp_normalized, res_normalized], axis=2)  # 9 channels
        
        # Convert to tensor and pad to 15 channels
        tensor = torch.from_numpy(combined).permute(2, 0, 1).float()
        if tensor.shape[0] < 15:
            padding_size = 15 - tensor.shape[0]
            padding = torch.zeros(padding_size, tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor, padding], dim=0)
        
        # Global features (JPEG metadata + palette uniformity)
        exif_entropy = 0.0
        exif_size_ratio = 0.0
        has_eoi_trailing = 0.0
        eoi_entropy = 0.0
        
        if file_path is not None and getattr(img, 'format', None) == 'JPEG':
            try:
                exif = img.getexif()
                exif_bytes = exif.tobytes() if exif else b''
                exif_entropy = compute_entropy(exif_bytes) / 8.0
                exif_size = len(exif_bytes)
                file_size = os.path.getsize(file_path)
                exif_size_ratio = exif_size / file_size if file_size > 0 else 0.0
                
                with open(file_path, 'rb') as f:
                    data = f.read()
                eoi_pos = data.rfind(b'\xff\xd9')
                if eoi_pos != -1:
                    trailing = data[eoi_pos + 2:]
                    # Strip legitimate padding (nulls, spaces, newlines, carriage returns)
                    trailing_clean = trailing.strip(b'\x00\x20\x0a\x0d')
                    if len(trailing_clean) > 10:  # Ignore small artifacts
                        has_eoi_trailing = 1.0
                        eoi_entropy = compute_entropy(trailing_clean) / 8.0
            except Exception as e:
                pass  # Silent fail for robustness
        
        # Global features: [EXIF entropy, EXIF ratio, EOI flag, EOI entropy, Palette uniformity]
        global_feats = torch.tensor([
            exif_entropy, 
            exif_size_ratio, 
            has_eoi_trailing, 
            eoi_entropy, 
            palette_uniformity
        ]).float()
        
        return tensor, global_feats

# ============= IMPROVED DATASET =============

class ImprovedStegoDataset(Dataset):
    """Dataset with better handling for steganalysis"""
    
    def __init__(self, dataset_root: str = 'datasets', train=True, max_samples=None):
        self.dataset_root = Path(dataset_root)
        self.train = train
        
        self.pairs = []
        submission_dirs = [d for d in self.dataset_root.glob('*') if d.is_dir()]
        
        print(f"Scanning {len(submission_dirs)} submission directories...")
        
        for submission_dir in submission_dirs:
            clean_dir = submission_dir / 'clean'
            stego_dir = submission_dir / 'stego'
            
            if not clean_dir.exists() or not stego_dir.exists():
                continue
            
            clean_files = list(clean_dir.glob('*.*'))
            submission_pairs = 0
            
            for clean_path in clean_files:
                for ext in ['.png', '.bmp', '.jpg', '.jpeg', '.webp', '.gif']:
                    stego_path = stego_dir / (clean_path.stem + ext)
                    if stego_path.exists():
                        self.pairs.append((clean_path, stego_path))
                        submission_pairs += 1
                        break
            
            if submission_pairs > 0:
                print(f"  ✓ {submission_dir.name}: {submission_pairs} pairs")
        
        if max_samples:
            self.pairs = self.pairs[:max_samples]
        
        print(f"\nTotal: {len(self.pairs)} image pairs from all submissions")
        
        self.transform = StegoTransform(size=256, augment=train)
    
    def __len__(self):
        return len(self.pairs) * 2
    
    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_stego = idx % 2
        
        clean_path, stego_path = self.pairs[pair_idx]
        
        try:
            if is_stego:
                img_path = stego_path
                label = 1
            else:
                img_path = clean_path
                label = 0
            
            img = Image.open(img_path)
            img_tensor, global_feats = self.transform(img, str(img_path))
            return img_tensor, global_feats, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy
            return torch.zeros(15, 256, 256), torch.zeros(5), 0

# ============= IMPROVED MODEL =============

class SRNet(nn.Module):
    """
    Spatial Rich Model Network - specialized for steganalysis
    Supports RGB (9ch), RGBA (12ch), Palette-indexed (12-15ch) as spatial channels
    Global features (5D): EXIF entropy, EXIF ratio, EOI flag, EOI entropy, Palette uniformity
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Spatial input: pad to 15 channels
        self.conv1 = nn.Conv2d(15, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification with concatenated global features (256 + 5 = 261)
        self.fc1 = nn.Linear(261, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x, g):
        if x.shape[1] < 15:
            padding_size = 15 - x.shape[1]
            padding = torch.zeros(x.shape[0], padding_size, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.max_pool2d(x, 2)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Concat with global features
        x = torch.cat([x, g], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# ============= IMPROVED TRAINING =============

class ImprovedTrainer:
    """Improved training with better optimization"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for imgs, globals_feats, labels in pbar:
            imgs = imgs.to(self.device)
            globals_feats = globals_feats.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(imgs, globals_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, globals_feats, labels in tqdm(dataloader, desc='Validation'):
                imgs = imgs.to(self.device)
                globals_feats = globals_feats.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(imgs, globals_feats)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=30, lr=0.0001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_stego_detector.pth')
                print(f'✓ Model saved (best accuracy: {best_val_acc:.2f}%)')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}")
    
    def save_history(self, filename='training_history.json'):
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)

# ============= MAIN EXECUTION =============

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train steganography detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trainer.py                          # Train with all data
  python trainer.py --limit 1000             # Train with max 1000 pairs
  python trainer.py --epochs 50 --lr 0.0003  # Custom training params
  python trainer.py --batch-size 16          # Smaller batches for less memory
        """
    )
    
    parser.add_argument('--dataset-root', default='datasets', help='Root directory of datasets')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of image pairs to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model-type', choices=['srnet', 'cnn'], default='srnet', help='Model architecture')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*60)
    print("Improved Steganography Detection - Project Starlight")
    print("="*60)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}\n")
    
    print("Configuration:")
    print(f"  Dataset root: {args.dataset_root}")
    print(f"  Limit pairs: {args.limit if args.limit else 'None (use all)'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model type: {args.model_type}")
    print(f"  Random seed: {args.seed}")
    print()
    
    print("Loading dataset...")
    train_dataset = ImprovedStegoDataset(dataset_root=args.dataset_root, train=True, max_samples=args.limit)
    val_dataset = ImprovedStegoDataset(dataset_root=args.dataset_root, train=False)
    
    val_dataset.pairs = train_dataset.pairs[-len(train_dataset.pairs)//5:]
    train_dataset.pairs = train_dataset.pairs[:-len(train_dataset.pairs)//5]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    use_pin_memory = (device == 'cuda')
    num_workers = 0 if device == 'mps' else 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    print("\nInitializing model...")
    model = SRNet(num_classes=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SRNet")
    print(f"Total parameters: {total_params:,}")
    
    print("\nStarting training...")
    print("Features:")
    print("- High-pass filtering and residual features")
    print("- JPEG EXIF/EOI detection")
    print("- Palette uniformity anomaly (chi-square based)")
    print("- Global feature fusion after GAP")
    print("- Learning rate: 0.0001 with cosine annealing")
    print("- Early stopping with patience=10")
    print()
    
    trainer = ImprovedTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    # Save history
    trainer.save_history()
    print("\n✓ Training history saved")

if __name__ == "__main__":
    main()

