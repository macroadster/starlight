#!/usr/bin/env python3
"""
Project Starlight - Claude's Detector Training Script (v2 - Fixed)
Trains a steganalysis model for PNG Alpha LSB and BMP Palette methods

Author: Claude (Anthropic)
Date: 2025
License: MIT

FIXES:
- Simplified architecture (easier to train on small dataset)
- Better data augmentation
- Proper learning rate
- Fixed preprocessing filter constraints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import onnx
import torch.onnx

class StegDataset(Dataset):
    """Dataset loader for clean/stego image pairs"""
    
    def __init__(self, clean_dir, stego_dir, transform=None):
        self.clean_dir = Path(clean_dir)
        self.stego_dir = Path(stego_dir)
        self.transform = transform
        
        # Get all image files
        self.clean_images = sorted(list(self.clean_dir.glob('*.png')) + 
                                   list(self.clean_dir.glob('*.bmp')))
        self.stego_images = sorted(list(self.stego_dir.glob('*.png')) + 
                                   list(self.stego_dir.glob('*.bmp')))
        
        # Create balanced dataset
        self.samples = []
        for img_path in self.clean_images:
            self.samples.append((img_path, 0))  # 0 = clean
        for img_path in self.stego_images:
            self.samples.append((img_path, 1))  # 1 = stego
        
        print(f"Loaded {len(self.clean_images)} clean + {len(self.stego_images)} stego = {len(self.samples)} total")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path)
        
        # CRITICAL: Always output 4-channel RGBA for model consistency
        # Convert to RGBA if it has alpha, otherwise add dummy alpha
        if img.mode in ['RGBA', 'LA']:
            img = img.convert('RGBA')
        else:
            # Convert to RGB first, then add fully opaque alpha channel
            img = img.convert('RGB')
            alpha = Image.new('L', img.size, 255)
            img.putalpha(alpha)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class SimplifiedStegNet(nn.Module):
    """
    Simplified steganalysis network optimized for small datasets
    Handles both RGB (palette) and RGBA (alpha channel) images
    """
    
    def __init__(self):
        super(SimplifiedStegNet, self).__init__()
        
        # Constrained preprocessing layer (handles 3 or 4 channels)
        self.preprocessing_rgb = nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False)
        self.preprocessing_alpha = nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False)
        self._init_preprocessing()
        
        # Simple convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 = 32 RGB + 32 Alpha
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128
            
            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64
            
            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32
        )
        
        # Global pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def _init_preprocessing(self):
        """Initialize with high-pass filter (but allow training)"""
        # Simple edge detection kernel
        kernel = torch.FloatTensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]).unsqueeze(0).unsqueeze(0)
        
        # Pad to 5x5
        kernel = torch.nn.functional.pad(kernel, (1, 1, 1, 1))
        
        # RGB preprocessing
        kernel_rgb = kernel.repeat(32, 3, 1, 1) / 4.0
        self.preprocessing_rgb.weight.data = kernel_rgb
        
        # Alpha preprocessing (single channel)
        kernel_alpha = kernel.repeat(32, 1, 1, 1) / 4.0
        self.preprocessing_alpha.weight.data = kernel_alpha
        
    def forward(self, x):
        # Input is always 4 channels (RGB images have dummy alpha channel added)
        # RGBA: channels 0-2 are RGB, channel 3 is alpha
        rgb = x[:, :3, :, :]
        alpha = x[:, 3:4, :, :]
        
        # Process RGB
        rgb_features = self.preprocessing_rgb(rgb)
        rgb_features = torch.abs(rgb_features)
        
        # Process alpha
        alpha_features = self.preprocessing_alpha(alpha)
        alpha_features = torch.abs(alpha_features)
        
        # Concatenate features (64 channels total)
        x = torch.cat([rgb_features, alpha_features], dim=1)
        
        # Feature extraction
        x = self.features(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


def custom_collate(batch):
    """
    Custom collate function to handle mixed RGB (3ch) and RGBA (4ch) images
    Converts all RGB to RGBA by adding a dummy alpha channel
    """
    images = []
    labels = []
    
    for img, label in batch:
        # Check number of channels
        if img.shape[0] == 3:
            # RGB - add dummy alpha channel (all 255 = fully opaque)
            alpha = torch.ones((1, img.shape[1], img.shape[2]), dtype=img.dtype)
            img = torch.cat([img, alpha], dim=0)
        
        images.append(img)
        labels.append(label)
    
    # Now all images are 4 channels, can stack normally
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels


def train_model(clean_dir='clean', stego_dir='stego', epochs=100, batch_size=8, lr=0.001):
    """Train the steganalysis model"""
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Enhanced data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    full_dataset = StegDataset(clean_dir, stego_dir, transform=None)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Create indices for splitting
    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_subset_indices = indices[:train_size]
    val_subset_indices = indices[train_size:]
    
    # Create separate datasets with different transforms
    train_dataset = StegDataset(clean_dir, stego_dir, transform=train_transform)
    val_dataset = StegDataset(clean_dir, stego_dir, transform=val_transform)
    
    # Apply indices
    train_dataset.samples = [full_dataset.samples[i] for i in train_subset_indices]
    val_dataset.samples = [full_dataset.samples[i] for i in val_subset_indices]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, collate_fn=custom_collate)
    
    # Model
    model = SimplifiedStegNet().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model/detector_best.pth')
            print(f"  → Saved best model (val_acc={val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step(val_acc)
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping after {epoch+1} epochs (no improvement for {max_patience} epochs)")
            break
    
    # Load best model
    model.load_state_dict(torch.load('model/detector_best.pth'))
    
    # Save training history
    with open('model/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    return model, history


def export_to_onnx(model, output_path='model/detector.onnx'):
    """Export trained model to ONNX format"""
    
    # Move model to CPU for ONNX export
    model.eval()
    model_cpu = model.cpu()
    
    # Create dummy input on CPU (4 channels: RGBA)
    dummy_input = torch.randn(1, 4, 256, 256)
    
    # Export
    torch.onnx.export(
        model_cpu,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and verified: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Starlight steganalysis detector')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--clean-dir', type=str, default='clean', help='Clean images directory')
    parser.add_argument('--stego-dir', type=str, default='stego', help='Stego images directory')
    
    args = parser.parse_args()
    
    # Create model directory
    Path('model').mkdir(exist_ok=True)
    
    print("="*60)
    print("Project Starlight - Detector Training v2")
    print("="*60)
    
    # Train model
    model, history = train_model(
        clean_dir=args.clean_dir,
        stego_dir=args.stego_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    export_to_onnx(model)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model: model/detector_best.pth")
    print(f"ONNX model: model/detector.onnx")
    print(f"Training history: model/training_history.json")
    print("="*60)
