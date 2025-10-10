#!/usr/bin/env python3
"""
Steganography Detection Model for Project Starlight
Detects hidden information in images using deep learning
Supports: PNG Alpha LSB, BMP Palette, PNG DCT, and general stego detection
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

# ============= DATASET LOADER =============

class StegoDataset(Dataset):
    """Dataset for steganography detection"""
    
    def __init__(self, dataset_root: str = 'datasets', transform=None, max_samples=None):
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        
        # Find all submission directories
        self.pairs = []
        submission_dirs = [d for d in self.dataset_root.glob('*') if d.is_dir()]
        
        print(f"Scanning {len(submission_dirs)} submission directories...")
        
        for submission_dir in submission_dirs:
            clean_dir = submission_dir / 'clean'
            stego_dir = submission_dir / 'stego'
            
            if not clean_dir.exists() or not stego_dir.exists():
                continue
            
            # Find all image pairs in this submission
            clean_files = list(clean_dir.glob('*.*'))
            submission_pairs = 0
            
            for clean_path in clean_files:
                # Try multiple extensions
                for ext in ['.png', '.bmp', '.jpg', '.jpeg', '.webp']:
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
    
    def __len__(self):
        return len(self.pairs) * 2  # Each pair provides 2 samples
    
    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_stego = idx % 2
        
        clean_path, stego_path = self.pairs[pair_idx]
        
        # Load image
        if is_stego:
            img = Image.open(stego_path).convert('RGB')
            label = 1  # Stego
        else:
            img = Image.open(clean_path).convert('RGB')
            label = 0  # Clean
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ============= MODEL ARCHITECTURES =============

class StegoDetectorCNN(nn.Module):
    """CNN architecture optimized for steganalysis"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Feature extraction layers with small kernels (good for detecting subtle changes)
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
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        
        # Global pooling and classification
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


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


# ============= TRAINING FRAMEWORK =============

class StegoDetectorTrainer:
    """Training framework for steganography detection"""
    
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
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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
            for images, labels in tqdm(dataloader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=20, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_stego_detector.pth')
                print(f'✓ Model saved (best accuracy: {best_val_acc:.2f}%)')
    
    def save_history(self, filename='training_history.json'):
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)


# ============= INFERENCE =============

class StegoDetectorInference:
    """Inference class for steganography detection"""
    
    def __init__(self, model_path, model_class, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model_class().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """Predict if image contains hidden data"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(1).item()
            confidence = probs[0][pred_class].item()
        
        label = 'STEGO' if pred_class == 1 else 'CLEAN'
        return label, confidence
    
    def batch_predict(self, image_paths: List[str]) -> Dict:
        """Batch prediction for multiple images"""
        results = {}
        for path in tqdm(image_paths, desc='Analyzing images'):
            label, conf = self.predict(path)
            results[path] = {'label': label, 'confidence': conf}
        return results


# ============= MAIN EXECUTION =============

def main():
    print("="*60)
    print("Steganography Detection Model - Project Starlight")
    print("="*60)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset from all submissions
    print("\nLoading dataset from all submissions...")
    dataset = StegoDataset(
        dataset_root='datasets',
        transform=transform
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize model (choose architecture)
    print("\nInitializing model...")
    model = StegoDetectorResNet(num_classes=2)  # or StegoDetectorCNN()
    
    # Train
    print("\nStarting training...")
    trainer = StegoDetectorTrainer(model)
    trainer.train(train_loader, val_loader, epochs=20, lr=0.001)
    
    # Save history
    trainer.save_history()
    print("\n✓ Training complete! Model saved as 'best_stego_detector.pth'")
    
    # Example inference
    print("\n" + "="*60)
    print("Running inference example...")
    detector = StegoDetectorInference('best_stego_detector.pth', StegoDetectorResNet)
    
    # Test on sample images from all submissions
    all_stego = []
    for submission_dir in Path('datasets').glob('*/stego'):
        all_stego.extend(list(submission_dir.glob('*.png'))[:2])  # 2 from each
        all_stego.extend(list(submission_dir.glob('*.bmp'))[:2])
    
    if all_stego:
        results = detector.batch_predict([str(p) for p in all_stego[:10]])
        print("\nSample predictions:")
        for img_path, result in results.items():
            print(f"  {Path(img_path).name}: {result['label']} ({result['confidence']:.2%})")


if __name__ == "__main__":
    main()
