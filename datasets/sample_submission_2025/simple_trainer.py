#!/usr/bin/env python3
"""
Simple Steganography Detector - 3-Class Trainer

A minimal, focused trainer for 3-class classification: Clean, LSB, Palette.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import json
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import KFold

# --- CONFIGURATION ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- DATASET ---
class StegoDataset(Dataset):
    def __init__(self, data_dir: Path, limit: int = None, crop_size: int = 224, train: bool = True):
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.image_paths = []
        self.train = train

        clean_dir = data_dir / 'clean'
        stego_dir = data_dir / 'stego'

        # Load all image paths
        clean_files = list(clean_dir.glob('*.png')) + list(clean_dir.glob('*.jpg')) + list(clean_dir.glob('*.webp'))
        lsb_files = []
        palette_files = []
        stego_json_files = list(stego_dir.glob('*.json'))
        for json_path in stego_json_files:
            with open(json_path, 'r') as f:
                meta = json.load(f)
            technique = meta.get('embedding', {}).get('technique')
            img_path = json_path.with_suffix('')
            if not img_path.exists():
                continue
            if technique == 'lsb.rgb':
                lsb_files.append(img_path)
            elif technique == 'palette':
                palette_files.append(img_path)

        if limit and limit > 0:
            # This limit is approximate
            num_per_class = limit // 3
            clean_files = clean_files[:num_per_class]
            lsb_files = lsb_files[:num_per_class]
            palette_files = palette_files[:num_per_class]

        # Balance the dataset by undersampling
        min_len = min(len(clean_files), len(lsb_files), len(palette_files))
        clean_sampled = random.sample(clean_files, min_len)
        lsb_sampled = random.sample(lsb_files, min_len)
        palette_sampled = random.sample(palette_files, min_len)

        for path in clean_sampled:
            self.image_paths.append((path, 0)) # 0 = clean
        for path in lsb_sampled:
            self.image_paths.append((path, 1)) # 1 = lsb
        for path in palette_sampled:
            self.image_paths.append((path, 2)) # 2 = palette
        
        random.shuffle(self.image_paths)
        print(f"Dataset initialized with {len(self.image_paths)} balanced images ({min_len} of each class).")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not open {img_path}, skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.train:
            transform = transforms.Compose([
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if img.size[0] < self.crop_size or img.size[1] < self.crop_size:
            padding_x = max(0, self.crop_size - img.size[0])
            padding_y = max(0, self.crop_size - img.size[1])
            img = transforms.functional.pad(img, (padding_x // 2, padding_y // 2, padding_x - padding_x // 2, padding_y - padding_y // 2))

        tensor = transform(img)
        return tensor, torch.tensor(label, dtype=torch.long) # Use Long for CrossEntropy

# --- MODEL ---
class SimpleStegoNet(nn.Module):
    def __init__(self, dim=32, num_classes=3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(dim * 2, dim * 4, 3, 1, 1),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(),
            nn.Conv2d(dim * 4, dim * 4, 3, 1, 1),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(dim * 4, num_classes)

    def forward(self, x):
        features = self.conv_net(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

# --- TRAINING LOOP ---
def train(args):
    dataset = StegoDataset(Path(args.data_dir), limit=args.limit)
    
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\n----- FOLD {fold+1}/{args.k_folds} -----')
        
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subsampler, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subsampler, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = SimpleStegoNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [Train]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation loop inside epoch to use scheduler
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%")
            
            scheduler.step(val_loss)

        # Final accuracy for the fold
        fold_results.append(accuracy)
        print(f"Fold {fold+1} Final Accuracy: {accuracy:.2f}%")

    print('\n----- K-Fold Cross-Validation Results -----')
    print(f"Average Accuracy: {np.mean(fold_results):.2f}% (+/- {np.std(fold_results):.2f}%) ")

    model_path = Path(args.model_path)
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Final model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple 3-Class Stego Trainer with K-Fold Validation")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing clean/ and stego/ subdirectories.")
    parser.add_argument("--model_path", type=str, default="simple_detector.pth", help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per fold.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--limit", type=int, default=0, help="Approximate total images to use (for faster iteration). 0 for no limit.")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation.")
    
    args = parser.parse_args()
    
    if not Path(args.data_dir).is_dir() or not Path(args.data_dir, 'clean').is_dir() or not Path(args.data_dir, 'stego').is_dir():
        print(f"Error: data_dir '{args.data_dir}' must be a directory containing 'clean' and 'stego' subfolders.", file=sys.stderr)
        sys.exit(1)

    train(args)
