#!/usr/bin/env python3
"""
Training script for ChatGPT submission 2025 dataset.
Uses the UniversalStegoDetector architecture from trainer.py
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader, default_collate
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import random

# Import the existing model architecture
import sys
sys.path.append('/home/eyang/sandbox/starlight')
from trainer import (
    UniversalStegoDetector, PairedStegoDataset, SiameseStegoNet, get_exif_features, 
    get_eoi_payload_size, ALGO_TO_ID, NUM_CLASSES, device
)

def train_model(args):
    """Train the UniversalStegoDetector on ChatGPT submission dataset"""
    
    # Create datasets using PairedStegoDataset
    train_ds = PairedStegoDataset('..', subdirs=['chatgpt_submission_2025'])
    val_ds = PairedStegoDataset('..', subdirs=['chatgpt_submission_2025'])
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Create data loaders with custom collate function for paired data
    def collate_pairs(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch: return None, None, None
        return default_collate(batch)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                            num_workers=4, collate_fn=collate_pairs, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                          num_workers=4, collate_fn=collate_pairs, pin_memory=False)
    
    # Initialize model
    base_model = UniversalStegoDetector(num_classes=NUM_CLASSES).to(device)
    model = SiameseStegoNet(base_model).to(device)
    
    # Load pretrained model if specified
    if args.resume and os.path.exists(args.resume):
        try:
            model.load_state_dict(torch.load(args.resume))
            print(f"[RESUME] Loaded model from {args.resume}")
        except Exception as e:
            print(f"[WARNING] Could not load resume model: {e}")
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Early stopping
    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_acc = 0.0
            self.early_stop = False
            
        def __call__(self, val_acc, model, save_path):
            if val_acc > self.best_acc + self.min_delta:
                self.best_acc = val_acc
                self.counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"  [SAVED] New best model with accuracy: {val_acc:.2f}%")
                return True
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                return False
    
    early_stopper = EarlyStopping(patience=args.patience)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[TRAINING] Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch_data in enumerate(train_pbar):
            if batch_data[0] is None:
                continue
                
            tensors1, tensors2, labels = batch_data
            # Move all tensors in the tuples to the device
            tensors1 = tuple(t.to(device) for t in tensors1)
            tensors2 = tuple(t.to(device) for t in tensors2)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            distances = model(tensors1, tensors2)
            loss = criterion(distances, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_distances = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch_idx, batch_data in enumerate(val_pbar):
                if batch_data[0] is None:
                    continue
                    
                tensors1, tensors2, labels = batch_data
                tensors1 = tuple(t.to(device) for t in tensors1)
                tensors2 = tuple(t.to(device) for t in tensors2)
                labels = labels.to(device)

                distances = model(tensors1, tensors2)
                val_loss += criterion(distances, labels).item()
                
                all_distances.extend(distances.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Loss': f'{val_loss/(batch_idx+1):.4f}'
                })
        
        # Calculate epoch metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        # Find best threshold for accuracy on validation set
        best_acc = 0
        best_thresh = 0
        for thresh in np.arange(0.1, 2.0, 0.05):
            preds = (np.array(all_distances) > thresh).astype(float)
            acc = np.mean(preds == np.array(all_labels)) * 100
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss_avg:.4f}")
        print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {best_acc:.2f}% (threshold: {best_thresh:.2f})")
        
        # Early stopping check
        improved = early_stopper(best_acc, model.base_model, str(model_path))
        scheduler.step()
        
        if not improved:
            print(f"  No improvement. Patience: {early_stopper.counter}/{early_stopper.patience}")
        
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    
    # Final evaluation
    if model_path.exists():
        print(f"\n[FINAL] Best model saved to {model_path}")
        print(f"[FINAL] Best validation accuracy: {early_stopper.best_acc:.2f}%")
        
        # Load best model for final evaluation
        model.base_model.load_state_dict(torch.load(str(model_path)))
        model.eval()
        
        # Calculate final metrics
        all_distances = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data[0] is None:
                    continue
                    
                tensors1, tensors2, labels = batch_data
                tensors1 = tuple(t.to(device) for t in tensors1)
                tensors2 = tuple(t.to(device) for t in tensors2)
                
                distances = model(tensors1, tensors2)
                all_distances.extend(distances.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Find best threshold and calculate accuracy
        best_acc = 0
        best_thresh = 0
        for thresh in np.arange(0.1, 2.0, 0.05):
            preds = (np.array(all_distances) > thresh).astype(float)
            acc = np.mean(preds == np.array(all_labels)) * 100
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        print(f"[FINAL] Best validation accuracy: {best_acc:.2f}% (threshold: {best_thresh:.2f})")
        
        return {
            'model_path': str(model_path),
            'val_accuracy': early_stopper.best_acc,
            'best_threshold': best_thresh,
            'num_classes': NUM_CLASSES
        }
    else:
        print("[ERROR] Model file was not saved!")
        return None

if __name__ == "__main__":
    print(f"[DEVICE] Using: {device}")
    if device.type == 'cuda':
        print(f"[DEVICE] CUDA Device: {torch.cuda.get_device_name()}")
    parser = argparse.ArgumentParser(description="ChatGPT Submission Trainer")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    parser.add_argument("--subdir", type=str, default="chatgpt_submission_2025", help="Training subdirectory")
    parser.add_argument("--val_subdir", type=str, default="val", help="Validation subdirectory")
    parser.add_argument("--model", type=str, default="datasets/chatgpt_submission_2025/model/detector.pth", 
                       help="Path to save the model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=7, help="Patience for early stopping")
    parser.add_argument("--resume", type=str, help="Path to resume training from")
    
    args = parser.parse_args()
    
    # Train the model
    results = train_model(args)
    
    if results:
        print(f"\n[SUCCESS] Training completed!")
        print(f"Model saved to: {results['model_path']}")
        print(f"Validation accuracy: {results['val_accuracy']:.2f}%")
        print(f"Best threshold: {results['best_threshold']:.2f}")
    else:
        print("\n[FAILED] Training did not complete successfully!")