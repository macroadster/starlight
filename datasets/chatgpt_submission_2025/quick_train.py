#!/usr/bin/env python3
"""
Quick training script for ChatGPT submission model.
Uses simplified training for faster results.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/eyang/sandbox/starlight')
from trainer import UniversalStegoDetector, PairedStegoDataset, SiameseStegoNet, device

def quick_train():
    """Quick training with minimal epochs for demo"""
    
    # Create dataset
    train_ds = PairedStegoDataset('..', subdirs=['chatgpt_submission_2025'])
    val_ds = PairedStegoDataset('..', subdirs=['chatgpt_submission_2025'])
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Data loaders (single worker for speed)
    def collate_pairs(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch: return None, None, None
        return torch.utils.data.default_collate(batch)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, 
                            num_workers=0, collate_fn=collate_pairs)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, 
                          num_workers=0, collate_fn=collate_pairs)
    
    # Initialize model
    base_model = UniversalStegoDetector().to(device)
    model = SiameseStegoNet(base_model).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training (3 epochs)
    model.train()
    for epoch in range(3):
        total_loss = 0
        for tensors1, tensors2, labels in train_loader:
            if tensors1 is None: continue
            
            tensors1 = tuple(t.to(device) for t in tensors1)
            tensors2 = tuple(t.to(device) for t in tensors2)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            distances = model(tensors1, tensors2)
            loss = criterion(distances, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/3 - Loss: {total_loss/len(train_loader):.4f}")
    
    # Save the base model
    model_path = "model/detector.pth"
    os.makedirs("model", exist_ok=True)
    torch.save(model.base_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model.base_model

if __name__ == "__main__":
    model = quick_train()