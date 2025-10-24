#!/usr/bin/env python3
import os
import re
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm

# Import the model, transforms, and feature extraction
from starlight_model import StarlightTwoStage, transform_train, transform_val, extract_features

# Focal Loss with class-specific alpha
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha_clean=3.0, alpha_stego=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha_clean = alpha_clean
        self.alpha_stego = alpha_stego
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha = torch.where(targets == 1, self.alpha_clean, self.alpha_stego)
        focal_loss = alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Custom Dataset with PROPER RGBA handling
class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform=None, augment=False):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        # Open image in original format
        img = Image.open(path)
        original_mode = img.mode
        
        # Extract features from ORIGINAL image
        features = extract_features(path, img)
        
        # CRITICAL: Only preserve alpha for images that ORIGINALLY had it
        if original_mode in ('RGBA', 'LA', 'PA'):
            img_processed = img.convert('RGBA')
        else:
            # For non-alpha images: convert to RGB, then add dummy alpha
            img_rgb = img.convert('RGB')
            img_processed = Image.new('RGBA', img_rgb.size, (0, 0, 0, 255))
            img_processed.paste(img_rgb, (0, 0))
        
        if self.transform:
            img_processed = self.transform(img_processed)
        
        return img_processed, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def collect_data(root_dir='datasets'):
    image_paths = []
    labels = []
    class_map = {'clean': 0, 'alpha': 1, 'palette': 2, 'dct': 3, 'lsb': 4, 'eoi': 5, 'exif': 6}
    pattern = re.compile(r'_({})_(\d{{3}})\.\w+$'.format('|'.join(class_map.keys())))
    
    for subdir in os.listdir(root_dir):
        if '_submission_' in subdir:
            base = os.path.join(root_dir, subdir)
            clean_dir = os.path.join(base, 'clean')
            stego_dir = os.path.join(base, 'stego')
            
            if os.path.exists(clean_dir):
                for fname in os.listdir(clean_dir):
                    fpath = os.path.join(clean_dir, fname)
                    if os.path.isfile(fpath):
                        image_paths.append(fpath)
                        labels.append(0)
            
            if os.path.exists(stego_dir):
                for fname in os.listdir(stego_dir):
                    fpath = os.path.join(stego_dir, fname)
                    if os.path.isfile(fpath):
                        match = pattern.search(fname)
                        if match:
                            algo = match.group(1)
                            image_paths.append(fpath)
                            labels.append(class_map[algo])
    
    if not image_paths:
        print("Warning: No valid images found in the dataset.")
    
    return image_paths, labels

# Main Training Script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Starlight Steganography Detection Model")
    parser.add_argument('--root_dir', default='datasets', help="Root directory containing training data")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--patience', type=int, default=8, help="Early stopping patience")
    args = parser.parse_args()
    
    root_dir = args.root_dir
    batch_size = args.batch_size
    max_epochs = args.epochs
    learning_rate = args.lr
    patience_val = args.patience
    
    # Collect data
    
    # Collect data
    image_paths, labels = collect_data(root_dir)
    if not image_paths:
        print("Error: No data found.")
        exit()

    class_map = {'clean': 0, 'alpha': 1, 'palette': 2, 'dct': 3, 'lsb': 4, 'eoi': 5, 'exif': 6}
    label_counts = Counter(labels)
    print("Dataset Statistics:")
    for algo, idx in class_map.items():
        count = label_counts.get(idx, 0)
        print(f"  {algo}: {count} samples ({100. * count / len(labels):.2f}%)")
    print(f"Total samples: {len(labels)}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # BETTER APPROACH: Use WeightedRandomSampler instead of manual oversampling
    # This gives better class balance without inflating dataset size
    class_sample_counts = [sum(1 for l in train_labels if l == i) for i in range(7)]
    weights = [1.0 / class_sample_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(train_labels), replacement=True)

    train_dataset = CustomDataset(train_paths, train_labels, transform=transform_train)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform_val)
    
    
    # Use sampler instead of shuffle, with configurable batch size for stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                             num_workers=4, drop_last=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, 
                           num_workers=4, drop_last=False, pin_memory=False)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    # Model now expects 24 features (enhanced from 15)
    model = StarlightTwoStage(num_stego_classes=6, feature_dim=24).to(device)
    
    # BETTER OPTIMIZER: Configurable learning rate, less aggressive weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing instead of ReduceLROnPlateau for smoother training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    patience = 8  # Increased patience
    best_val_loss = float('inf')
    best_balanced_acc = 0.0
    counter = 0
    accum_steps = 1  # Remove gradient accumulation for cleaner updates

    # More balanced class weights
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor([
        total_samples / (len(class_map) * label_counts[i]) 
        for i in range(len(class_map))
    ], dtype=torch.float32).to(device)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_map)
    
    stego_class_weights = class_weights[1:].clone()
    stego_class_weights = stego_class_weights / stego_class_weights.sum() * 6

    focal_loss = FocalLoss(alpha_clean=2.0, alpha_stego=1.0, gamma=2.0).to(device)

    print(f"\nClass weights: {class_weights}")
    print(f"Stego weights: {stego_class_weights}")
    print(f"Training config: batch_size={batch_size}, epochs={max_epochs}, lr={learning_rate}\n")

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{max_epochs} [Train]", unit="batch") as pbar:
            for i, (images, features, labels_batch) in enumerate(train_loader):
                images, features, labels_batch = images.to(device), features.to(device), labels_batch.to(device)
                
                optimizer.zero_grad()
                
                outputs, normality_score, stego_type_logits = model(images, features)
                
                is_clean = (labels_batch == 0).float().unsqueeze(1)
                stage1_loss = focal_loss(normality_score, is_clean)
                
                stego_mask = (labels_batch > 0)
                if stego_mask.any():
                    stego_labels = labels_batch[stego_mask] - 1
                    stego_logits = stego_type_logits[stego_mask]
                    stage2_loss = F.cross_entropy(stego_logits, stego_labels, weight=stego_class_weights)
                else:
                    stage2_loss = torch.tensor(0.0, device=device)
                
                overall_loss = F.cross_entropy(outputs, labels_batch, weight=class_weights)
                
                # Adjusted loss weighting: focus more on overall classification
                loss = 1.0 * stage1_loss + 1.0 * stage2_loss + 2.0 * overall_loss
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()
                
                pbar.update(1)
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100. * train_correct / train_total:.2f}%"
                })

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_trues = []
        all_outputs = []
        all_normality_scores = []
        
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{max_epochs} [Val]", unit="batch") as pbar:
                for images, features, labels_batch in val_loader:
                    images, features, labels_batch = images.to(device), features.to(device), labels_batch.to(device)
                    
                    outputs, normality_score, stego_type_logits = model(images, features)
                    
                    is_clean = (labels_batch == 0).float().unsqueeze(1)
                    stage1_loss = focal_loss(normality_score, is_clean)
                    
                    stego_mask = (labels_batch > 0)
                    if stego_mask.any():
                        stego_labels = labels_batch[stego_mask] - 1
                        stego_logits = stego_type_logits[stego_mask]
                        stage2_loss = F.cross_entropy(stego_logits, stego_labels, weight=stego_class_weights)
                    else:
                        stage2_loss = torch.tensor(0.0, device=device)
                    
                    overall_loss = F.cross_entropy(outputs, labels_batch, weight=class_weights)
                    
                    loss = 1.0 * stage1_loss + 1.0 * stage2_loss + 2.0 * overall_loss
                    val_loss += loss.item()
                    
                    preds = outputs.argmax(1).cpu().tolist()
                    all_preds.extend(preds)
                    all_trues.extend(labels_batch.cpu().tolist())
                    all_outputs.append(outputs.cpu())
                    all_normality_scores.extend(normality_score.squeeze().cpu().tolist())
                    
                    pbar.update(1)

        val_loss /= len(val_loader)
        all_outputs = torch.cat(all_outputs, dim=0)
        trues_np = np.array(all_trues)
        preds_np = np.array(all_preds)

        bal_acc = balanced_accuracy_score(trues_np, preds_np)
        f1 = f1_score(trues_np, preds_np, average='macro')
        per_class_f1 = f1_score(trues_np, preds_np, average=None, zero_division=0)
        acc = accuracy_score(trues_np, preds_np)
        
        binary_true = (trues_np > 0).astype(int)
        auc = roc_auc_score(binary_true, 1 - np.array(all_normality_scores)) if len(set(binary_true)) > 1 else 0.0

        clean_recall = np.sum((trues_np == 0) & (preds_np == 0)) / max(np.sum(trues_np == 0), 1)
        clean_precision = np.sum((trues_np == 0) & (preds_np == 0)) / max(np.sum(preds_np == 0), 1)
        stego_recall = np.sum((trues_np > 0) & (preds_np > 0)) / max(np.sum(trues_np > 0), 1)

        print(f"Val Loss: {val_loss:.4f}, Balanced Acc: {bal_acc:.4f}, F1: {f1:.4f}, Top-1 Acc: {acc:.4f}")
        print(f"Clean: Recall={clean_recall:.4f}, Precision={clean_precision:.4f}")
        print(f"Stego: Recall={stego_recall:.4f}, AUC={auc:.4f}")
        
        class_names = [name for name, idx in sorted(class_map.items(), key=lambda item: item[1])]
        print(f"Per-class F1: {', '.join([f'{class_names[i]}: {v:.4f}' for i, v in enumerate(per_class_f1)])}")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(trues_np, preds_np)
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
        print(f"Per-class Accuracy: {', '.join([f'{class_names[i]}: {v:.4f}' for i, v in enumerate(per_class_acc)])}")

        scheduler.step()

        # RELAXED saving criteria - focus on balanced accuracy
        min_stego_acc = np.min([per_class_acc[i] for i in range(1, 7)])
        
        # Progressive thresholds based on epoch
        if epoch < 5:
            threshold_bal_acc = 0.50
            threshold_clean_precision = 0.60
            threshold_stego_recall = 0.50
        elif epoch < 10:
            threshold_bal_acc = 0.60
            threshold_clean_precision = 0.70
            threshold_stego_recall = 0.60
        else:
            threshold_bal_acc = 0.70
            threshold_clean_precision = 0.75
            threshold_stego_recall = 0.70
        
        save_condition = (bal_acc > threshold_bal_acc and
                          clean_precision > threshold_clean_precision and
                          stego_recall > threshold_stego_recall and
                          bal_acc > best_balanced_acc)
        
        if save_condition:
            best_val_loss = val_loss
            best_balanced_acc = bal_acc
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'balanced_acc': bal_acc,
                'val_loss': val_loss,
            }, 'best_model.pth')
            print(f"✓ SAVED - Balanced Acc: {bal_acc:.4f}, Clean: {clean_recall:.4f}/{clean_precision:.4f}, Stego: {stego_recall:.4f}")
        else:
            counter += 1
            reasons = []
            if bal_acc <= threshold_bal_acc:
                reasons.append(f"Bal acc {bal_acc:.4f} ≤ {threshold_bal_acc:.2f}")
            if clean_precision <= threshold_clean_precision:
                reasons.append(f"Clean prec {clean_precision:.4f} ≤ {threshold_clean_precision:.2f}")
            if stego_recall <= threshold_stego_recall:
                reasons.append(f"Stego recall {stego_recall:.4f} ≤ {threshold_stego_recall:.2f}")
            if bal_acc <= best_balanced_acc:
                reasons.append(f"Not improved from {best_balanced_acc:.4f}")
            
            if reasons:
                print(f"⚠ Not saving: {', '.join(reasons)}")
            
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best balanced accuracy: {best_balanced_acc:.4f}")
