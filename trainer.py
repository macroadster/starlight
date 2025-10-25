#!/usr/bin/env python3
"""
Integrated Starlight Loop Training: RF Teacher → CNN Student

This script integrates:
1. RF Baseline Training (if model doesn't exist)
2. Starlight Loop Training with RF as teacher
3. Progress bars for all phases

Phase 1: Train RF on all 7 classes (if needed)
Phase 2: RF labels clean images with high confidence
Phase 3: CNN learns from RF's pseudo-labels + ground truth stego labels
Result: CNN that can detect both clean AND stego types (5 classes total)
"""
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tqdm import tqdm
import pickle
import argparse

from starlight_model import extract_features, transform_train, transform_val, StarlightCNN


class PseudoLabelDataset(Dataset):
    """Dataset with RF pseudo-labels for clean + ground truth for stego"""
    def __init__(self, paths, labels, confidences, transform=None):
        self.paths = paths
        self.labels = labels  # 0=clean, 1=alpha, 2=palette, 3=sdm, 4=lsb
        self.confidences = confidences  # RF confidence for weighting
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        confidence = self.confidences[idx]
        
        img = Image.open(path)
        original_mode = img.mode
        
        features = extract_features(path, img)
        
        # Convert to RGBA
        if original_mode in ('RGBA', 'LA', 'PA'):
            img_processed = img.convert('RGBA')
        else:
            img_rgb = img.convert('RGB')
            img_processed = Image.new('RGBA', img_rgb.size, (0, 0, 0, 255))
            img_processed.paste(img_rgb, (0, 0))
        
        if self.transform:
            img_processed = self.transform(img_processed)
        
        return (img_processed, 
                torch.tensor(features, dtype=torch.float32), 
                torch.tensor(label, dtype=torch.long),
                torch.tensor(confidence, dtype=torch.float32))

def collect_rf_data(root_dir='datasets'):
    """Collect image paths and labels for RF training (all 7 classes)"""
    image_paths = []
    labels = []
    class_map = {'clean': 0, 'alpha': 1, 'palette': 2, 'sdm': 3, 'lsb': 4, 'eoi': 5, 'exif': 6}
    pattern = re.compile(r'_({})_(\d{{3}})\.\w+$'.format('|'.join(class_map.keys())))
    
    for subdir in os.listdir(root_dir):
        if '_submission_' not in subdir:
            continue
            
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
    
    return image_paths, labels

def extract_features_batch(paths, desc="Extracting features"):
    """Extract features from a batch of images with progress bar"""
    features_list = []
    valid_paths = []
    
    with tqdm(total=len(paths), desc=desc) as pbar:
        for path in paths:
            try:
                img = Image.open(path)
                features = extract_features(path, img)
                features_list.append(features)
                valid_paths.append(path)
            except Exception as e:
                # Silently skip errors during feature extraction
                pass
            pbar.update(1)
    
    return np.array(features_list), valid_paths

def train_starlight_rf(root_dir='datasets', output_path='starlight_rf.pkl'):
    """Train Random Forest baseline model"""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST BASELINE")
    print("=" * 80)
    
    # Collect data
    print("\nCollecting data...")
    image_paths, labels = collect_rf_data(root_dir)
    
    class_map = {'clean': 0, 'alpha': 1, 'palette': 2, 'sdm': 3, 'lsb': 4, 'eoi': 5, 'exif': 6}
    label_counts = Counter(labels)
    print("\nDataset Statistics:")
    for algo, idx in class_map.items():
        count = label_counts.get(idx, 0)
        print(f"  {algo}: {count} samples ({100. * count / len(labels):.2f}%)")
    print(f"Total samples: {len(labels)}")
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Extract features with progress bars
    print("\nExtracting features from training set...")
    X_train, train_paths_valid = extract_features_batch(train_paths, desc="Training features")
    train_labels_valid = [train_labels[i] for i in range(len(train_paths)) if train_paths[i] in train_paths_valid]
    
    print("Extracting features from validation set...")
    X_val, val_paths_valid = extract_features_batch(val_paths, desc="Validation features")
    val_labels_valid = [val_labels[i] for i in range(len(val_paths)) if val_paths[i] in val_paths_valid]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train Random Forest
    print("\nTraining Random Forest Classifier...")
    print("Parameters:")
    print("  - n_estimators: 200")
    print("  - max_depth: 20")
    print("  - min_samples_split: 10")
    print("  - class_weight: balanced")
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf.fit(X_train, train_labels_valid)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    y_pred = rf.predict(X_val)
    y_prob = rf.predict_proba(X_val)
    
    # Overall metrics
    bal_acc = balanced_accuracy_score(val_labels_valid, y_pred)
    print(f"\nBalanced Accuracy: {bal_acc:.4f}")
    
    # Per-class metrics
    print("\nClassification Report:")
    class_names = [name for name, idx in sorted(class_map.items(), key=lambda x: x[1])]
    print(classification_report(val_labels_valid, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels_valid, y_pred)
    print("              " + "  ".join(f"{name:>6}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>6}: {' '.join(f'{val:>6}' for val in row)}")
    
    # Clean vs Stego binary metrics
    binary_true = np.array([0 if l == 0 else 1 for l in val_labels_valid])
    binary_pred = np.array([0 if p == 0 else 1 for p in y_pred])
    
    clean_recall = np.sum((binary_true == 0) & (binary_pred == 0)) / np.sum(binary_true == 0)
    clean_precision = np.sum((binary_true == 0) & (binary_pred == 0)) / max(np.sum(binary_pred == 0), 1)
    stego_recall = np.sum((binary_true == 1) & (binary_pred == 1)) / np.sum(binary_true == 1)
    stego_precision = np.sum((binary_true == 1) & (binary_pred == 1)) / max(np.sum(binary_pred == 1), 1)
    
    print("\n" + "=" * 80)
    print("CLEAN VS STEGO METRICS")
    print("=" * 80)
    print(f"Clean Recall (True Negative Rate):    {clean_recall:.4f}")
    print(f"Clean Precision:                       {clean_precision:.4f}")
    print(f"Stego Recall (True Positive Rate):    {stego_recall:.4f}")
    print(f"Stego Precision:                       {stego_precision:.4f}")
    
    # Feature importance
    print("\n" + "=" * 80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 80)
    
    feature_names = [
        'file_size_norm', 'exif_present', 'exif_length_norm', 'comment_length', 'exif_entropy',
        'palette_present', 'palette_length', 'palette_entropy', 'palette_lsb_bias',
        'palette_color_variance', 'palette_usage_entropy', 'palette_sequential_bias', 'palette_lsb_chi2',
        'eof_length_norm',
        'has_alpha', 'alpha_variance', 'alpha_mean', 'alpha_unique_ratio', 'alpha_lsb_bias',
        'rgb_lsb_bias', 'rgb_lsb_chi2', 'rgb_correlation',
        'is_jpeg', 'is_png'
    ]
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
    
    # Save model
    print(f"\nSaving model to '{output_path}'...")
    with open(output_path, 'wb') as f:
        pickle.dump(rf, f)
    
    print("\n✓ RF Training complete!")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Clean Recall: {clean_recall:.4f}")
    
    return rf, bal_acc, clean_recall

def collect_data_with_pseudolabels(rf_model, root_dir='datasets', rf_threshold=0.9):
    """
    Collect data with RF pseudo-labels for clean images
    Only use high-confidence RF predictions (>threshold)
    """
    image_paths = []
    labels = []
    confidences = []
    
    # Stego types (pixel-based only)
    stego_map = {'alpha': 1, 'palette': 2, 'sdm': 3, 'lsb': 4}
    pattern = re.compile(r'_({})_(\d{{3}})\.\w+$'.format('|'.join(stego_map.keys())))
    
    # Collect clean images with RF pseudo-labels
    print("\nPhase 1: RF Teacher labels clean images...")
    clean_count = 0
    clean_rejected = 0
    
    # First collect all clean paths
    clean_paths = []
    for subdir in os.listdir(root_dir):
        if '_submission_' not in subdir:
            continue
        
        base = os.path.join(root_dir, subdir)
        clean_dir = os.path.join(base, 'clean')
        
        if os.path.exists(clean_dir):
            for fname in os.listdir(clean_dir):
                fpath = os.path.join(clean_dir, fname)
                if os.path.isfile(fpath):
                    clean_paths.append(fpath)
    
    # Process with progress bar
    with tqdm(total=len(clean_paths), desc="Labeling clean images") as pbar:
        for fpath in clean_paths:
            try:
                # Use RF to predict
                img = Image.open(fpath)
                features = extract_features(fpath, img)
                rf_pred = rf_model.predict([features])[0]
                rf_probs = rf_model.predict_proba([features])[0]
                
                # Only use if RF is confident it's clean
                if rf_pred == 0 and rf_probs[0] >= rf_threshold:
                    image_paths.append(fpath)
                    labels.append(0)  # Clean
                    confidences.append(rf_probs[0])
                    clean_count += 1
                else:
                    clean_rejected += 1
                    
            except Exception as e:
                clean_rejected += 1
            
            pbar.update(1)
    
    print(f"  Clean images accepted: {clean_count} (RF conf >= {rf_threshold})")
    print(f"  Clean images rejected: {clean_rejected} (RF conf < {rf_threshold})")
    
    # Collect pixel-based stego with ground truth labels
    print("\nPhase 2: Ground truth labels for stego...")
    stego_paths = []
    stego_labels = []
    
    for subdir in os.listdir(root_dir):
        if '_submission_' not in subdir:
            continue
        
        base = os.path.join(root_dir, subdir)
        stego_dir = os.path.join(base, 'stego')
        
        if os.path.exists(stego_dir):
            for fname in os.listdir(stego_dir):
                fpath = os.path.join(stego_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                
                match = pattern.search(fname)
                if match:
                    algo = match.group(1)
                    stego_paths.append(fpath)
                    stego_labels.append(stego_map[algo])
    
    # Add stego samples
    image_paths.extend(stego_paths)
    labels.extend(stego_labels)
    confidences.extend([1.0] * len(stego_paths))  # Ground truth = full confidence
    
    print(f"  Stego images: {len(stego_paths)}")
    
    return image_paths, labels, confidences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Starlight Loop Training: RF → CNN")
    parser.add_argument('--rf_model', default='starlight_rf.pkl', help="RF teacher model (will train if not exists)")
    parser.add_argument('--rf_threshold', type=float, default=0.9, 
                       help="RF confidence threshold for pseudo-labels (default: 0.9)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=30, help="Epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=8, help="Early stopping patience")
    parser.add_argument('--force_rf_train', action='store_true', help="Force RF retraining even if model exists")
    parser.add_argument('--data_dir', default='datasets', help="Root directory for datasets")
    parser.add_argument('--resume_cnn', default=None, help="Resume CNN training from checkpoint (e.g., starlight_cnn.pth)")
    parser.add_argument('--cnn_output', default='starlight_cnn.pth', help="Output path for trained CNN model")
    args = parser.parse_args()
    
    print("=" * 80)
    print("INTEGRATED FEEDBACK LOOP TRAINING: RF Teacher → CNN Student")
    print("=" * 80)
    print(f"RF acts as teacher for clean detection")
    print(f"CNN learns 5 classes: clean (pseudo-labeled) + 4 stego (ground truth)")
    print("=" * 80)
    
    # Check if RF model exists, train if not
    if not os.path.exists(args.rf_model) or args.force_rf_train:
        if args.force_rf_train:
            print(f"\n--force_rf_train specified: Retraining RF model...")
        else:
            print(f"\nRF model not found at '{args.rf_model}', training from scratch...")
        rf_model, _, _ = train_starlight_rf(root_dir=args.data_dir, output_path=args.rf_model)
    else:
        print(f"\nLoading existing RF Teacher from {args.rf_model}...")
        with open(args.rf_model, 'rb') as f:
            rf_model = pickle.load(f)
        print("✓ RF Teacher loaded")
    
    rf_model.verbose = 0
    
    # Collect data with pseudo-labels
    print("\n" + "=" * 80)
    print("COLLECTING DATA WITH PSEUDO-LABELS")
    print("=" * 80)
    
    image_paths, labels, confidences = collect_data_with_pseudolabels(
        rf_model, root_dir=args.data_dir, rf_threshold=args.rf_threshold
    )
    
    class_names = ['clean', 'alpha', 'palette', 'sdm', 'lsb']
    label_counts = Counter(labels)
    print(f"\nDataset with Pseudo-Labels:")
    for i, name in enumerate(class_names):
        count = label_counts.get(i, 0)
        print(f"  {name}: {count} samples ({100. * count / len(labels):.2f}%)")
    print(f"Total samples: {len(labels)}")
    
    # Split data
    train_paths, val_paths, train_labels, val_labels, train_conf, val_conf = train_test_split(
        image_paths, labels, confidences, test_size=0.2, stratify=labels, random_state=42
    )
    
    train_dataset = PseudoLabelDataset(train_paths, train_labels, train_conf, transform=transform_train)
    val_dataset = PseudoLabelDataset(val_paths, val_labels, val_conf, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, 
                           num_workers=4, pin_memory=False)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # Model (5 classes: clean + 4 pixel stego)
    print("\n" + "=" * 80)
    print("TRAINING STARLIGHT CNN")
    print("=" * 80)
    
    model = StarlightCNN(num_classes=5, feature_dim=24).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    best_acc = 0.0
    best_clean_recall = 0.0
    
    # Resume from checkpoint if specified
    if args.resume_cnn and os.path.exists(args.resume_cnn):
        print(f"\nResuming CNN training from checkpoint: {args.resume_cnn}")
        checkpoint = torch.load(args.resume_cnn, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('balanced_acc', 0.0)
        best_clean_recall = checkpoint.get('clean_recall', 0.0)
        print(f"✓ Resumed from epoch {start_epoch}")
        print(f"  Previous Best Balanced Acc: {best_acc:.4f}")
        print(f"  Previous Best Clean Recall: {best_clean_recall:.4f}")
    elif args.resume_cnn:
        print(f"\nWarning: Resume checkpoint '{args.resume_cnn}' not found, training from scratch")
    else:
        print("\nTraining CNN from scratch")
    
    # Class weights
    class_weights = torch.tensor([
        1.0 / label_counts[i] for i in range(5)
    ], dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * 5
    
    print(f"\nClass weights: {class_weights}")
    print(f"Training config: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print(f"RF threshold for pseudo-labels: {args.rf_threshold}")
    print(f"Output model: {args.cnn_output}")
    
    patience = args.patience
    counter = 0
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{start_epoch + args.epochs} [Train]") as pbar:
            for images, features, labels_batch, conf_batch in train_loader:
                images = images.to(device)
                features = features.to(device)
                labels_batch = labels_batch.to(device)
                conf_batch = conf_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(images, features)
                
                # Confidence-weighted loss (pseudo-labels weighted by RF confidence)
                loss = F.cross_entropy(outputs, labels_batch, weight=class_weights, reduction='none')
                loss = (loss * conf_batch).mean()
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()
                
                pbar.update(1)
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 
                                 'Acc': f'{100.*train_correct/train_total:.2f}%'})
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation with progress bar
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{start_epoch + args.epochs} [Val]  ") as pbar:
                for images, features, labels_batch, conf_batch in val_loader:
                    images = images.to(device)
                    features = features.to(device)
                    labels_batch = labels_batch.to(device)
                    conf_batch = conf_batch.to(device)
                    
                    outputs = model(images, features)
                    loss = F.cross_entropy(outputs, labels_batch, weight=class_weights, reduction='none')
                    loss = (loss * conf_batch).mean()
                    val_loss += loss.item()
                    
                    preds = outputs.argmax(1).cpu().tolist()
                    all_preds.extend(preds)
                    all_trues.extend(labels_batch.cpu().tolist())
                    
                    pbar.update(1)
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        trues_np = np.array(all_trues)
        preds_np = np.array(all_preds)
        
        acc = accuracy_score(trues_np, preds_np)
        bal_acc = balanced_accuracy_score(trues_np, preds_np)
        
        # Clean recall (most important metric!)
        clean_recall = np.sum((trues_np == 0) & (preds_np == 0)) / max(np.sum(trues_np == 0), 1)
        clean_precision = np.sum((trues_np == 0) & (preds_np == 0)) / max(np.sum(preds_np == 0), 1)
        
        print(f"\nEpoch {epoch+1}/{start_epoch + args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}, Balanced Acc: {bal_acc:.4f}")
        print(f"  Clean Recall: {clean_recall:.4f}, Clean Precision: {clean_precision:.4f}")
        
        # Per-class accuracy
        cm = confusion_matrix(trues_np, preds_np)
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
        print(f"  Per-class Acc: {', '.join([f'{class_names[i]}: {v:.2f}' for i, v in enumerate(per_class_acc)])}")
        
        scheduler.step()
        
        # Save if improved (prioritize clean recall + balanced accuracy)
        save_score = bal_acc * 0.5 + clean_recall * 0.5
        best_score = best_acc * 0.5 + best_clean_recall * 0.5
        
        if save_score > best_score:
            best_acc = bal_acc
            best_clean_recall = clean_recall
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'balanced_acc': bal_acc,
                'clean_recall': clean_recall,
                'accuracy': acc,
            }, args.cnn_output)
            print(f"  ✓ SAVED to {args.cnn_output} (Balanced Acc: {bal_acc:.4f}, Clean Recall: {clean_recall:.4f})")
        else:
            counter += 1
            print(f"  Not improved. Early stopping: {counter}/{patience}")
            if counter >= patience:
                print("\nEarly stopping triggered.")
                break
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best Balanced Accuracy: {best_acc:.4f}")
    print(f"Best Clean Recall: {best_clean_recall:.4f}")
    print(f"\nStarlight CNN Summary:")
    print(f"  - Learned clean detection from RF teacher")
    print(f"  - Can classify 5 types: clean + 4 pixel-based stego")
    print(f"  - Self-contained: no RF needed at inference time")
    print(f"  - EXIF/EOI: still use RF features at runtime")
    print(f"{'='*80}")
