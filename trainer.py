#!/usr/bin/env python3
import os
import re
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm

# Import the model, transforms, and feature extraction from the new centralized file
from starlight_model import StarlightTwoStage, transform_train, transform_val, extract_features

# Custom Dataset
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
        img = Image.open(path)
        
        # Extract features using the imported function
        features = extract_features(path, img)
        
        # Convert to RGBA
        img_rgba = img.convert('RGBA')
        
        if self.transform:
            img_rgba = self.transform(img_rgba)
        
        return img_rgba, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

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
    image_paths, labels = collect_data()
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

    train_dataset = CustomDataset(train_paths, train_labels, transform=transform_train)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    model = StarlightTwoStage(num_stego_classes=6, feature_dim=15).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    patience = 20
    best_val_loss = float('inf')
    best_clean_recall = 0.0
    counter = 0

    # Class counts from dataset statistics (computed dynamically)
    class_counts = [label_counts[0], label_counts[1], label_counts[2], label_counts[3], label_counts[4], label_counts[5], label_counts[6]]
    class_weights = torch.tensor([1.0 / (count + 1e-6) for count in class_counts], dtype=torch.float32).to(device)  # Avoid division by zero
    stego_class_weights = torch.tensor([1.0 / (count + 1e-6) for count in class_counts[1:]], dtype=torch.float32).to(device)

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/100 [Train]", unit="batch") as pbar:
            for images, features, labels_batch in train_loader:
                images, features, labels_batch = images.to(device), features.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                outputs, normality_score, stego_type_logits = model(images, features)
                
                # Three-part loss with proper weighting
                is_clean = (labels_batch == 0).float().unsqueeze(1)
                
                # Stage 1: STRONG clean vs stego signal
                stage1_loss = F.binary_cross_entropy(normality_score, is_clean)
                
                # Stage 2: Stego type classification (only for stego samples)
                stego_mask = (labels_batch > 0)
                if stego_mask.any():
                    stego_labels = labels_batch[stego_mask] - 1
                    stego_logits = stego_type_logits[stego_mask]
                    stage2_loss = F.cross_entropy(stego_logits, stego_labels, weight=stego_class_weights)
                else:
                    stage2_loss = 0.0
                
                # Overall classification loss
                overall_loss = F.cross_entropy(outputs, labels_batch, weight=class_weights)
                
                # Rebalanced: Adjusted weights for better balance
                loss = 1.0 * stage1_loss + 1.0 * stage2_loss + 0.5 * overall_loss
                
                loss.backward()
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
        print(f"Epoch {epoch+1}/100, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_trues = []
        all_outputs = []
        all_normality_scores = []
        
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/100 [Val]", unit="batch") as pbar:
                for images, features, labels_batch in val_loader:
                    images, features, labels_batch = images.to(device), features.to(device), labels_batch.to(device)
                    
                    outputs, normality_score, stego_type_logits = model(images, features)
                    
                    # Calculate losses
                    is_clean = (labels_batch == 0).float().unsqueeze(1)
                    stage1_loss = F.binary_cross_entropy(normality_score, is_clean)
                    
                    stego_mask = (labels_batch > 0)
                    if stego_mask.any():
                        stego_labels = labels_batch[stego_mask] - 1
                        stego_logits = stego_type_logits[stego_mask]
                        stage2_loss = F.cross_entropy(stego_logits, stego_labels, weight=stego_class_weights)
                    else:
                        stage2_loss = 0.0
                    
                    overall_loss = F.cross_entropy(outputs, labels_batch, weight=class_weights)
                    
                    loss = 1.0 * stage1_loss + 1.0 * stage2_loss + 0.5 * overall_loss
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
        per_class_f1 = f1_score(trues_np, preds_np, average=None)
        acc = accuracy_score(trues_np, preds_np)
        
        # Binary clean vs stego metrics
        binary_true = (trues_np > 0).astype(int)
        # 1 - normality_score is the anomaly score (probability of stego)
        auc = roc_auc_score(binary_true, 1 - np.array(all_normality_scores)) if len(set(binary_true)) > 1 else 0.0

        binary_pred = (preds_np > 0).astype(int)
        clean_recall = np.sum((trues_np == 0) & (preds_np == 0)) / np.sum(trues_np == 0)
        clean_precision = np.sum((trues_np == 0) & (preds_np == 0)) / max(np.sum(preds_np == 0), 1)
        stego_recall = np.sum((trues_np > 0) & (preds_np > 0)) / max(np.sum(trues_np > 0), 1)

        print(f"Val Loss: {val_loss:.4f}, Balanced Acc: {bal_acc:.4f}, F1: {f1:.4f}, Top-1 Acc: {acc:.4f}")
        print(f"Clean: Recall={clean_recall:.4f}, Precision={clean_precision:.4f}")
        print(f"Stego: Recall={stego_recall:.4f}, AUC={auc:.4f}")
        
        # Determine the order of classes for metrics display
        # Note: class_map keys are algo names, values are 0-6
        class_names = [name for name, idx in sorted(class_map.items(), key=lambda item: item[1])]

        print(f"Per-class F1: {', '.join([f'{class_names[i]}: {v:.4f}' for i, v in enumerate(per_class_f1)])}")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(trues_np, preds_np)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        print(f"Per-class Accuracy: {', '.join([f'{class_names[i]}: {v:.4f}' for i, v in enumerate(per_class_acc)])}")

        scheduler.step(val_loss)

        # Save condition (relaxed thresholds)
        min_stego_acc = np.min([per_class_acc[i] for i in range(1, 7)])
        save_condition = (clean_recall > 0.55 and
                          clean_precision > 0.75 and
                          stego_recall > 0.75 and 
                          min_stego_acc > 0.40 and 
                          bal_acc > 0.70 and
                          val_loss < best_val_loss)
        
        if save_condition:
            best_val_loss = val_loss
            best_clean_recall = clean_recall
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Saved - Clean Recall: {clean_recall:.4f}, Stego Recall: {stego_recall:.4f}, Val Loss: {val_loss:.4f}")
        else:
            counter += 1
            reasons = []
            if clean_recall <= 0.55:
                reasons.append(f"Clean recall: {clean_recall:.4f} ≤ 0.55")
            if clean_precision <= 0.75:
                reasons.append(f"Clean precision: {clean_precision:.4f} ≤ 0.75")
            if stego_recall <= 0.75:
                reasons.append(f"Stego recall: {stego_recall:.4f} ≤ 0.75")
            if min_stego_acc <= 0.40:
                reasons.append(f"Min stego acc: {min_stego_acc:.4f} ≤ 0.40")
            if bal_acc <= 0.70:
                reasons.append(f"Balanced acc: {bal_acc:.4f} ≤ 0.70")
            if val_loss >= best_val_loss and clean_recall > 0.55:
                reasons.append(f"Val loss not improved")
            
            if reasons:
                print(f"⚠ Not saving: {', '.join(reasons)}")
            
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best clean recall: {best_clean_recall:.4f}")
