#!/usr/bin/env python3
import os
import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from PIL.ExifTags import TAGS
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

# Define transforms with stronger augmentation for training
transform_train = transforms.Compose([
    transforms.RandomRotation(15),  # Increased from 10
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),  # Add vertical flip
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Add color jitter
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        img = Image.open(path).convert('RGB')
        features = self.extract_features(path, img, label)
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def extract_features(self, path, img, label):
        width, height = img.size
        area = width * height if width * height > 0 else 1.0
        
        file_size = os.path.getsize(path) / 1024.0
        file_size_norm = file_size / area
        
        exif_bytes = img.info.get('exif')
        exif_present = 1.0 if exif_bytes else 0.0
        exif_length = len(exif_bytes) if exif_bytes else 0.0
        exif_length_norm = min(exif_length / area, 1.0)
        
        # EXIF comment length and entropy
        comment_length = 0.0
        exif_entropy = 0.0
        if exif_bytes:
            try:
                exif_dict = img.getexif()
                tag_values = []
                for tag_id, value in exif_dict.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'UserComment' and isinstance(value, bytes):
                        comment_length = min(len(value) / area, 1.0)
                    if isinstance(value, (bytes, str)):
                        tag_values.append(value if isinstance(value, bytes) else value.encode('utf-8'))
                if tag_values:
                    lengths = [len(v) for v in tag_values]
                    hist = np.histogram(lengths, bins=10, range=(0, max(lengths or [1])))[0]
                    exif_entropy = entropy(hist + 1e-10) / area if any(hist) else 0.0
            except:
                comment_length = 0.0
                exif_entropy = 0.0
        
        palette_present = 1.0 if img.mode == 'P' else 0.0
        palette = img.getpalette()
        palette_length = len(palette) / 3 if palette else 0.0
        if palette_present:
            hist = img.histogram()
            palette_entropy_value = entropy([h + 1 for h in hist if h > 0]) if any(hist) else 0.0
        else:
            palette_entropy_value = 0.0
        
        with open(path, 'rb') as f:
            data = f.read()
        if img.format == 'JPEG':
            eoi_pos = data.rfind(b'\xff\xd9')
            eof_length = len(data) - (eoi_pos + 2) if eoi_pos >= 0 else 0.0
        else:
            eof_length = 0.0
        eof_length_norm = min(eof_length / area, 1.0)
        
        # Add alpha channel detection
        has_alpha = 1.0 if img.mode in ('RGBA', 'LA', 'PA') else 0.0
        alpha_variance = 0.0
        if has_alpha and img.mode == 'RGBA':
            # Calculate variance in alpha channel
            img_array = np.array(img)
            if img_array.shape[-1] == 4:
                alpha_channel = img_array[:, :, 3].astype(float)
                alpha_variance = np.var(alpha_channel) / 65025.0  # Normalize by max variance
        
        is_jpeg = 1.0 if img.format == 'JPEG' else 0.0
        is_png = 1.0 if img.format == 'PNG' else 0.0
        
        # Return updated feature list (13 features)
        return [file_size_norm, exif_present, exif_length_norm, comment_length, 
                exif_entropy, palette_present, palette_length, 
                palette_entropy_value, eof_length_norm, has_alpha, 
                alpha_variance, is_jpeg, is_png]

# Function to collect data without undersampling
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
            
            # Collect ALL clean images
            if os.path.exists(clean_dir):
                for fname in os.listdir(clean_dir):
                    fpath = os.path.join(clean_dir, fname)
                    if os.path.isfile(fpath):
                        image_paths.append(fpath)
                        labels.append(0)
            
            # Collect all stego images
            if os.path.exists(stego_dir):
                for fname in os.listdir(stego_dir):
                    fpath = os.path.join(stego_dir, fname)
                    if os.path.isfile(fpath):
                        match = pattern.search(fname)
                        if match:
                            algo = match.group(1)
                            image_paths.append(fpath)
                            labels.append(class_map[algo])
                        else:
                            print(f"Invalid filename format: {fname}")
    
    if not image_paths:
        print("Warning: No valid images found in the dataset.")
    
    return image_paths, labels

# Model Definition
class Starlight(nn.Module):
    def __init__(self, num_classes=7, feature_dim=11):
        super(Starlight, self).__init__()
        kernel = torch.tensor([[[ -1.,  2., -1.],
                                [  2., -4.,  2.],
                                [ -1.,  2., -1.]]]).repeat(3, 1, 1, 1)
        self.srm_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        self.srm_conv.weight = nn.Parameter(kernel, requires_grad=False)
        
        resnet = models.resnet18(weights=None)
        self.pd_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Increased SF branch capacity for better metadata detection
        self.sf_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.fd_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Better balanced fusion with more capacity for SF branch
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, features):
        filtered = self.srm_conv(image)
        pd_feat = self.pd_backbone(filtered).flatten(1)
        sf_feat = self.sf_mlp(features)
        dct_img = self.dct2d(image)
        fd_feat = self.fd_cnn(dct_img)
        concatenated = torch.cat([pd_feat, sf_feat, fd_feat], dim=1)
        out = self.fusion(concatenated)
        return out
    
    def dct2d(self, x):
        def dct1d(y):
            N = y.size(-1)
            even = y[..., ::2]
            odd = y[..., 1::2].flip(-1)
            v = torch.cat([even, odd], dim=-1)
            Vc = torch.fft.fft(v, dim=-1)
            k = torch.arange(N, dtype=x.dtype, device=x.device) * (math.pi / (2 * N))
            W_r = torch.cos(k)
            W_i = torch.sin(k)
            V = 2 * (Vc.real * W_r - Vc.imag * W_i)
            return V
        dct_col = dct1d(x)
        dct_col = dct_col.transpose(2, 3)
        dct_row = dct1d(dct_col)
        dct_row = dct_row.transpose(2, 3)
        return dct_row

# Main Training Script
if __name__ == "__main__":
    # Collect all data without undersampling
    image_paths, labels = collect_data()
    if not image_paths:
        print("Error: No data found. Ensure dataset is populated and filenames follow the convention.")
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

    train_dataset = CustomDataset(train_paths, train_labels, transform=transform_train, augment=True)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform_val, augment=False)
    
    # Use simple random sampling - no weighted sampler
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=True)

    # Calculate class weights for loss function
    train_count = Counter(train_labels)
    num_classes = 7
    class_freq = np.array([train_count.get(i, 1) for i in range(num_classes)])
    
    # Square root of inverse frequency (gentlest effective method)
    class_weights = np.sqrt(class_freq.sum() / class_freq)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    print("\nClass weights (square root inverse frequency):")
    for algo, idx in class_map.items():
        print(f"  {algo}: {class_weights[idx]:.4f}")
    print(f"Weight ratio (max/min): {class_weights.max()/class_weights.min():.2f}x")

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    model = Starlight(num_classes=num_classes, feature_dim=13).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Use gentle class weights with minimal label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)

    # Use step decay instead of cosine annealing for more stability
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    patience = 15  # Increased patience
    best_val_loss = float('inf')
    best_balanced_f1 = 0.0  # Track best balanced F1
    counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/100 [Train]", unit="batch") as pbar:
            for images, features, labels_batch in train_loader:
                images, features, labels_batch = images.to(device), features.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = model(images, features)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()
                pbar.update(1)
                pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}", "Running Acc": f"{100. * train_correct / train_total:.2f}%"})

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        print(f"Epoch {epoch+1}/100, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_trues = []
        all_outputs = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/100 [Val]", unit="batch") as pbar:
                for images, features, labels_batch in val_loader:
                    images, features, labels_batch = images.to(device), features.to(device), labels_batch.to(device)
                    outputs = model(images, features)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
                    preds = outputs.argmax(1).cpu().tolist()
                    all_preds.extend(preds)
                    all_trues.extend(labels_batch.cpu().tolist())
                    all_outputs.append(outputs.cpu())
                    _, predicted = outputs.max(1)
                    val_total += labels_batch.size(0)
                    val_correct += predicted.eq(labels_batch).sum().item()
                    pbar.update(1)
                    pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}", "Running Acc": f"{100. * val_correct / val_total:.2f}%"})

        val_loss /= len(val_loader)
        all_outputs = torch.cat(all_outputs, dim=0)
        softmax_probs = F.softmax(all_outputs, dim=1).numpy()
        trues_np = np.array(all_trues)
        preds_np = np.array(all_preds)

        bal_acc = balanced_accuracy_score(trues_np, preds_np)
        f1 = f1_score(trues_np, preds_np, average='macro')
        per_class_f1 = f1_score(trues_np, preds_np, average=None)
        acc = accuracy_score(trues_np, preds_np)
        binary_true = (trues_np > 0).astype(int)
        stego_scores = 1 - softmax_probs[:, 0]
        auc = roc_auc_score(binary_true, stego_scores) if len(set(binary_true)) > 1 else 0.0

        print(f"Val Loss: {val_loss:.4f}, Balanced Acc: {bal_acc:.4f}, F1: {f1:.4f}, AUC (Clean vs Stego): {auc:.4f}, Top-1 Acc: {acc:.4f}")
        print(f"Per-class F1: {', '.join([f'{list(class_map.keys())[i]}: {v:.4f}' for i, v in enumerate(per_class_f1)])}")
        
        # Calculate per-class accuracy for better insight
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(trues_np, preds_np)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        print(f"Per-class Accuracy: {', '.join([f'{list(class_map.keys())[i]}: {v:.4f}' for i, v in enumerate(per_class_acc)])}")
        
        # Highlight minority class performance
        minority_classes = [1, 3]  # alpha, dct
        minority_acc = np.mean([per_class_acc[i] for i in minority_classes])
        print(f"Minority classes (alpha, dct) avg accuracy: {minority_acc:.4f}")
        
        # Calculate precision and recall for clean class specifically
        clean_precision = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() > 0 else 0.0
        clean_recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0.0
        print(f"Clean class - Precision: {clean_precision:.4f}, Recall: {clean_recall:.4f}")

        scheduler.step()
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save model based on balanced performance across ALL classes
        # Require: clean recall > 0.4 AND average minority class accuracy > 0.25
        clean_recall = per_class_acc[0]
        minority_classes = [1, 3]  # alpha, dct (hardest classes)
        minority_acc = np.mean([per_class_acc[i] for i in minority_classes])
        
        save_condition = val_loss < best_val_loss and clean_recall > 0.40 and minority_acc > 0.25
        
        if save_condition:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Saved best model - Val Loss: {val_loss:.4f}, Clean Recall: {clean_recall:.4f}, Minority Avg: {minority_acc:.4f}")
        else:
            counter += 1
            reasons = []
            if clean_recall <= 0.40:
                reasons.append(f"Clean recall too low ({clean_recall:.4f})")
            if minority_acc <= 0.25:
                reasons.append(f"Minority acc too low ({minority_acc:.4f})")
            if val_loss >= best_val_loss and clean_recall > 0.40 and minority_acc > 0.25:
                reasons.append(f"Val loss not improved ({val_loss:.4f} >= {best_val_loss:.4f})")
            
            if reasons:
                print(f"⚠ Not saving: {', '.join(reasons)}")
            
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break
