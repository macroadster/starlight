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

# Define transforms with RGBA support
transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
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
        img = Image.open(path)
        
        # Extract features BEFORE converting
        features = self.extract_features(path, img, label)
        
        # Convert to RGBA
        img_rgba = img.convert('RGBA')
        
        if self.transform:
            img_rgba = self.transform(img_rgba)
        
        return img_rgba, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def extract_features(self, path, img, label):
        width, height = img.size
        area = width * height if width * height > 0 else 1.0
        
        file_size = os.path.getsize(path) / 1024.0
        file_size_norm = file_size / area
        
        exif_bytes = img.info.get('exif')
        exif_present = 1.0 if exif_bytes else 0.0
        exif_length = len(exif_bytes) if exif_bytes else 0.0
        exif_length_norm = min(exif_length / area, 1.0)
        
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
                pass
        
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
        
        has_alpha = 1.0 if img.mode in ('RGBA', 'LA', 'PA') else 0.0
        alpha_variance = 0.0
        alpha_mean = 0.5
        alpha_unique_ratio = 0.0
        
        if has_alpha and img.mode == 'RGBA':
            img_array = np.array(img)
            if img_array.shape[-1] == 4:
                alpha_channel = img_array[:, :, 3].astype(float)
                alpha_variance = np.var(alpha_channel) / 65025.0
                alpha_mean = np.mean(alpha_channel) / 255.0
                unique_alphas = len(np.unique(alpha_channel))
                total_pixels = alpha_channel.size
                alpha_unique_ratio = unique_alphas / min(total_pixels, 256)
        
        is_jpeg = 1.0 if img.format == 'JPEG' else 0.0
        is_png = 1.0 if img.format == 'PNG' else 0.0
        
        return [file_size_norm, exif_present, exif_length_norm, comment_length, 
                exif_entropy, palette_present, palette_length, 
                palette_entropy_value, eof_length_norm, has_alpha, 
                alpha_variance, alpha_mean, alpha_unique_ratio, is_jpeg, is_png]

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

# TWO-STAGE ARCHITECTURE: Anomaly Detection + Type Classification
class StarlightTwoStage(nn.Module):
    def __init__(self, num_stego_classes=6, feature_dim=15):
        super(StarlightTwoStage, self).__init__()
        
        # ============== STAGE 1: ANOMALY DETECTOR ==============
        # Learns what "normal" images look like
        # Output: Single score indicating "how normal" the image is
        
        # Statistical Feature Analyzer (primary for anomaly detection)
        self.anomaly_sf_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Pixel-level anomaly detection (lightweight)
        self.anomaly_pixel_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Anomaly score fusion
        self.anomaly_fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single normality score
            nn.Sigmoid()
        )
        
        # ============== STAGE 2: STEGO TYPE CLASSIFIER ==============
        # Only activated if Stage 1 detects anomaly
        # Determines WHICH type of stego
        
        # SRM filter for stego type detection
        kernel = torch.tensor([[[ -1.,  2., -1.],
                                [  2., -4.,  2.],
                                [ -1.,  2., -1.]]])
        kernel_rgba = kernel.repeat(4, 1, 1, 1)
        self.srm_conv = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=False)
        self.srm_conv.weight = nn.Parameter(kernel_rgba, requires_grad=False)
        
        # Pixel Domain branch for type classification
        resnet = models.resnet18(weights=None)
        self.rgba_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            nn.init.kaiming_normal_(self.rgba_conv1.weight, mode='fan_out', nonlinearity='relu')
        resnet.conv1 = self.rgba_conv1
        self.pd_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # SF branch for type classification
        self.type_sf_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Frequency Domain branch
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
        
        # Type classification fusion
        self.type_fusion = nn.Sequential(
            nn.Linear(512 + 256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_stego_classes)  # 6 stego types (no clean)
        )

    def forward(self, image, features, return_stage1=False):
        # ============== STAGE 1: Anomaly Detection ==============
        sf_anomaly = self.anomaly_sf_branch(features)
        pixel_anomaly = self.anomaly_pixel_branch(image)
        
        anomaly_features = torch.cat([sf_anomaly, pixel_anomaly], dim=1)
        normality_score = self.anomaly_fusion(anomaly_features)  # [batch, 1], 1=normal, 0=anomalous
        
        if return_stage1:
            return normality_score
        
        # ============== STAGE 2: Stego Type Classification ==============
        # Compute features for type classification
        filtered = self.srm_conv(image)
        pd_feat = self.pd_backbone(filtered).flatten(1)
        
        sf_feat = self.type_sf_branch(features)
        
        image_rgb = image[:, :3, :, :]
        dct_img = self.dct2d(image_rgb)
        fd_feat = self.fd_cnn(dct_img)
        
        concatenated = torch.cat([pd_feat, sf_feat, fd_feat], dim=1)
        stego_type_logits = self.type_fusion(concatenated)  # [batch, 6]
        
        # IMPROVED COMBINATION: Use logits, not weighted probabilities
        # This allows Stage 2 to override Stage 1 for subtle stego
        
        # Convert normality score to logit scale
        eps = 1e-7
        normality_logit = torch.log(normality_score + eps) - torch.log(1 - normality_score + eps)
        
        # Combine as: [clean_logit, stego_type_logits]
        # Use temperature scaling to balance - higher temp = more balanced
        temperature = 1.5
        clean_logit = normality_logit * temperature
        
        # Stack all logits
        all_logits = torch.cat([clean_logit, stego_type_logits], dim=1)  # [batch, 7]
        
        return all_logits, normality_score, stego_type_logits
    
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    patience = 15
    best_val_loss = float('inf')
    best_clean_recall = 0.0
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
                
                # Forward pass
                outputs, normality_score, stego_type_logits = model(images, features)
                
                # Three-part loss with proper weighting
                is_clean = (labels_batch == 0).float().unsqueeze(1)
                
                # Stage 1: STRONG clean vs stego signal (most important!)
                stage1_loss = F.binary_cross_entropy(normality_score, is_clean)
                
                # Stage 2: Stego type classification (only for stego samples)
                stego_mask = (labels_batch > 0)
                if stego_mask.any():
                    stego_labels = labels_batch[stego_mask] - 1
                    stego_logits = stego_type_logits[stego_mask]
                    stage2_loss = F.cross_entropy(stego_logits, stego_labels)
                else:
                    stage2_loss = 0.0
                
                # Overall classification loss
                overall_loss = F.cross_entropy(outputs, labels_batch)
                
                # Rebalanced: Stage 1 gets highest weight to learn anomaly detection properly
                loss = 1.0 * stage1_loss + 0.7 * stage2_loss + 0.3 * overall_loss
                
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
                        stage2_loss = F.cross_entropy(stego_logits, stego_labels)
                    else:
                        stage2_loss = 0.0
                    
                    overall_loss = F.cross_entropy(outputs, labels_batch)
                    
                    loss = 1.0 * stage1_loss + 0.7 * stage2_loss + 0.3 * overall_loss
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
        binary_pred = (preds_np > 0).astype(int)
        clean_recall = np.sum((trues_np == 0) & (preds_np == 0)) / np.sum(trues_np == 0)
        clean_precision = np.sum((trues_np == 0) & (preds_np == 0)) / max(np.sum(preds_np == 0), 1)
        stego_recall = np.sum((trues_np > 0) & (preds_np > 0)) / max(np.sum(trues_np > 0), 1)
        
        # AUC using normality scores
        auc = roc_auc_score(1 - binary_true, all_normality_scores) if len(set(binary_true)) > 1 else 0.0

        print(f"Val Loss: {val_loss:.4f}, Balanced Acc: {bal_acc:.4f}, F1: {f1:.4f}, Top-1 Acc: {acc:.4f}")
        print(f"Clean: Recall={clean_recall:.4f}, Precision={clean_precision:.4f}")
        print(f"Stego: Recall={stego_recall:.4f}, AUC={auc:.4f}")
        print(f"Per-class F1: {', '.join([f'{list(class_map.keys())[i]}: {v:.4f}' for i, v in enumerate(per_class_f1)])}")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(trues_np, preds_np)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        print(f"Per-class Accuracy: {', '.join([f'{list(class_map.keys())[i]}: {v:.4f}' for i, v in enumerate(per_class_acc)])}")

        scheduler.step()

        # Save condition: balanced requirements
        min_stego_acc = np.min([per_class_acc[i] for i in range(1, 7)])
        save_condition = (clean_recall > 0.60 and  # More realistic
                         stego_recall > 0.80 and 
                         min_stego_acc > 0.50 and 
                         bal_acc > 0.75 and  # Add balanced accuracy requirement
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
            if clean_recall <= 0.60:
                reasons.append(f"Clean recall: {clean_recall:.4f} ≤ 0.60")
            if stego_recall <= 0.80:
                reasons.append(f"Stego recall: {stego_recall:.4f} ≤ 0.80")
            if min_stego_acc <= 0.50:
                reasons.append(f"Min stego acc: {min_stego_acc:.4f} ≤ 0.50")
            if bal_acc <= 0.75:
                reasons.append(f"Balanced acc: {bal_acc:.4f} ≤ 0.75")
            if val_loss >= best_val_loss and clean_recall > 0.60:
                reasons.append(f"Val loss not improved")
            
            if reasons:
                print(f"⚠ Not saving: {', '.join(reasons)}")
            
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best clean recall: {best_clean_recall:.4f}")
