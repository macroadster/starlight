#!/usr/bin/env python3
"""
Universal 6-class Stego Detector - Fixed Siamese Training

Key Fixes:
1. Properly combines contrastive loss with classification loss
2. Improved LSBDetector with residual normalization
3. Balanced class weights (can be adjusted)
4. Complete validation metrics for both contrastive and classification
5. Better monitoring of per-class accuracy
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

try:
    import piexif
except ImportError:
    piexif = None

# --- CONFIGURATION ---
# Prioritize CUDA > MPS > CPU for best performance
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[DEVICE] Using CUDA: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[DEVICE] Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("[DEVICE] Using CPU (no acceleration available)")

ALGO_TO_ID = {
    "alpha": 0, "palette": 1, "lsb": 2,
    "exif": 3, "eoi": 4, "clean": 5,
}
ID_TO_ALGO = {v: k for k, v in ALGO_TO_ID.items()}
NUM_CLASSES = 6

# --- METADATA FEATURE EXTRACTORS ---
def get_eoi_payload_size(filepath):
    filepath_str = str(filepath)
    if not filepath_str.lower().endswith(('.jpg', '.jpeg')):
        return 0
    try:
        with open(filepath_str, 'rb') as f:
            data = f.read()
        eoi_pos = data.rfind(b'\xff\xd9')
        if eoi_pos > 0:
            return len(data) - (eoi_pos + 2)
    except Exception:
        return 0
    return 0

def get_exif_features(img, filepath):
    exif_present = 0.0
    exif_len = 0.0
    filepath_str = str(filepath)
    exif_bytes = img.info.get('exif')
    if exif_bytes:
        exif_present = 1.0
        exif_len = len(exif_bytes)
    elif piexif and filepath_str.lower().endswith(('.jpg', '.jpeg')):
        try:
            exif_dict = piexif.load(filepath_str)
            if exif_dict and any(val for val in exif_dict.values() if val):
                exif_present = 1.0
                exif_len = len(piexif.dump(exif_dict))
        except Exception:
            pass
    return torch.tensor([exif_present, exif_len / 1000.0], dtype=torch.float)

# --- DETECTOR MODULES ---
class LSBDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.trainable_conv = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1), # Input is 3-channel RGB
            nn.BatchNorm2d(dim), 
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), 
            nn.BatchNorm2d(dim), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, rgb):
        # Pass the raw (cropped) RGB tensor directly
        return self.trainable_conv(rgb).flatten(1)

class PaletteIndexDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        # Enhanced network to better detect LSB patterns in palette indices
        self.conv = nn.Sequential(
            nn.Conv2d(1, dim*2, 3, 1, 1), 
            nn.BatchNorm2d(dim*2), 
            nn.ReLU(),
            nn.Conv2d(dim*2, dim, 3, 1, 1), 
            nn.BatchNorm2d(dim), 
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, indices):
        # Extract LSB from palette indices
        indices_255 = (indices * 255).long()
        lsb = (indices_255 & 1).float()
        
        # Also compute statistics of LSB distribution
        # Random LSB should be ~50% ones, stego LSB will have different statistics
        lsb_mean = lsb.mean(dim=[2, 3], keepdim=True)
        lsb_centered = lsb - lsb_mean
        
        return self.conv(lsb_centered).flatten(1)

class AlphaDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, alpha):
        alpha_lsb = (alpha * 255).long() & 1
        
        # Detect AI42 marker (MSB-first: 01000001 01001001 00110100 00110010)
        batch_size, _, h, w = alpha.shape
        marker_feature = torch.zeros(batch_size, 1).to(alpha.device)
        
        if h >= 4 and w >= 8:
            alpha_int = (alpha[:, 0, :4, :8] * 255).long()
            bits = alpha_int & 1
            # Check for AI42: A=65, I=73, 4=52, 2=50 in MSB-first order
            byte0 = (bits[:, 0, 0] << 7) | (bits[:, 0, 1] << 6) | (bits[:, 0, 2] << 5) | (bits[:, 0, 3] << 4) | \
                    (bits[:, 0, 4] << 3) | (bits[:, 0, 5] << 2) | (bits[:, 0, 6] << 1) | bits[:, 0, 7]
            byte1 = (bits[:, 1, 0] << 7) | (bits[:, 1, 1] << 6) | (bits[:, 1, 2] << 5) | (bits[:, 1, 3] << 4) | \
                    (bits[:, 1, 4] << 3) | (bits[:, 1, 5] << 2) | (bits[:, 1, 6] << 1) | bits[:, 1, 7]
            byte2 = (bits[:, 2, 0] << 7) | (bits[:, 2, 1] << 6) | (bits[:, 2, 2] << 5) | (bits[:, 2, 3] << 4) | \
                    (bits[:, 2, 4] << 3) | (bits[:, 2, 5] << 2) | (bits[:, 2, 6] << 1) | bits[:, 2, 7]
            byte3 = (bits[:, 3, 0] << 7) | (bits[:, 3, 1] << 6) | (bits[:, 3, 2] << 5) | (bits[:, 3, 3] << 4) | \
                    (bits[:, 3, 4] << 3) | (bits[:, 3, 5] << 2) | (bits[:, 3, 6] << 1) | bits[:, 3, 7]
            
            marker_match = (byte0 == 65) & (byte1 == 73) & (byte2 == 52) & (byte3 == 50)
            marker_feature = marker_match.float().unsqueeze(1)
        
        conv_features = self.alpha_conv(alpha_lsb.float()).flatten(1)
        return torch.cat([conv_features, marker_feature], dim=1)

class PaletteDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, dim, 3, 1, 1), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, dim, 3, 1, 1), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, palette):
        x = palette.permute(0, 2, 1)  # (batch, 3, 256)
        return self.conv(x).flatten(1)

class ExifEoiDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU()
        )
    def forward(self, exif, eoi):
        return self.fc(torch.cat([exif, eoi], dim=1))

# --- MAIN MODEL ---
class UniversalStegoDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dim=64):
        super().__init__()
        self.lsb = LSBDetector(dim)
        self.alpha = AlphaDetector(dim)
        self.meta = ExifEoiDetector(dim)
        self.palette = PaletteDetector(dim)
        self.palette_index = PaletteIndexDetector(dim)
        # self.rgb_base = nn.Sequential(
        #     nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
        #     nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1)
        # )

        fusion_dim = dim + (dim + 1) + dim + dim + dim # No f_rgb
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward_features(self, rgb, alpha, exif, eoi, palette, indices):
        f_lsb = self.lsb(rgb)
        f_alpha = self.alpha(alpha)
        f_meta = self.meta(exif, eoi)
        f_palette = self.palette(palette)
        # f_rgb = self.rgb_base(rgb).flatten(1)
        f_palette_index = self.palette_index(indices)
        
        combined_features = torch.cat([f_lsb, f_alpha, f_meta, f_palette, f_palette_index], dim=1)
        return self.feature_fusion(combined_features)

    def forward(self, rgb, alpha, exif, eoi, palette, indices):
        features = self.forward_features(rgb, alpha, exif, eoi, palette, indices)
        return self.classifier(features)


class ContrastiveLoss(nn.Module):
    """Contrastive loss function."""
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # label=1 for different (stego vs clean), label=0 for same (clean vs clean)
        loss_different = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss_same = (1 - label) * torch.pow(distance, 2)
        return torch.mean(loss_different + loss_same)


class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance by down-weighting easy examples."""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class PairedStegoDataset(Dataset):
    def __init__(self, base_dir, subdirs=None):
        self.base_dir = Path(base_dir)
        self.stego_pairs = []
        self.clean_files = []

        if subdirs is None: subdirs = ["*"]
        print(f"[DATASET] Searching for pairs in {subdirs}...")

        for subdir_pattern in subdirs:
            for matched_dir in self.base_dir.glob(subdir_pattern):
                clean_folder = matched_dir / "clean"
                stego_folder = matched_dir / "stego"

                if not (clean_folder.exists() and stego_folder.exists()):
                    continue

                for ext in ['*.png', '*.bmp', '*.jpg', '*.jpeg', '*.gif', '*.webp', '*.tiff', '*.tif']:
                    self.clean_files.extend(clean_folder.glob(ext))

                for stego_file in stego_folder.glob("*.json"):
                    try:
                        with open(stego_file, 'r') as f:
                            metadata = json.load(f)
                        
                        clean_filename = metadata.get('clean_file')
                        if not clean_filename:
                            continue

                        actual_stego_file = stego_file.with_suffix('')
                        clean_file_path = clean_folder / clean_filename
                        
                        if actual_stego_file.exists() and clean_file_path.exists():
                            self.stego_pairs.append({'stego': actual_stego_file, 'clean': clean_file_path})
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        print(f"[DATASET] Found {len(self.stego_pairs)} stego/clean pairs via JSON metadata.")
        print(f"[DATASET] Found {len(self.clean_files)} total clean files for negative pairing.")

    def __len__(self):
        return len(self.stego_pairs) * 2

    def preprocess_image(self, img_path):
        img = Image.open(img_path)
        exif_features = get_exif_features(img, img_path)
        eoi_features = torch.tensor([1.0 if get_eoi_payload_size(img_path) > 0 else 0.0], dtype=torch.float)

        crop_transform = transforms.RandomCrop((224, 224))
        tensor_transform = transforms.ToTensor()

        rgb_tensor = torch.zeros(3, 224, 224)
        alpha_tensor = torch.zeros(1, 224, 224)
        palette_tensor = torch.zeros(256, 3)
        indices_tensor = torch.zeros(1, 224, 224)

        def process_and_crop(img_pil, crop_transform, tensor_transform):
            if img_pil.size[0] < 224 or img_pil.size[1] < 224:
                padding_x = max(0, 224 - img_pil.size[0])
                padding_y = max(0, 224 - img_pil.size[1])
                img_pil = transforms.functional.pad(img_pil, (padding_x // 2, padding_y // 2, padding_x - padding_x // 2, padding_y - padding_y // 2))
            return tensor_transform(crop_transform(img_pil))

        if img.mode == 'P':
            palette_data = img.getpalette()
            if palette_data:
                palette_padded = (palette_data + [0] * (768 - len(palette_data)))[:768]
                palette_array = np.array(palette_padded).reshape(256, 3) / 255.0
                palette_tensor = torch.from_numpy(palette_array).float()
            
            indices_img = Image.fromarray(np.array(img))
            indices_tensor = process_and_crop(indices_img, crop_transform, tensor_transform)

        elif img.mode == 'RGBA':
            rgb_pil = img.convert('RGB')
            rgb_tensor = process_and_crop(rgb_pil, crop_transform, tensor_transform)
            
            alpha_np = np.array(img)[:, :, 3]
            alpha_img = Image.fromarray(alpha_np)
            alpha_tensor = process_and_crop(alpha_img, crop_transform, tensor_transform)

        else:
            rgb_pil = img.convert('RGB')
            rgb_tensor = process_and_crop(rgb_pil, crop_transform, tensor_transform)
        
        return rgb_tensor, alpha_tensor, exif_features, eoi_features, palette_tensor, indices_tensor

    def __getitem__(self, idx):
        actual_idx = idx % len(self.stego_pairs)

        if idx < len(self.stego_pairs):
            pair = self.stego_pairs[actual_idx]
            img1_path, img2_path = pair['clean'], pair['stego']
            label_pair = torch.tensor(1.0, dtype=torch.float32)
            label1 = ALGO_TO_ID['clean']
            
            json_path = img2_path.with_suffix(img2_path.suffix + '.json')
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                technique = metadata['embedding']['technique']
                if technique == 'alpha': label2 = ALGO_TO_ID['alpha']
                elif technique == 'palette': label2 = ALGO_TO_ID['palette']
                elif technique == 'lsb.rgb': label2 = ALGO_TO_ID['lsb']
                elif technique == 'exif': label2 = ALGO_TO_ID['exif']
                elif technique == 'raw': label2 = ALGO_TO_ID['eoi']
                else: label2 = ALGO_TO_ID['clean']
            except:
                label2 = ALGO_TO_ID['clean']
        else:
            img1_path = self.clean_files[actual_idx % len(self.clean_files)]
            img2_path = random.choice(self.clean_files)
            label_pair = torch.tensor(0.0, dtype=torch.float32)
            label1 = ALGO_TO_ID['clean']
            label2 = ALGO_TO_ID['clean']

        try:
            tensors1 = self.preprocess_image(img1_path)
            tensors2 = self.preprocess_image(img2_path)
            return tensors1, tensors2, label_pair, label1, label2
        except Exception as e:
            print(f"ERROR loading pair for {img1_path} or {img2_path}: {e}")
            return None, None, None, None, None


class SiameseStegoNet(nn.Module):
    def __init__(self, base_model):
        super(SiameseStegoNet, self).__init__()
        self.base_model = base_model

    def forward_one(self, tensors):
        return self.base_model.forward_features(*tensors)

    def forward(self, tensors1, tensors2):
        features1 = self.forward_one(tensors1)
        features2 = self.forward_one(tensors2)
        distance = F.pairwise_distance(features1, features2)
        return distance, features1, features2

# --- TRAINING & UTILS ---
def collate_pairs(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return None, None, None, None, None
    return default_collate(batch)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, verbose=True, monitor='acc'):
        self.patience, self.min_delta, self.verbose, self.monitor = patience, min_delta, verbose, monitor
        self.counter, self.best_value, self.early_stop = 0, -np.inf if monitor == 'acc' else np.inf, False
        self.best_model_state = None

    def __call__(self, value, model=None):
        improved = False
        if self.monitor == 'acc':
            if value - self.best_value > self.min_delta:
                self.best_value = value
                improved = True
        else:
            if self.best_value - value > self.min_delta:
                self.best_value = value
                improved = True

        if improved:
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict().copy()
            return True
        else:
            self.counter += 1
            if self.verbose: print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience: self.early_stop = True
            return False

def train_model(args):
    train_subdirs = args.train_subdirs.split(',') if args.train_subdirs else ["sample"]
    val_subdirs = args.val_subdirs.split(',') if args.val_subdirs else ["val"]
    
    train_ds = PairedStegoDataset(args.datasets_dir, subdirs=train_subdirs)
    val_ds = PairedStegoDataset(args.datasets_dir, subdirs=val_subdirs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_pairs, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_pairs, pin_memory=False)

    base_model = UniversalStegoDetector().to(device)
    model = SiameseStegoNet(base_model).to(device)
    
    # Allow custom class weights via command line
    if args.class_weights:
        weights = [float(w) for w in args.class_weights.split(',')]
        if len(weights) != 6:
            raise ValueError("class_weights must have 6 values: alpha,palette,lsb,exif,eoi,clean")
        class_weights = torch.tensor(weights).to(device)
        print(f"[TRAINING] Using CUSTOM class weights: {class_weights.tolist()}")
    else:
        # BALANCED DEFAULT: Moderate weights that don't completely ignore clean
        class_weights = torch.tensor([2.0, 3.0, 3.0, 2.0, 2.0, 0.5]).to(device)
        #                              alpha palette lsb  exif  eoi  clean
        # Palette/LSB get 3.0x (hard classes), clean gets 0.5x (abundant class)
        print(f"[TRAINING] Using DEFAULT class weights: {class_weights.tolist()}")

    if args.resume:
        try:
            model.load_state_dict(torch.load(args.model))
            print(f"[RESUME] Loaded Siamese model from {args.model}")
        except RuntimeError:
            base_model.load_state_dict(torch.load(args.model))
            print(f"[RESUME] Loaded base model weights into Siamese network from {args.model}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    contrastive_criterion = ContrastiveLoss(margin=args.margin)
    
    # CRITICAL: Use Focal Loss instead of standard cross-entropy for better class imbalance handling
    if args.use_focal_loss:
        classification_criterion = FocalLoss(alpha=class_weights, gamma=args.focal_loss_gamma)
        print("[TRAINING] Using Focal Loss for classification")
    else:
        classification_criterion = lambda logits, labels: F.cross_entropy(logits, labels, weight=class_weights)
        print("[TRAINING] Using weighted Cross-Entropy for classification")
    
    early_stopper = EarlyStopping(patience=args.patience, monitor='acc')
    
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_contrastive_loss = 0
        train_class_loss = 0
        
        for tensors1, tensors2, label_pair, label1, label2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if tensors1 is None: continue

            tensors1 = tuple(t.to(device) for t in tensors1)
            tensors2 = tuple(t.to(device) for t in tensors2)
            label_pair = label_pair.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)

            optimizer.zero_grad()
            
            # FIXED: Get both distance and features
            distance, features1, features2 = model(tensors1, tensors2)

            # FIXED: Compute BOTH contrastive and classification losses
            contrastive_loss = contrastive_criterion(distance, label_pair)
            
            logits1 = model.base_model.classifier(features1)
            logits2 = model.base_model.classifier(features2)
            class_loss1 = classification_criterion(logits1, label1)
            class_loss2 = classification_criterion(logits2, label2)
            class_loss = (class_loss1 + class_loss2) / 2

            # CRITICAL FIX: Reduce contrastive loss weight significantly
            # The classification task is more important than pair similarity
            loss = args.contrastive_weight * contrastive_loss + args.classification_weight * class_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_contrastive_loss += contrastive_loss.item()
            train_class_loss += class_loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_contrastive_loss = 0
        val_class_loss = 0
        
        # For contrastive accuracy
        correct_pairs = 0
        total_pairs = 0
        
        # For classification accuracy
        all_pred_classes = []
        all_true_classes = []
        
        # Per-class accuracy tracking
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        
        with torch.no_grad():
            for tensors1, tensors2, label_pair, label1, label2 in tqdm(val_loader, desc="Validating"):
                if tensors1 is None: continue
                
                tensors1 = tuple(t.to(device) for t in tensors1)
                tensors2 = tuple(t.to(device) for t in tensors2)
                label_pair = label_pair.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)

                distance, features1, features2 = model(tensors1, tensors2)

                # Contrastive loss
                contrastive_loss = contrastive_criterion(distance, label_pair)
                val_contrastive_loss += contrastive_loss.item()
                
                # Classification loss
                logits1 = model.base_model.classifier(features1)
                logits2 = model.base_model.classifier(features2)
                class_loss1 = classification_criterion(logits1, label1)
                class_loss2 = classification_criterion(logits2, label2)
                class_loss = (class_loss1 + class_loss2) / 2
                val_class_loss += class_loss.item()

                val_loss += (args.contrastive_weight * contrastive_loss + args.classification_weight * class_loss).item()

                # FIXED: Contrastive accuracy (threshold at margin/2)
                threshold = args.margin / 2.0
                pred_different = (distance > threshold).float()
                correct_pairs += (pred_different == label_pair).sum().item()
                total_pairs += label_pair.size(0)

                # Classification accuracy
                pred1 = torch.argmax(logits1, dim=1)
                pred2 = torch.argmax(logits2, dim=1)
                
                all_pred_classes.extend(pred1.cpu().numpy())
                all_pred_classes.extend(pred2.cpu().numpy())
                all_true_classes.extend(label1.cpu().numpy())
                all_true_classes.extend(label2.cpu().numpy())
                
                # Per-class tracking
                for pred, true in zip(pred1.cpu().numpy(), label1.cpu().numpy()):
                    class_total[true] += 1
                    if pred == true:
                        class_correct[true] += 1
                for pred, true in zip(pred2.cpu().numpy(), label2.cpu().numpy()):
                    class_total[true] += 1
                    if pred == true:
                        class_correct[true] += 1

        val_loss /= len(val_loader)
        val_contrastive_loss /= len(val_loader)
        val_class_loss /= len(val_loader)

        # FIXED: Compute both accuracies
        contrastive_acc = (correct_pairs / total_pairs * 100) if total_pairs > 0 else 0
        class_acc = np.mean(np.array(all_pred_classes) == np.array(all_true_classes)) * 100
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} (Contrastive: {train_contrastive_loss/len(train_loader):.4f}, Class: {train_class_loss/len(train_loader):.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Contrastive: {val_contrastive_loss:.4f}, Class: {val_class_loss:.4f})")
        print(f"  Contrastive Acc: {contrastive_acc:.2f}% | Classification Acc: {class_acc:.2f}%")
        
        # FIXED: Per-class accuracy breakdown
        print(f"  Per-class accuracy:")
        for class_id, class_name in ID_TO_ALGO.items():
            if class_total[class_id] > 0:
                acc = class_correct[class_id] / class_total[class_id] * 100
                print(f"    {class_name}: {acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})")

        # Use classification accuracy for early stopping
        improved = early_stopper(class_acc, model)
        if improved:
            torch.save(model.base_model.state_dict(), str(model_path))
            print(f"  [SAVED] Model improved! Base model weights saved to {model_path}")

        scheduler.step()

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    if model_path.exists():
        print(f"\n[FINAL] Best model saved to {model_path} ({model_path.stat().st_size} bytes)")
        print(f"[FINAL] Best validation classification accuracy: {early_stopper.best_value:.4f}")
    else:
        print(f"[ERROR] Model file was not saved!")


def orchestrate_training(args):
    datasets_dir = Path(args.datasets_dir)
    submissions = [d for d in datasets_dir.iterdir() if d.is_dir() and d.name.endswith('_submission_2025') and d.name not in ['sample_submission_2025', 'maya_submission_2025']]

    for submission in submissions:
        print(f"[ORCHESTRATE] Processing {submission.name}")

        # Clean clean and stego directories
        clean_dir = submission / "clean"
        stego_dir = submission / "stego"
        if clean_dir.exists():
            import shutil
            shutil.rmtree(clean_dir)
        if stego_dir.exists():
            import shutil
            shutil.rmtree(stego_dir)
        print(f"[ORCHESTRATE] Cleaned directories for {submission.name}")

        # Run data_generator.py --limit
        cmd_data = f"cd {submission} && python data_generator.py --limit {args.limit}"
        print(f"[ORCHESTRATE] Running: {cmd_data}")
        os.system(cmd_data)

        # Run train.py --epochs
        cmd_train = f"cd {submission} && python train.py --epochs {args.epoch}"
        print(f"[ORCHESTRATE] Running: {cmd_train}")
        os.system(cmd_train)

if __name__ == "__main__":
    print(f"[DEVICE] Using: {device}")
    parser = argparse.ArgumentParser(description="Starlight Trainer - Orchestrate training for all submissions")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--limit", type=int, default=5, help="Limit for data generation")
    args = parser.parse_args()
    orchestrate_training(args)
