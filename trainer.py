#!/usr/bin/env python3
"""
Universal 6-class Stego Detector - Aligned with data_generator.py

Fixed Issues:
1. Alpha marker changed from "AI24" to "AI42" (matching generator)
2. Alpha marker detection uses MSB-first bit order (matching generator)
3. LSB detection no longer tries to reclassify based on alpha channel
4. Class weights for balanced training
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

try:
    import piexif
except ImportError:
    piexif = None

# --- CONFIGURATION ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ALGO_TO_ID = {
    "alpha": 0, "palette": 1, "lsb": 2,
    "exif": 3, "eoi": 4, "clean": 5,
}
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

# --- DATASET ---
class StegoImageDataset(Dataset):
    def __init__(self, base_dir, subdirs=None):
        self.base_dir = Path(base_dir)
        self.image_files = []
        if subdirs is None: subdirs = ["*"]
        
        print(f"[DATASET] Searching in {subdirs}...")
        for subdir_pattern in subdirs:
            for matched_dir in self.base_dir.glob(subdir_pattern):
                # Add clean files
                clean_folder = matched_dir / "clean"
                if clean_folder.exists():
                    for ext in ['*.png', '*.bmp', '*.jpg', '*.jpeg', '*.gif', '*.webp', '*.tiff', '*.tif']:
                        self.image_files.extend([(f, "clean") for f in clean_folder.glob(ext)])
                
                # Add stego files only if they have a .json sidecar
                stego_folder = matched_dir / "stego"
                if stego_folder.exists():
                    for ext in ['*.png', '*.bmp', '*.jpg', '*.jpeg', '*.gif', '*.webp', '*.tiff', '*.tif']:
                        for img_file in stego_folder.glob(ext):
                            if img_file.with_suffix(img_file.suffix + '.json').exists():
                                self.image_files.append((img_file, "stego"))

        print(f"[DATASET] Found {len(self.image_files)} images with valid metadata.")

    def __len__(self): return len(self.image_files)

    def get_label_from_json(self, img_path):
        json_path = img_path.with_suffix(img_path.suffix + '.json')
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            technique = metadata['embedding']['technique']
            
            if technique == 'alpha': return ALGO_TO_ID['alpha']
            elif technique == 'palette': return ALGO_TO_ID['palette']
            elif technique == 'lsb.rgb': return ALGO_TO_ID['lsb']
            elif technique == 'exif': return ALGO_TO_ID['exif']
            elif technique == 'raw': return ALGO_TO_ID['eoi']
            else: return None
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None

    def __getitem__(self, idx):
        img_path, folder_type = self.image_files[idx]
        try:
            img = Image.open(img_path)
            exif_features = get_exif_features(img, img_path)
            eoi_features = torch.tensor([1.0 if get_eoi_payload_size(img_path) > 0 else 0.0], dtype=torch.float)

            # Common transforms
            resize_transform = transforms.Resize((224, 224), antialias=True)
            tensor_transform = transforms.ToTensor()

            # Initialize all tensors to zeros
            rgb_tensor = torch.zeros(3, 224, 224)
            alpha_tensor = torch.zeros(1, 224, 224)
            palette_tensor = torch.zeros(256, 3)
            indices_tensor = torch.zeros(1, 224, 224)

            if img.mode == 'P':
                # Extract palette
                palette_data = img.getpalette()
                if palette_data:
                    palette_padded = (palette_data + [0] * (768 - len(palette_data)))[:768]
                    palette_array = np.array(palette_padded).reshape(256, 3) / 255.0
                    palette_tensor = torch.from_numpy(palette_array).float()
                
                # Extract indices, treat as a single-channel image
                indices_img = Image.fromarray(np.array(img))
                indices_resized_img = resize_transform(indices_img)
                indices_tensor = tensor_transform(indices_resized_img) # This will normalize to [0,1]

            elif img.mode == 'RGBA':
                rgb_pil = img.convert('RGB')
                rgb_tensor = tensor_transform(resize_transform(rgb_pil))
                
                # Extract alpha channel
                alpha_np = np.array(img)[:, :, 3]
                alpha_img = Image.fromarray(alpha_np)
                alpha_resized_img = resize_transform(alpha_img)
                alpha_tensor = tensor_transform(alpha_resized_img)

            else: # Grayscale, RGB, etc.
                rgb_pil = img.convert('RGB')
                rgb_tensor = tensor_transform(resize_transform(rgb_pil))

            if folder_type == 'stego':
                label = self.get_label_from_json(img_path)
                if label is None:
                    print(f"WARNING: Could not get label from JSON for {img_path}. Skipping.")
                    return None
            else: # clean
                label = ALGO_TO_ID['clean']
            
            return rgb_tensor, alpha_tensor, exif_features, eoi_features, palette_tensor, indices_tensor, label
        except Exception as e:
            print(f"ERROR loading {img_path}: {e}")
            return None

# --- DETECTOR MODULES ---
class LSBDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        
        # Define a fixed high-pass filter kernel (from SRM)
        srm_kernel = [[-1, 2, -2, 2, -1],
                      [2, -6, 8, -6, 2],
                      [-2, 8, -12, 8, -2],
                      [2, -6, 8, -6, 2],
                      [-1, 2, -2, 2, -1]]
        srm_kernel = torch.tensor(srm_kernel, dtype=torch.float32) / 12.0
        srm_kernel = srm_kernel.view(1, 1, 5, 5)
        
        # Create a non-trainable conv layer to apply this filter to each RGB channel
        self.srm_filter = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False, groups=3)
        self.srm_filter.weight.data = torch.cat([srm_kernel] * 3, dim=0)
        self.srm_filter.weight.requires_grad = False

        # The rest of the network is trainable and processes the noise residuals
        self.trainable_conv = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1), # Takes the 3-channel residual as input
            nn.BatchNorm2d(dim), 
            nn.Tanh(),
            nn.Conv2d(dim, dim, 3, 1, 1), 
            nn.BatchNorm2d(dim), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, rgb):
        # De-normalize the image tensor from [0, 1] to [0, 255] before filtering
        rgb_255 = rgb * 255.0

        # Pass the image through the fixed high-pass filter to get noise residuals
        residuals = self.srm_filter(rgb_255)
        
        # Pass residuals to the trainable part of the network
        return self.trainable_conv(residuals).flatten(1)

class PaletteIndexDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, indices):
        # indices are normalized to [0,1]. We need to scale them back to [0, 255] to get LSB.
        lsb = (indices * 255).long() & 1
        return self.conv(lsb.float()).flatten(1)

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
        self.rgb_base = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        fusion_dim = dim + (dim + 1) + dim + dim + dim + dim # lsb, alpha, meta, rgb, palette, palette_index
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, alpha, exif, eoi, palette, indices):
        f_lsb = self.lsb(rgb)
        f_alpha = self.alpha(alpha)
        f_meta = self.meta(exif, eoi)
        f_palette = self.palette(palette)
        f_rgb = self.rgb_base(rgb).flatten(1)
        f_palette_index = self.palette_index(indices)
        
        features = torch.cat([f_lsb, f_alpha, f_meta, f_rgb, f_palette, f_palette_index], dim=1)
        return self.classifier(features)

# --- TRAINING & UTILS ---
def collate_fn(batch):
    batch = list(filter(None, batch))
    return default_collate(batch) if batch else (None,)*7

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, verbose=True, monitor='loss'):
        self.patience, self.min_delta, self.verbose, self.monitor = patience, min_delta, verbose, monitor
        self.counter, self.best_value, self.early_stop = 0, -np.inf if monitor == 'acc' else np.inf, False
        self.best_model_state = None

    def __call__(self, value, model=None):
        improved = False
        if self.monitor == 'acc':
            if value - self.best_value > self.min_delta:
                self.best_value = value
                improved = True
        else:  # loss
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
    train_ds = StegoImageDataset(args.datasets_dir, subdirs=train_subdirs)
    val_ds = StegoImageDataset(args.datasets_dir, subdirs=val_subdirs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=False)

    # Calculate class weights from training data
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    for _, _, _, _, _, _, label in train_ds:
        if label is not None:
            class_counts[label] += 1
    
    total_samples = sum(class_counts.values())
    class_weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            class_weights[i] = total_samples / (NUM_CLASSES * class_counts[i])
        else:
            class_weights[i] = 1.0
    #class_weights[ALGO_TO_ID["lsb"]] = class_weights[ALGO_TO_ID["lsb"]] * 2.0  # Triple the weight for LSB class 
    id_to_algo = {v: k for k, v in ALGO_TO_ID.items()}
    weights_dict = {id_to_algo[i]: f"{class_weights[i]:.4f}" for i in range(NUM_CLASSES)}
    print(f"[CLASS WEIGHTS] {weights_dict}")
    class_weights = class_weights.to(device)

    model = UniversalStegoDetector().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.model))
        print(f"[RESUME] Loaded model from {args.model}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    early_stopper = EarlyStopping(patience=args.patience, monitor='acc')
    
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for rgb, alpha, exif, eoi, palette, indices, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if rgb is None: continue
            rgb, alpha, exif, eoi, palette, indices, labels = rgb.to(device), alpha.to(device), exif.to(device), eoi.to(device), palette.to(device), indices.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rgb, alpha, exif, eoi, palette, indices), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0, 0, 0
        class_correct = {i: 0 for i in range(NUM_CLASSES)}
        class_total = {i: 0 for i in range(NUM_CLASSES)}
        with torch.no_grad():
            for rgb, alpha, exif, eoi, palette, indices, labels in tqdm(val_loader, desc="Validating"):
                if rgb is None: continue
                rgb, alpha, exif, eoi, palette, indices, labels = rgb.to(device), alpha.to(device), exif.to(device), eoi.to(device), palette.to(device), indices.to(device), labels.to(device)
                outputs = model(rgb, alpha, exif, eoi, palette, indices)
                val_loss += criterion(outputs, labels).item()
                
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1
        
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        print("  Per-algorithm accuracy:")
        for class_id, count in class_total.items():
            if count > 0:
                accuracy = 100 * class_correct[class_id] / count
                print(f"    - {id_to_algo[class_id]}: {accuracy:.2f}% ({class_correct[class_id]}/{count})")
        
        improved = early_stopper(val_acc, model)
        if improved:
            torch.save(model.state_dict(), str(model_path))
            print(f"  [SAVED] Model improved! Saved to {model_path}")

        scheduler.step()

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    if model_path.exists():
        print(f"\n[FINAL] Best model saved to {model_path} ({model_path.stat().st_size} bytes)")
        metric_name = "accuracy" if early_stopper.monitor == 'acc' else "loss"
        print(f"[FINAL] Best validation {metric_name}: {early_stopper.best_value:.4f}")
    else:
        print(f"[ERROR] Model file was not saved!")


if __name__ == "__main__":
    print(f"[DEVICE] Using: {device}")
    parser = argparse.ArgumentParser(description="Starlight Trainer")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    parser.add_argument("--model", default="models/starlight.pth", help="Path to save/load the model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume training from the model path")
    parser.add_argument("--train_subdirs", type=str, default="sample_submission_2025", help="Comma-separated list of training subdir patterns")
    parser.add_argument("--val_subdirs", type=str, default="val", help="Comma-separated list of validation subdir patterns")
    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    train_model(args)
