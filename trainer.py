#!/usr/bin/env python3
"""
Universal 6-class Stego Detector

- Hybrid model with parallel expert branches for different stego types.
- Separate branches for pixel analysis (LSB, Alpha, Palette) and metadata (EXIF/EOI).
- Palette features extracted from image palettes for palette-based steganography.
- AlphaDetector checks for AI24 marker in alpha LSB to classify as alpha steganography.
- LSBDetector handles RGB LSB and alpha LSB without AI24 marker.
- Batch normalization on all feature branches before fusion to prevent feature scaling issues.
- Robust data loading and conservative training hyperparameters to prevent overfitting.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        for subdir_pattern in subdirs:
            for matched_dir in self.base_dir.glob(subdir_pattern):
                for folder_type in ["clean", "stego"]:
                    folder = matched_dir / folder_type
                    if folder.exists():
                        for ext in ['*.png', '*.bmp', '*.jpg', '*.jpeg', '*.gif', '*.webp', '*.tiff', '*.tif']:
                            self.image_files.extend([(f, folder_type) for f in folder.glob(ext)])
        print(f"[DATASET] Found {len(self.image_files)} images from {subdirs}")

    def __len__(self): return len(self.image_files)

    def parse_label(self, filepath, folder_type):
        if folder_type == "clean": return ALGO_TO_ID["clean"]
        stem = filepath.stem.lower()
        for algo, idx in ALGO_TO_ID.items():
            if algo != "clean" and algo in stem: return idx
        return ALGO_TO_ID["clean"]

    def __getitem__(self, idx):
        img_path, folder_type = self.image_files[idx]
        try:
            img = Image.open(img_path)
            exif_features = get_exif_features(img, img_path)
            eoi_features = torch.tensor([get_eoi_payload_size(img_path) / 1000.0], dtype=torch.float)

            palette_tensor = torch.zeros(256, 3)
            if img.mode == 'P':
                palette = img.getpalette()
                if palette:
                    # Group into RGB triples and sort by luminance for consistency
                    colors = [(palette[i], palette[i+1], palette[i+2]) for i in range(0, len(palette), 3)]
                    colors.sort(key=lambda c: 0.299*c[0] + 0.587*c[1] + 0.114*c[2])  # Sort by luminance
                    sorted_palette = [val for color in colors for val in color]
                    palette_padded = (sorted_palette + [0] * (768 - len(sorted_palette)))[:768]
                    palette_array = np.array(palette_padded).reshape(256, 3) / 255.0
                    palette_tensor = torch.from_numpy(palette_array).float()

            if img.mode == 'RGBA':
                rgb, alpha_channel = img.convert('RGB'), np.array(img)[:, :, 3] / 255.0
                alpha = torch.from_numpy(alpha_channel).float().unsqueeze(0)
            else:
                rgb, alpha = img.convert('RGB'), torch.zeros(1, img.size[1], img.size[0])

            resize = transforms.Resize((224, 224), antialias=True)
            rgb_tensor = transforms.ToTensor()(resize(rgb))
            alpha_tensor = F.interpolate(alpha.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

            label = self.parse_label(img_path, folder_type)
            # Detect AI24 marker in alpha LSB
            alpha_lsb_bits = (alpha_tensor * 255).long() & 1
            marker_present = False
            bits = alpha_lsb_bits.flatten()
            if bits.numel() >= 32:
                bytes_ = []
                for b in range(4):
                    byte = 0
                    for i in range(8):
                        bit_idx = b * 8 + i
                        if bit_idx < bits.numel():
                            byte |= (bits[bit_idx].item() << (7 - i))
                    bytes_.append(byte)
                if bytes_ == [ord(c) for c in "AI24"]:
                    marker_present = True
            # Reclassify: if LSB in alpha with AI24 marker, classify as alpha; else if LSB in alpha without marker, keep as lsb
            if label == ALGO_TO_ID["lsb"] and alpha_lsb_bits.sum() > 0:
                if marker_present:
                    label = ALGO_TO_ID["alpha"]
            return rgb_tensor, alpha_tensor, exif_features, eoi_features, palette_tensor, label
        except Exception as e:
            print(f"ERROR loading {img_path}: {e}")
            return None

# --- DETECTOR MODULES ---
class LSBDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.lsb_conv = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU())
    def forward(self, rgb):
        lsb = (rgb * 255).long() & 1
        return F.adaptive_avg_pool2d(self.lsb_conv(lsb.float()), 1).flatten(1)

class AlphaDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.alpha_conv = nn.Sequential(nn.Conv2d(1, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU())
    def forward(self, alpha):
        alpha_lsb = (alpha * 255).long() & 1
        # Detect AI24 marker in first 32 LSB bits
        bits = alpha_lsb.flatten()
        marker_present = 0.0
        if bits.numel() >= 32:
            bytes_ = []
            for b in range(4):
                byte = 0
                for i in range(8):
                    bit_idx = b * 8 + i
                    if bit_idx < bits.numel():
                        byte |= (bits[bit_idx].item() << (7 - i))
                bytes_.append(byte)
            if bytes_ == [ord(c) for c in "AI24"]:
                marker_present = 1.0
        marker_tensor = torch.full((alpha.size(0), 1), marker_present, device=alpha.device)
        features = F.adaptive_avg_pool2d(self.alpha_conv(alpha_lsb.float()), 1).flatten(1)
        return torch.cat([features, marker_tensor], dim=1)

class PaletteDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.conv = nn.Conv1d(3, dim, 3, 1, 1)
        self.bn = nn.BatchNorm1d(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, palette):
        x = palette.permute(0, 2, 1)  # (batch, 3, 256)
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x).flatten(1)

class ExifEoiDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(3, dim), nn.ReLU())
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
        self.rgb_base = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(), nn.AdaptiveAvgPool2d(1))

        fusion_dim = dim + (dim + 1) + dim + dim + dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, alpha, exif, eoi, palette):
        f_lsb = self.lsb(rgb)
        f_alpha = self.alpha(alpha)
        f_meta = self.meta(exif, eoi)
        f_palette = self.palette(palette)
        f_rgb = self.rgb_base(rgb).flatten(1)
        
        features = torch.cat([f_lsb, f_alpha, f_meta, f_rgb, f_palette], dim=1)
        return self.classifier(features)

# --- TRAINING & UTILS ---
def collate_fn(batch):
    batch = list(filter(None, batch))
    return default_collate(batch) if batch else (None,)*6

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
            return True  # Improvement detected
        else:
            self.counter += 1
            if self.verbose: print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience: self.early_stop = True
            return False  # No improvement

def train_model(args):
    train_subdirs = args.train_subdirs.split(',') if args.train_subdirs else ["sample"]
    val_subdirs = args.val_subdirs.split(',') if args.val_subdirs else ["val"]
    train_ds = StegoImageDataset(args.datasets_dir, subdirs=train_subdirs)
    val_ds = StegoImageDataset(args.datasets_dir, subdirs=val_subdirs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=False)

    model = UniversalStegoDetector().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.model))
        print(f"[RESUME] Loaded model from {args.model}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=args.patience, monitor='acc')
    
    # Ensure the directory exists
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for rgb, alpha, exif, eoi, palette, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if rgb is None: continue
            rgb, alpha, exif, eoi, palette, labels = rgb.to(device), alpha.to(device), exif.to(device), eoi.to(device), palette.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rgb, alpha, exif, eoi, palette), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0, 0, 0
        class_correct = {i: 0 for i in range(NUM_CLASSES)}
        class_total = {i: 0 for i in range(NUM_CLASSES)}
        with torch.no_grad():
            for rgb, alpha, exif, eoi, palette, labels in tqdm(val_loader, desc="Validating"):
                if rgb is None: continue
                rgb, alpha, exif, eoi, palette, labels = rgb.to(device), alpha.to(device), exif.to(device), eoi.to(device), palette.to(device), labels.to(device)
                outputs = model(rgb, alpha, exif, eoi, palette)
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

        # Per-class accuracy
        print("  Per-algorithm accuracy:")
        id_to_algo = {v: k for k, v in ALGO_TO_ID.items()}
        for class_id, count in class_total.items():
            if count > 0:
                accuracy = 100 * class_correct[class_id] / count
                print(f"    - {id_to_algo[class_id]}: {accuracy:.2f}% ({class_correct[class_id]}/{count})")
        
        # Check for improvement and save if better
        improved = early_stopper(val_acc, model)
        if improved:
            torch.save(model.state_dict(), str(model_path))
            print(f"  [SAVED] Model improved! Saved to {model_path}")

        scheduler.step()

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Final save verification
    if model_path.exists():
        print(f"\n[FINAL] Best model saved to {model_path} ({model_path.stat().st_size} bytes)")
        metric_name = "accuracy" if early_stopper.monitor == 'acc' else "loss"
        print(f"[FINAL] Best validation {metric_name}: {early_stopper.best_value:.4f}")
    else:
        print(f"[ERROR] Model file was not saved!")


if __name__ == "__main__":
    print(f"[DEVICE] Using: {device}")
    parser = argparse.ArgumentParser(description="Starlight v3")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    parser.add_argument("--model", default="models/starlight.pth", help="Path to save/load the model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume training from the model path")
    parser.add_argument("--train_subdirs", type=str, default="sample", help="Comma-separated list of training subdir patterns (e.g., 'sample' or '*_submission_*')")
    parser.add_argument("--val_subdirs", type=str, default="val", help="Comma-separated list of validation subdir patterns")
    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    train_model(args)
