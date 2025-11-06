#!/usr/bin/env python3
"""
Training script for ChatGPT submission 2025 dataset.
Uses the UniversalStegoDetector architecture from trainer.py
"""

import os
from data_generator import main as generate_dataset

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
from torchvision.transforms import functional as TF
import random

# --- Model and dataset utilities (copied from trainer.py) ---
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from PIL import Image
import random
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, default_collate

# Device configuration (same logic as trainer)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Algorithm mappings
ALGO_TO_ID = {"alpha": 0, "palette": 1, "lsb": 2, "exif": 3, "eoi": 4, "clean": 5}
ID_TO_ALGO = {v: k for k, v in ALGO_TO_ID.items()}
NUM_CLASSES = 6

# Metadata extractors (same as trainer)
try:
    import piexif
except Exception:
    piexif = None

def get_eoi_payload_size(filepath):
    fp = str(filepath)
    if not fp.lower().endswith(('jpg', '.jpeg')):
        return 0
    try:
        with open(fp, 'rb') as f:
            data = f.read()
        eoi = data.rfind(b'\xff\xd9')
        if eoi > 0:
            return len(data) - (eoi + 2)
    except Exception:
        return 0
    return 0

def get_exif_features(img, filepath):
    exif_present = 0.0
    exif_len = 0.0
    fp = str(filepath)
    exif_bytes = img.info.get('exif')
    if exif_bytes:
        exif_present = 1.0
        exif_len = len(exif_bytes)
    elif piexif and fp.lower().endswith(('jpg', '.jpeg')):
        try:
            exif_dict = piexif.load(fp)
            if exif_dict and any(v for v in exif_dict.values() if v):
                exif_present = 1.0
                exif_len = len(piexif.dump(exif_dict))
        except Exception:
            pass
    return torch.tensor([exif_present, exif_len / 1000.0], dtype=torch.float)

# --- Detector modules (copy from trainer) ---
class LSBDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.trainable_conv = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, rgb):
        return self.trainable_conv(rgb).flatten(1)

class PaletteIndexDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, dim*2, 3, 1, 1), nn.BatchNorm2d(dim*2), nn.ReLU(),
            nn.Conv2d(dim*2, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, indices):
        indices_255 = (indices * 255).long()
        lsb = (indices_255 & 1).float()
        lsb_mean = lsb.mean(dim=[2,3], keepdim=True)
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
        batch, _, h, w = alpha.shape
        marker_feature = torch.zeros(batch, 1).to(alpha.device)
        if h >= 4 and w >= 8:
            alpha_int = (alpha[:,0,:4,:8] * 255).long()
            bits = alpha_int & 1
            byte0 = (bits[:,0,0] << 7) | (bits[:,0,1] << 6) | (bits[:,0,2] << 5) | (bits[:,0,3] << 4) | \
                    (bits[:,0,4] << 3) | (bits[:,0,5] << 2) | (bits[:,0,6] << 1) | bits[:,0,7]
            byte1 = (bits[:,1,0] << 7) | (bits[:,1,1] << 6) | (bits[:,1,2] << 5) | (bits[:,1,3] << 4) | \
                    (bits[:,1,4] << 3) | (bits[:,1,5] << 2) | (bits[:,1,6] << 1) | bits[:,1,7]
            byte2 = (bits[:,2,0] << 7) | (bits[:,2,1] << 6) | (bits[:,2,2] << 5) | (bits[:,2,3] << 4) | \
                    (bits[:,2,4] << 3) | (bits[:,2,5] << 2) | (bits[:,2,6] << 1) | bits[:,2,7]
            byte3 = (bits[:,3,0] << 7) | (bits[:,3,1] << 6) | (bits[:,3,2] << 5) | (bits[:,3,3] << 4) | \
                    (bits[:,3,4] << 3) | (bits[:,3,5] << 2) | (bits[:,3,6] << 1) | bits[:,3,7]
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
        x = palette.permute(0,2,1)
        return self.conv(x).flatten(1)

class ExifEoiDetector(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
    def forward(self, exif, eoi):
        return self.fc(torch.cat([exif, eoi], dim=1))

class UniversalStegoDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dim=64):
        super().__init__()
        self.lsb = LSBDetector(dim)
        self.alpha = AlphaDetector(dim)
        self.meta = ExifEoiDetector(dim)
        self.palette = PaletteDetector(dim)
        self.palette_index = PaletteIndexDetector(dim)
        fusion_dim = dim + (dim + 1) + dim + dim + dim
        self.feature_fusion = nn.Sequential(nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 128))
        self.classifier = nn.Linear(128, num_classes)
    def forward_features(self, rgb, alpha, exif, eoi, palette, indices):
        f_lsb = self.lsb(rgb)
        f_alpha = self.alpha(alpha)
        f_meta = self.meta(exif, eoi)
        f_palette = self.palette(palette)
        f_palette_index = self.palette_index(indices)
        combined = torch.cat([f_lsb, f_alpha, f_meta, f_palette, f_palette_index], dim=1)
        return self.feature_fusion(combined)
    def forward(self, rgb, alpha, exif, eoi, palette, indices):
        feats = self.forward_features(rgb, alpha, exif, eoi, palette, indices)
        return self.classifier(feats)

class PairedStegoDataset(Dataset):
    def __init__(self, base_dir, subdirs=None):
        self.base_dir = Path(base_dir)
        self.stego_pairs = []
        self.clean_files = []
        if subdirs is None:
            subdirs = ["*"]
        for subdir_pattern in subdirs:
            for matched_dir in self.base_dir.glob(subdir_pattern):
                clean_folder = matched_dir / "clean"
                stego_folder = matched_dir / "stego"
                if not (clean_folder.exists() and stego_folder.exists()):
                    continue
                for ext in ['*.png','*.bmp','*.jpg','*.jpeg','*.gif','*.webp','*.tiff','*.tif']:
                    self.clean_files.extend(clean_folder.glob(ext))
                for stego_file in stego_folder.glob("*.json"):
                    try:
                        with open(stego_file) as f:
                            meta = json.load(f)
                        clean_name = meta.get('clean_file')
                        if not clean_name:
                            continue
                        stego_path = stego_file.with_suffix('')
                        clean_path = clean_folder / clean_name
                        if stego_path.exists() and clean_path.exists():
                            self.stego_pairs.append({'stego': stego_path, 'clean': clean_path})
                    except Exception:
                        continue
    def __len__(self):
        return len(self.stego_pairs) * 2
    def preprocess_image(self, img_path):
        img = Image.open(img_path)
        exif_feat = get_exif_features(img, img_path)
        eoi_feat = torch.tensor([1.0 if get_eoi_payload_size(img_path) > 0 else 0.0], dtype=torch.float)
        crop = transforms.RandomCrop((224,224))
        to_tensor = transforms.ToTensor()
        rgb = torch.zeros(3,224,224)
        alpha = torch.zeros(1,224,224)
        palette = torch.zeros(256,3)
        indices = torch.zeros(1,224,224)
        def proc(pil, crop_tf, tensor_tf):
            if pil.size[0] < 224 or pil.size[1] < 224:
                pad_x = max(0, 224 - pil.size[0])
                pad_y = max(0, 224 - pil.size[1])
                pil = TF.pad(pil, [pad_x//2, pad_y//2, pad_x - pad_x//2, pad_y - pad_y//2])
            return tensor_tf(crop_tf(pil))
        if img.mode == 'P':
            pal = img.getpalette()
            if pal:
                pal = (pal + [0]* (768 - len(pal)))[:768]
                palette = torch.from_numpy(np.array(pal).reshape(256,3)/255.0).float()
            indices = proc(Image.fromarray(np.array(img)), crop, to_tensor)
        elif img.mode == 'RGBA':
            rgb = proc(img.convert('RGB'), crop, to_tensor)
            alpha = proc(Image.fromarray(np.array(img)[:,:,3]), crop, to_tensor)
        else:
            rgb = proc(img.convert('RGB'), crop, to_tensor)
        return rgb, alpha, exif_feat, eoi_feat, palette, indices
    def __getitem__(self, idx):
        actual = idx % len(self.stego_pairs)
        if idx < len(self.stego_pairs):
            pair = self.stego_pairs[actual]
            img1, img2 = pair['clean'], pair['stego']
            label_pair = torch.tensor(1.0, dtype=torch.float32)
            label1 = ALGO_TO_ID['clean']
            # infer label2 from json
            try:
                with open(pair['stego'].with_suffix(pair['stego'].suffix + '.json')) as f:
                    meta = json.load(f)
                tech = meta['embedding']['technique']
                if tech == 'alpha': label2 = ALGO_TO_ID['alpha']
                elif tech == 'palette': label2 = ALGO_TO_ID['palette']
                elif tech == 'lsb.rgb': label2 = ALGO_TO_ID['lsb']
                elif tech == 'exif': label2 = ALGO_TO_ID['exif']
                elif tech == 'raw': label2 = ALGO_TO_ID['eoi']
                else: label2 = ALGO_TO_ID['clean']
            except Exception:
                label2 = ALGO_TO_ID['clean']
        else:
            img1 = self.clean_files[actual % len(self.clean_files)]
            img2 = random.choice(self.clean_files)
            label_pair = torch.tensor(0.0, dtype=torch.float32)
            label1 = ALGO_TO_ID['clean']
            label2 = ALGO_TO_ID['clean']
        t1 = self.preprocess_image(img1)
        t2 = self.preprocess_image(img2)
        return t1, t2, label_pair, label1, label2

class SiameseStegoNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward_one(self, tensors):
        return self.base_model.forward_features(*tensors)
    def forward(self, tensors1, tensors2):
        f1 = self.forward_one(tensors1)
        f2 = self.forward_one(tensors2)
        return F.pairwise_distance(f1, f2), f1, f2

# End of copied utilities

def collate_pairs(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: 
        return None, None, None, None, None
    return default_collate(batch)

def train_model(args):
    """Train the UniversalStegoDetector on ChatGPT submission dataset"""
    
    # Generate dataset if needed (per proposal spec)
    if not os.path.exists("clean") or not os.path.exists("stego"):
        print("[DATASET] Generating dataset...")
        generate_dataset()
        print("[DATASET] Dataset generation complete.")
    
    # Create datasets using PairedStegoDataset
    train_ds = PairedStegoDataset('..', subdirs=['chatgpt_submission_2025'])
    val_ds = PairedStegoDataset('..', subdirs=['chatgpt_submission_2025'])
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Create data loaders with custom collate function for paired data
    
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
                
            tensors1, tensors2, label_pair, label1, label2 = batch_data
            # Move all tensors in the tuples to the device
            tensors1 = tuple(t.to(device) for t in tensors1)
            tensors2 = tuple(t.to(device) for t in tensors2)
            label_pair = label_pair.to(device)
            
            optimizer.zero_grad()
            distances, _, _ = model(tensors1, tensors2)
            loss = criterion(distances, label_pair)
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
                    
                tensors1, tensors2, label_pair, label1, label2 = batch_data
                tensors1 = tuple(t.to(device) for t in tensors1)
                tensors2 = tuple(t.to(device) for t in tensors2)
                label_pair = label_pair.to(device)

                distances, _, _ = model(tensors1, tensors2)
                val_loss += criterion(distances, label_pair).item()
                
                all_distances.extend(distances.cpu().numpy())
                all_labels.extend(label_pair.cpu().numpy())
                
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
                    
                tensors1, tensors2, label_pair, label1, label2 = batch_data
                tensors1 = tuple(t.to(device) for t in tensors1)
                tensors2 = tuple(t.to(device) for t in tensors2)
                
                distances, _, _ = model(tensors1, tensors2)
                all_distances.extend(distances.cpu().numpy())
                all_labels.extend(label_pair.numpy())
        
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
    parser.add_argument("--batch_size", type=int, default=32)
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