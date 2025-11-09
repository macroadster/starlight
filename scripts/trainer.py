import json
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import struct
import os
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from collections import Counter
from pathlib import Path
import copy
from starlight_utils import extract_post_tail, load_multi_input

class StarlightDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Metadata MLP (for EXIF, EOI data)
        self.meta_mlp = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 64))

        # 2. Alpha Channel CNN
        self.alpha_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # 3. LSB Channel CNN
        self.lsb_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 4. Palette MLP
        self.palette_mlp = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 32))

        # Fused dimension: 64 (meta) + 32 (alpha) + 32 (lsb) + 32 (palette) = 160
        fused_dim = 64 + 32 + 32 + 32

        # Method classifier (5 methods)
        self.method_head = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 5))

        # Final stego head (outputs logits)
        self.stego_head = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, meta, alpha, lsb, palette):
        m = self.meta_mlp(meta)
        a = self.alpha_cnn(alpha).flatten(1)
        l = self.lsb_cnn(lsb).flatten(1)
        p = self.palette_mlp(palette)

        fused = torch.cat([m, a, l, p], dim=1)

        method_logits = self.method_head(fused)
        stego_logits = self.stego_head(fused).squeeze(1)
        
        method_probs = torch.softmax(method_logits, dim=1)
        method_id = method_probs.argmax(1)
        return stego_logits, method_logits, method_id, method_probs

class StegoDataset(Dataset):
    def __init__(self, clean_dir, stego_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.method_map = {
            "alpha": 0, "palette": 1, "lsb.rgb": 2,
            "exif": 3, "raw": 4 # EOI is labeled as 'raw' by the generator
        }
        
        stego_path_obj = Path(stego_dir)
        clean_path_obj = Path(clean_dir)
        
        for stego_filepath in stego_path_obj.iterdir():
            if not stego_filepath.suffix.lower().endswith(('.jpg', '.png', '.gif', '.webp', '.bmp', '.jpeg')):
                continue
            json_path = stego_filepath.with_suffix(stego_filepath.suffix + '.json')
            if not json_path.exists():
                continue
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                technique = metadata.get('embedding', {}).get('technique')
                clean_filename = metadata.get('clean_file')
                if not (technique and clean_filename):
                    continue
                clean_filepath = clean_path_obj / clean_filename
                if not clean_filepath.exists():
                    continue
                if technique in self.method_map:
                    method_id = self.method_map[technique]
                    self.samples.append((str(stego_filepath), 1, method_id))
                    self.samples.append((str(clean_filepath), 0, -1))
            except (json.JSONDecodeError, KeyError):
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, stego_label, method_label = self.samples[idx]
        meta, alpha, lsb, palette = load_multi_input(path, self.transform)
        return meta, alpha, lsb, palette, torch.tensor(stego_label, dtype=torch.float), torch.tensor(method_label, dtype=torch.long)
        return pixel, meta, torch.tensor(stego_label, dtype=torch.float), torch.tensor(method_label, dtype=torch.long)

def train_model(clean_dir, stego_dir, epochs=10, batch_size=8, lr=1e-4, out_path="models/detector_dual_stream.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Augmentation and Splitting ---
    train_transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
    ])

    full_dataset = StegoDataset(clean_dir, stego_dir, transform=None)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    train_dataset = copy.deepcopy(full_dataset)
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices.indices]
    train_dataset.transform = train_transform

    val_dataset = copy.deepcopy(full_dataset)
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices.indices]
    val_dataset.transform = val_transform

    print(f"\nDataset split into {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    # --- Dataloaders and Class Weights ---
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    stego_count = sum(1 for _, s, _ in train_dataset.samples if s == 1)
    method_counts = Counter(m for _, s, m in train_dataset.samples if s == 1)
    class_weights = None
    if stego_count > 0:
        num_stego_classes = len(full_dataset.method_map)
        class_weights_tensor = torch.zeros(num_stego_classes)
        for method_id, count in method_counts.items():
            if count > 0:
                class_weights_tensor[method_id] = float(stego_count) / (num_stego_classes * count)
        class_weights = class_weights_tensor / class_weights_tensor.sum() * num_stego_classes
        print(f"Calculated training class weights: {class_weights.tolist()}")

    # --- Model, Optimizer, Loss ---
    model = StarlightDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stego_criterion = nn.BCEWithLogitsLoss()
    
    if class_weights is not None:
        method_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=-1)
    else:
        method_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # --- Training Loop with Validation ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for meta, alpha, lsb, palette, stego_labels, method_labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            meta, alpha, lsb, palette = meta.to(device), alpha.to(device), lsb.to(device), palette.to(device)
            stego_labels, method_labels = stego_labels.to(device), method_labels.to(device)

            optimizer.zero_grad()
            stego_logits, method_logits, _, _ = model(meta, alpha, lsb, palette)
            loss = stego_criterion(stego_logits, stego_labels) + method_criterion(method_logits, method_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for meta, alpha, lsb, palette, stego_labels, method_labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                meta, alpha, lsb, palette = meta.to(device), alpha.to(device), lsb.to(device), palette.to(device)
                stego_labels, method_labels = stego_labels.to(device), method_labels.to(device)
                stego_logits, method_logits, _, _ = model(meta, alpha, lsb, palette)
                loss = stego_criterion(stego_logits, stego_labels) + method_criterion(method_logits, method_labels)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_path.replace('.onnx', '.pth'))
            print(f"Validation loss improved. Saved best model to {out_path.replace('.onnx', '.pth')}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement in validation loss for {patience} epochs. Stopping early.")
                break

    # --- ONNX Export ---
    print(f"Loading best model from {out_path.replace('.onnx', '.pth')} for ONNX export.")
    model.load_state_dict(torch.load(out_path.replace('.onnx', '.pth')))
    model.eval()
    
    dummy_meta = torch.randn(1, 1024).to(device)
    dummy_alpha = torch.randn(1, 1, 256, 256).to(device)
    dummy_lsb = torch.randn(1, 3, 256, 256).to(device)
    dummy_palette = torch.randn(1, 768).to(device)
    
    output_names = ["stego_logits", "method_logits", "method_id", "method_probs"]
    
    torch.onnx.export(
        model,
        (dummy_meta, dummy_alpha, dummy_lsb, dummy_palette),
        out_path,
        input_names=["metadata", "alpha", "lsb", "palette"],
        output_names=output_names,
        dynamic_axes={
            "metadata": {0: "batch"}, "alpha": {0: "batch"}, 
            "lsb": {0: "batch"}, "palette": {0: "batch"}
        }
    )
    print(f"Model exported to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", default="clean")
    parser.add_argument("--stego_dir", default="stego")
    parser.add_argument("--epochs", type=int, default=50) # Increased default epochs
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="models/detector_multi_stream.onnx")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    train_model(args.clean_dir, args.stego_dir, args.epochs, args.batch_size, args.lr, args.out)