import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import struct
import os
from torch.utils.data import Dataset, DataLoader
import argparse

class StarlightDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Pixel backbone (MobileNet-V3-Small)
        self.pixel_backbone = models.mobilenet_v3_small(pretrained=False).features
        # Modify first conv to accept 4 channels (for RGBA)
        self.pixel_backbone[0] = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.pixel_pool = nn.AdaptiveAvgPool2d(1)

        # Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 64)
        )

        # Method classifier (5 methods)
        self.method_head = nn.Sequential(
            nn.Linear(576 + 64, 128), nn.ReLU(),
            nn.Linear(128, 5)
        )

        # Final stego head
        self.stego_head = nn.Sequential(
            nn.Linear(576 + 64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, pixel, metadata):
        # Pixel path
        p = self.pixel_pool(self.pixel_backbone(pixel)).flatten(1)  # (B,576)

        # Metadata path
        m = self.meta_mlp(metadata)  # (B,64)

        # Fuse
        fused = torch.cat([p, m], dim=1)  # (B,640)

        method_logits = self.method_head(fused)
        method_probs = torch.softmax(method_logits, dim=1)
        stego_prob = torch.sigmoid(self.stego_head(fused)).squeeze(1)

        return {
            "stego_prob": stego_prob,
            "method_id": method_probs.argmax(1),
            "method_probs": method_probs
        }

def load_dual_input(path):
    # --- Pixel path ---
    img = Image.open(path)
    if img.mode == "P":
        # For palette, keep indices as 1-channel
        img = img.resize((256,256), resample=0)  # NEAREST
        indices = np.array(img).astype(np.float32) / 255.0  # (256,256)
        pixel = np.stack([indices] * 4, axis=-1)  # (256,256,4), all channels same
    else:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        pixel = np.array(img.resize((256,256), resample=0)).astype(np.float32)/255.0  # (256,256,4)
    pixel = torch.from_numpy(pixel).permute(2,0,1).unsqueeze(0)  # (1,4,256,256)

    # --- Metadata path ---
    with open(path, 'rb') as f:
        raw = f.read()

    # EXIF: find first APP1 (0xFFE1)
    exif = b""
    pos = raw.find(b'\xFF\xE1')
    if pos != -1:
        length = struct.unpack('>H', raw[pos+2:pos+4])[0]
        exif = raw[pos+4:pos+4+length-2]

    # EOI tail: after last 0xFFD9
    eoi_pos = raw.rfind(b'\xFF\xD9')
    tail = raw[eoi_pos+2:] if eoi_pos != -1 else b""

    # Combine + pad to 1024
    meta = np.frombuffer(exif + tail, dtype=np.uint8)[:1024]
    meta = np.pad(meta, (0, 1024 - len(meta)), 'constant')
    meta = torch.from_numpy(meta.astype(np.float32)/255.0).unsqueeze(0)  # (1,1024)

    return pixel, meta

class StegoDataset(Dataset):
    def __init__(self, clean_dir, stego_dir):
        self.samples = []
        method_map = {"alpha": 0, "palette": 1, "rgb_lsb": 2, "exif": 3, "eoi": 4}

        # Clean samples
        for f in os.listdir(clean_dir):
            if f.endswith(('.jpg', '.png', '.gif', '.webp', '.bmp', '.jpeg')):
                self.samples.append((os.path.join(clean_dir, f), 0, -1))  # stego=0, method=-1

        # Stego samples
        for f in os.listdir(stego_dir):
            if f.endswith(('.jpg', '.png', '.gif', '.webp', '.bmp', '.jpeg')):
                # Extract method from filename, e.g., current_alpha_000.png -> alpha
                parts = f.split('_')
                if len(parts) >= 2:
                    method_str = parts[1]
                    if method_str in method_map:
                        method_id = method_map[method_str]
                        self.samples.append((os.path.join(stego_dir, f), 1, method_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, stego_label, method_label = self.samples[idx]
        pixel, meta = load_dual_input(path)
        return pixel.squeeze(0), meta.squeeze(0), torch.tensor(stego_label, dtype=torch.float), torch.tensor(method_label, dtype=torch.long)

def train_model(clean_dir, stego_dir, epochs=10, batch_size=8, lr=1e-3, out_path="models/detector_dual.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StegoDataset(clean_dir, stego_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StarlightDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stego_criterion = nn.BCELoss()
    method_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore clean samples for method loss

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for pixel, meta, stego_labels, method_labels in dataloader:
            pixel, meta = pixel.to(device), meta.to(device)
            stego_labels, method_labels = stego_labels.to(device), method_labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel, meta)

            stego_loss = stego_criterion(outputs["stego_prob"], stego_labels)
            method_loss = method_criterion(outputs["method_probs"], method_labels)
            loss = stego_loss + method_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Export to ONNX
    model.eval()
    dummy_pixel = torch.randn(1,4,256,256).to(device)
    dummy_meta = torch.randn(1,1024).to(device)

    torch.onnx.export(
        model,
        (dummy_pixel, dummy_meta),
        out_path,
        input_names=["pixel", "metadata"],
        output_names=["stego_prob", "method_id", "method_probs"],
        dynamic_axes={"pixel": {0: "batch"}, "metadata": {0: "batch"}}
    )
    print(f"Model exported to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", default="clean")
    parser.add_argument("--stego_dir", default="stego")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/detector_dual.onnx")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    train_model(args.clean_dir, args.stego_dir, args.epochs, args.batch_size, args.lr, args.out)