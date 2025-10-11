#!/usr/bin/env python3
# ============================================================
# Project Starlight - Steganography Scanner
# Detecting Hidden Data in Images
# ============================================================

import os
import sys
import torch
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Try to import SRNet and StegoTransform from trainer.py
# ------------------------------------------------------------
try:
    from trainer import SRNet, StegoTransform
    _HAS_TRAINER = True
except Exception:
    SRNet = None
    StegoTransform = None
    _HAS_TRAINER = False


# ============================================================
# Model Definitions (legacy ResNet/CNN + new SRNet)
# ============================================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StegoDetectorResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class StegoDetectorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# Dataset Loader
# ============================================================

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.folder = folder
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)  # Do NOT convert to RGB here - let transform handle modes
        if self.transform:
            img = self.transform(img)
        return img, path


# ============================================================
# StegoScanner Wrapper
# ============================================================

class StegoScanner:
    def __init__(self, model_path, model_type='srnet', device='auto', batch_size=8):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
            elif device == 'mps' and not torch.backends.mps.is_available():
                print("MPS not available, falling back to CPU")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)
        self.model_type = model_type.lower()
        self.batch_size = batch_size

        # Choose model
        if self.model_type == 'cnn':
            self.model = StegoDetectorCNN()
        elif self.model_type == 'resnet':
            self.model = StegoDetectorResNet()
        elif self.model_type == 'srnet' and _HAS_TRAINER:
            self.model = SRNet()
        else:
            self.model = StegoDetectorResNet()

        self.model.to(self.device)

        # Load checkpoint with robust fallback
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded checkpoint into {self.model.__class__.__name__}")
        except RuntimeError as e:
            print(f"[Warning] Primary model load failed: {e}")
            if _HAS_TRAINER and self.model_type != 'srnet':
                print("Retrying load with SRNet architecture from trainer.py ...")
                self.model = SRNet().to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model_type = 'srnet'
                print("Loaded checkpoint into SRNet.")
            else:
                raise

        # Choose preprocessing
        if _HAS_TRAINER and self.model_type == 'srnet' and StegoTransform is not None:
            self.transform = StegoTransform(size=256, augment=False)
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB')),  # Convert for legacy models
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

    # ------------------------------------------------------------
    def scan_folder(self, folder):
        dataset = ImageFolderDataset(folder, self.transform)

        # Device-specific DataLoader settings
        use_pin_memory = (self.device.type == 'cuda')
        num_workers = 4 if self.device.type == 'cuda' else 0  # More workers for CUDA; 0 for others to avoid issues

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
        results = []

        self.model.eval()
        with torch.no_grad():
            for imgs, paths in loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = probs[:, 1].cpu().numpy()

                for p, s in zip(paths, preds):
                    results.append((p, float(s)))
        return results


# ============================================================
# Main CLI Entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Project Starlight - Steganography Scanner")
    parser.add_argument('--input', required=True, help="Input folder to scan")
    parser.add_argument('--model', required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument('--model_type', default='srnet', help="Model type: resnet | cnn | srnet")  # Default to srnet
    parser.add_argument('--device', default='auto', help="Device to use (auto, cpu, cuda, or mps)")
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print("============================================================")
    print("Project Starlight - Steganography Scanner")
    print("Detecting Hidden Data in Images")
    print("============================================================")

    scanner = StegoScanner(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        batch_size=args.batch_size
    )

    print(f"Using device: {scanner.device.type}")

    results = scanner.scan_folder(args.input)

    print("\nScan results:")
    for path, prob in results:
        print(f"{path}: stego probability = {prob:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
