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
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

# Define transforms for images
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        features = self.extract_features(path, img)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def extract_features(self, path, img):
        file_size = os.path.getsize(path) / 1024.0  # in KB
        exif = img.getexif()
        exif_present = 1.0 if exif else 0.0
        exif_length = len(exif) if exif else 0.0
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
        return [file_size, exif_present, exif_length, palette_present, palette_length, palette_entropy_value, eof_length]

# Function to collect data from dataset folders
def collect_data(root_dir='dataset'):
    image_paths = []
    labels = []
    class_map = {'clean': 0, 'alpha': 1, 'palette': 2, 'dct': 3, 'lsb': 4, 'eoi': 5, 'exif': 6}
    # Updated regex to match known algorithm names
    pattern = re.compile(r'_({})_(\d{{3}})\.\w+$'.format('|'.join(class_map.keys())))
    for subdir in os.listdir(root_dir):
        if '_submission_' in subdir:
            base = os.path.join(root_dir, subdir)
            clean_dir = os.path.join(base, 'clean')
            stego_dir = os.path.join(base, 'stego')
            if os.path.exists(clean_dir):
                for fname in os.listdir(clean_dir):
                    if os.path.isfile(os.path.join(clean_dir, fname)):
                        image_paths.append(os.path.join(clean_dir, fname))
                        labels.append(0)  # clean
            if os.path.exists(stego_dir):
                for fname in os.listdir(stego_dir):
                    if os.path.isfile(os.path.join(stego_dir, fname)):
                        match = pattern.search(fname)
                        if match:
                            algo = match.group(1)
                            image_paths.append(os.path.join(stego_dir, fname))
                            labels.append(class_map[algo])
                        else:
                            print(f"Invalid filename format: {fname}")
    if not image_paths:
        print("Warning: No valid images found in the dataset. Check folder structure and filenames.")
    return image_paths, labels

# Model Definition
class Starlight(nn.Module):
    def __init__(self, num_classes=7, feature_dim=7):
        super(Starlight, self).__init__()
        # SRM Filter (simple high-pass kernel)
        kernel = torch.tensor([[[ -1.,  2., -1.],
                                [  2., -4.,  2.],
                                [ -1.,  2., -1.]]]).repeat(3, 1, 1, 1)
        self.srm_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        self.srm_conv.weight = nn.Parameter(kernel, requires_grad=False)
        
        # PD Branch: ResNet-18 without FC
        resnet = models.resnet18(weights=None)  # Updated to use weights=None
        self.pd_backbone = nn.Sequential(*list(resnet.children())[:-1])  # Output: (B, 512, 1, 1)
        
        # SF Branch: MLP
        self.sf_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # FD Branch: Custom CNN for DCT coefficients
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
        
        # Fusion Head
        self.fusion = nn.Sequential(
            nn.Linear(512 + 32 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, features):
        # PD Branch
        filtered = self.srm_conv(image)
        pd_feat = self.pd_backbone(filtered).flatten(1)  # (B, 512)
        
        # SF Branch
        sf_feat = self.sf_mlp(features)  # (B, 32)
        
        # FD Branch
        dct_img = self.dct2d(image)  # (B, 3, H, W)
        fd_feat = self.fd_cnn(dct_img)  # (B, 256)
        
        # Concatenation
        concatenated = torch.cat([pd_feat, sf_feat, fd_feat], dim=1)
        out = self.fusion(concatenated)
        return out
    
    def dct2d(self, x):
        # Approximate DCT using FFT-based method (Type-II DCT)
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
        
        # Apply DCT on columns (dim=3)
        dct_col = dct1d(x)
        # Transpose to apply on rows
        dct_col = dct_col.transpose(2, 3)
        dct_row = dct1d(dct_col)
        # Transpose back
        dct_row = dct_row.transpose(2, 3)
        return dct_row

# Main Training Script
if __name__ == "__main__":
    # Collect data
    image_paths, labels = collect_data()
    if not image_paths:
        print("Error: No data found. Ensure dataset is populated and filenames follow the convention.")
        exit()

    # Print dataset statistics
    class_map = {'clean': 0, 'alpha': 1, 'palette': 2, 'dct': 3, 'lsb': 4, 'eoi': 5, 'exif': 6}
    label_counts = Counter(labels)
    print("Dataset Statistics:")
    for algo, idx in class_map.items():
        count = label_counts.get(idx, 0)
        print(f"  {algo}: {count} samples ({100. * count / len(labels):.2f}%)")
    print(f"Total samples: {len(labels)}")

    # Split into train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Datasets and Loaders
    train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Class weights for imbalance
    train_count = Counter(train_labels)
    num_classes = 7
    class_freq = [train_count.get(i, 1) for i in range(num_classes)]
    class_weights = torch.tensor([1.0 / f for f in class_freq], dtype=torch.float32)

    # Device: Support MPS, CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = Starlight(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Early Stopping
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(100):
        # Train
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

        # Validate
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

        # Metrics
        bal_acc = balanced_accuracy_score(trues_np, preds_np)
        f1 = f1_score(trues_np, preds_np, average='macro')
        acc = accuracy_score(trues_np, preds_np)
        binary_true = (trues_np > 0).astype(int)
        stego_scores = 1 - softmax_probs[:, 0]
        auc = roc_auc_score(binary_true, stego_scores) if len(set(binary_true)) > 1 else 0.0

        print(f"Val Loss: {val_loss:.4f}, Balanced Acc: {bal_acc:.4f}, F1: {f1:.4f}, AUC (Clean vs Stego): {auc:.4f}, Top-1 Acc: {acc:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model with Val Loss: {:.4f}".format(best_val_loss))
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break
