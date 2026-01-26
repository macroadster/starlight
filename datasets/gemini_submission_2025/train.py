import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- CONFIGURATION ---
HINT_BYTES = b"AI42"
TERMINATOR_BYTES = b"\x00"
MAX_PAYLOAD_BITS = 2048 * 8  # Max payload size in bits for Alpha LSB
MAX_PAYLOAD_BYTES = 1024  # Max payload for EOI

# --- Model Architectures ---


class AlphaDetector(nn.Module):
    """A 2D CNN to detect steganography in the alpha channel of PNG images."""

    def __init__(self):
        super(AlphaDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 32, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class EoiDetector(nn.Module):
    """A 1D CNN to detect steganography in the EOI section of JPEG images."""

    def __init__(self):
        super(EoiDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 256, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class AlphaExtractor(nn.Module):
    """A 2D U-Net like model to extract bits from the alpha channel."""

    def __init__(self):
        super(AlphaExtractor, self).__init__()
        # Encoder
        self.down_conv1 = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.down_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())

        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_conv1 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU())
        self.up_conv2 = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x):
        # Encoder
        c1 = self.down_conv1(x)
        p1 = self.pool(c1)
        c2 = self.down_conv2(p1)

        # Decoder
        u1 = self.upsample(c2)
        u1 = torch.cat([u1, c1], dim=1)  # Concatenate skip connection
        c3 = self.up_conv1(u1)
        out = self.up_conv2(c3)
        return out


class EoiExtractor(nn.Module):
    """A 1D CNN to extract bytes from the EOI section."""

    def __init__(self):
        super(EoiExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.conv_layers(x)


# --- Datasets ---


def get_payload_bits(full_byte_payload):
    payload_bits = []
    for byte in full_byte_payload:
        for i in range(8):
            payload_bits.append((byte >> i) & 1)
    return payload_bits


def load_payload(image_path):
    """Loads the payload corresponding to a stego image."""
    base_name = os.path.basename(image_path).split("_")[0]
    payload_file = f"{base_name}.md"
    if not os.path.exists(payload_file):
        # Fallback for random payload
        return HINT_BYTES + os.urandom(2048) + TERMINATOR_BYTES
    with open(payload_file, "r", encoding="utf-8") as f:
        payload_bytes = f.read().encode("utf-8")
    return HINT_BYTES + payload_bytes + TERMINATOR_BYTES


class StegoDataset(Dataset):
    def __init__(self, file_list, preprocess_fn, task, method):
        self.file_list = file_list
        self.preprocess_fn = preprocess_fn
        self.task = task
        self.method = method

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = self.preprocess_fn(file_path)

        if self.task == "detect":
            label = 1.0 if "stego" in file_path else 0.0
            return torch.tensor(data, dtype=torch.float32), torch.tensor(
                label, dtype=torch.float32
            )

        elif self.task == "extract":
            if "clean" in file_path:  # No payload in clean images
                if self.method == "alpha":
                    payload_label = np.zeros((256, 256), dtype=np.float32)
                else:  # eoi
                    payload_label = np.zeros((1, MAX_PAYLOAD_BYTES), dtype=np.float32)
            else:
                payload_bytes = load_payload(file_path)
                if self.method == "alpha":
                    payload_bits = get_payload_bits(payload_bytes)
                    payload_mask = np.zeros(256 * 256, dtype=np.float32)
                    payload_mask[: len(payload_bits)] = payload_bits
                    payload_label = payload_mask.reshape((256, 256))
                else:  # eoi
                    payload_padded = np.zeros(MAX_PAYLOAD_BYTES, dtype=np.float32)
                    # Truncate payload_bytes if it's too long
                    actual_payload_bytes = payload_bytes[:MAX_PAYLOAD_BYTES]
                    payload_padded[: len(actual_payload_bytes)] = [
                        b for b in actual_payload_bytes
                    ]
                    payload_label = np.expand_dims(payload_padded, 0)

            return torch.tensor(data, dtype=torch.float32), torch.tensor(
                payload_label, dtype=torch.float32
            )


def preprocess_alpha(img_path):
    img = Image.open(img_path).convert("RGBA").resize((256, 256))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def preprocess_eoi(img_path):
    with open(img_path, "rb") as f:
        f.seek(-1024, 2)
        tail = f.read()
    eoi_pos = tail.find(b"\xff\xd9")
    appended = tail[eoi_pos + 2 :] if eoi_pos != -1 else tail
    data = np.frombuffer(appended, dtype=np.uint8)[:1024]
    padded = np.zeros(1024, dtype=np.float32)
    padded[: len(data)] = data.astype(np.float32) / 255.0
    return padded.reshape(1, -1)  # Reshape to (1, 1024) for (channel, length)


# --- Training ---


def train(model, dataloader, criterion, optimizer, device, task):
    model.train()
    total_loss = 0
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        if task == "detect":
            loss = criterion(outputs, labels.unsqueeze(1))
        else:  # extract
            if model.__class__.__name__ == "AlphaExtractor":
                loss = criterion(outputs.squeeze(1), labels)
            else:
                loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(
        description="Train all steganography detectors and extractors."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    args = parser.parse_args()

    # --- Device Configuration ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # --- Iterate over all methods and tasks ---
    for method in ["alpha", "eoi"]:
        for task in ["detect", "extract"]:
            print(f"--- Starting training for method='{method}', task='{task}' ---")

            # --- Model and Data Config ---
            if method == "alpha":
                preprocess_fn = preprocess_alpha
                file_extension = "png"
                if task == "detect":
                    model = AlphaDetector()
                    criterion = nn.BCELoss()
                else:  # extract
                    model = AlphaExtractor()
                    criterion = nn.BCEWithLogitsLoss()
            else:  # eoi
                preprocess_fn = preprocess_eoi
                file_extension = "jpeg"
                if task == "detect":
                    model = EoiDetector()
                    criterion = nn.BCELoss()
                else:  # extract
                    model = EoiExtractor()
                    criterion = nn.MSELoss()

            # --- Data Loading ---
            clean_files = glob.glob(f"clean/*_{method}_*.{file_extension}")
            stego_files = glob.glob(f"stego/*_{method}_*.{file_extension}")
            all_files = clean_files + stego_files

            if not all_files:
                print(f"No files found for method '{method}'. Skipping.")
                continue

            train_files, _ = train_test_split(all_files, test_size=0.2, random_state=42)
            train_dataset = StegoDataset(train_files, preprocess_fn, task, method)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )

            # --- Training ---
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            print(f"Training {method} {task} on {device} for {args.epochs} epochs...")
            for epoch in range(args.epochs):
                loss = train(model, train_loader, criterion, optimizer, device, task)
                print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

            # --- Save Model ---
            model_dir = f"model/{method}"
            os.makedirs(model_dir, exist_ok=True)

            output_path = f"{model_dir}/{task}or.onnx"

            print(f"Saving model to {output_path}")

            # Ensure dummy_input has the correct shape
            if not train_dataset:
                print("Skipping model export due to empty dataset.")
                continue

            dummy_input = torch.randn(1, *train_dataset[0][0].shape, device=device)

            input_names = ["input"]
            output_names = ["output"]

            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=11,
                verbose=False,
            )
            print(f"--- Finished training for method='{method}', task='{task}' ---\n")

    print("All training complete.")


if __name__ == "__main__":
    main()
