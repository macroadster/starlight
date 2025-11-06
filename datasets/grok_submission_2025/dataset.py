# dataset.py – JSON = stego_image + '.json'
import os
import json
import torch
import cv2
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio as psnr_metric


class StegoPairDataset(Dataset):
    def __init__(
        self, clean_root, stego_root, msg_len=100, transform=None, strict=False
    ):
        self.clean_root = clean_root
        self.stego_root = stego_root
        self.msg_len = msg_len
        self.transform = transform
        self.strict = strict
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []

        # --- SCAN STEGO DIR FOR LSB FILES ---
        stego_files = [
            f
            for f in os.listdir(self.stego_root)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and "lsb" in f.lower()
        ]
        print(f"[DEBUG] Found {len(stego_files)} LSB stego files")

        for stego_name in stego_files:
            stego_path = os.path.join(self.stego_root, stego_name)
            json_path = stego_path + ".json"  # ← THIS IS THE KEY FIX

            if not os.path.exists(json_path):
                print(f"[SKIP] JSON missing: {json_path}")
                continue

            # --- LOAD JSON ---
            try:
                with open(json_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[ERROR] JSON load failed {json_path}: {e}")
                continue

            clean_file_rel = meta.get("clean_file")
            if not clean_file_rel:
                print(f"[SKIP] No 'clean_file' in {json_path}")
                continue

            # --- RESOLVE CLEAN PATH ---
            clean_path = os.path.join(self.clean_root, clean_file_rel)
            if not os.path.exists(clean_path):
                print(f"[SKIP] Clean file not found: {clean_path}")
                continue

            # --- LOAD MESSAGE ---
            msg_list = meta.get("message", [])
            msg = (
                torch.tensor(msg_list[: self.msg_len], dtype=torch.float32)
                if len(msg_list) >= self.msg_len
                else torch.zeros(self.msg_len)
            )

            # --- LOAD IMAGES ---
            cover = cv2.imread(clean_path)
            stego = cv2.imread(stego_path)
            if cover is None or stego is None:
                print("[SKIP] Image load failed")
                continue

            # --- PSNR FILTER ---
            psnr = psnr_metric(cover, stego, data_range=255)
            if psnr < 35.0:
                print(f"[FILTER] PSNR {psnr:.1f} < 35 → skip {stego_name}")
                continue

            pairs.append((clean_path, stego_path, msg))
            print(f"[OK] PSNR {psnr:.1f} | {stego_name} → {clean_file_rel}")

        print(f"[Dataset] {len(pairs)} LSB pairs loaded")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, stego_path, msg = self.pairs[idx]
        cover_img = cv2.imread(clean_path)
        stego_img = cv2.imread(stego_path)

        if cover_img is None or stego_img is None:
            raise ValueError(f"Failed to load images: {clean_path}, {stego_path}")

        cover = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
        stego = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            aug = self.transform(image=cover, mask=stego)
            cover, stego = aug["image"], aug["mask"]

        cover = torch.from_numpy(cover).permute(2, 0, 1) / 255.0
        stego = torch.from_numpy(stego).permute(2, 0, 1) / 255.0
        return cover, stego, msg
