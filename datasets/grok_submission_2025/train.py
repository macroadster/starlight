# train.py - Grok Submission Steganography Training
# --------------------------------------------------------------
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import cv2
import albumentations as A
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from dataset import StegoPairDataset

# ------------------- CONFIG -------------------
CFG = {
    "msg_len": 100,
    "batch_size": 4,  # Smaller batch for better convergence
    "epochs": 50,  # Shorter training
    "lr": 5e-4,  # Higher learning rate
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "save_dir": "checkpoints",
    "img_size": 256,
}
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
os.makedirs(CFG["save_dir"], exist_ok=True)

# ------------------- TRANSFORMS (FIXED: NO CROP ERROR) -------------------
train_tf = A.Compose(
    [
        A.LongestMaxSize(max_size=CFG["img_size"] + 64),
        A.PadIfNeeded(
            min_height=CFG["img_size"],
            min_width=CFG["img_size"],
            border_mode=cv2.BORDER_CONSTANT,
            fill_value=0,
        ),
        A.RandomCrop(width=CFG["img_size"], height=CFG["img_size"]),
        A.HorizontalFlip(p=0.5),
        A.ToFloat(max_value=255),
    ],
    additional_targets={"mask": "image"},
)

val_tf = A.Compose(
    [
        A.LongestMaxSize(max_size=CFG["img_size"]),
        A.PadIfNeeded(
            min_height=CFG["img_size"],
            min_width=CFG["img_size"],
            border_mode=cv2.BORDER_CONSTANT,
            fill_value=0,
        ),
        A.ToFloat(max_value=255),
    ],
    additional_targets={"mask": "image"},
)


# ------------------- LOADERS -------------------
def get_loaders():
    train_set = StegoPairDataset(
        clean_root="../sample_submission_2025/clean",
        stego_root="../sample_submission_2025/stego",
        msg_len=CFG["msg_len"],
        transform=train_tf,
        strict=False,
    )
    val_set = StegoPairDataset(
        clean_root="../val/clean",
        stego_root="../val/stego",
        msg_len=CFG["msg_len"],
        transform=val_tf,
        strict=False,
    )
    print(f"[Loader] Train: {len(train_set)} | Val: {len(val_set)}")

    if len(train_set) == 0:
        raise ValueError("No training data! Check JSONs and clean_file paths.")

    return (
        DataLoader(
            train_set,
            batch_size=CFG["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        DataLoader(
            val_set,
            batch_size=CFG["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
    )


# ------------------- MODEL -------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, norm=True):
        super().__init__()
        p = k // 2
        layers = [nn.Conv2d(cin, cout, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(cout)]
        layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(c, c), ConvBlock(c, c, norm=False))

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, msg_len):
        super().__init__()
        self.msg_proj = nn.Sequential(
            nn.Linear(msg_len, 64 * 64), nn.ReLU(inplace=True)
        )
        self.init = ConvBlock(3, 64)
        self.d1 = nn.Sequential(ConvBlock(64, 128, s=2), ResBlock(128))
        self.d2 = nn.Sequential(ConvBlock(128, 256, s=2), ResBlock(256))
        self.d3 = nn.Sequential(ConvBlock(256, 512, s=2), ResBlock(512))
        self.attn = nn.Sequential(nn.Conv2d(512, 1, 1), nn.Sigmoid())
        self.res_head = nn.Conv2d(512, 3, 3, padding=1)

    def forward(self, cover, msg):
        b = cover.size(0)
        msg_feat = self.msg_proj(msg).view(b, 1, 64, 64)
        msg_feat = F.interpolate(
            msg_feat, size=cover.shape[2:], mode="bilinear", align_corners=False
        )
        x = self.init(cover)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        attn = F.interpolate(
            self.attn(x), size=cover.shape[2:], mode="bilinear", align_corners=False
        )
        residual = (
            F.interpolate(
                self.res_head(x),
                size=cover.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            * attn
        )
        msg_mod = msg_feat * 2.0 - 1.0
        residual = residual * (1.0 + 3.0 * msg_mod)  # ← weaker modulation
        residual = residual * 0.15  # ← stronger base
        stego = torch.clamp(cover + residual, 0, 1)
        return stego, residual


class Decoder(nn.Module):
    def __init__(self, msg_len):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 64),
            ResBlock(64),
            ConvBlock(64, 128, s=2),
            ResBlock(128),
            ConvBlock(128, 256, s=2),
            ResBlock(256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, msg_len),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 64),
            ResBlock(64),
            ConvBlock(64, 128, s=2),
            ResBlock(128),
            ConvBlock(128, 256, s=2),
            ResBlock(256),
            ConvBlock(256, 512, s=2),
            ResBlock(512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x)


def set_requires_grad(m, flag):
    for p in m.parameters():
        p.requires_grad = flag


bce = nn.BCELoss()
mse = nn.MSELoss()


# ------------------- TRAIN EPOCH -------------------
def train_one_epoch(enc, dec, crit, loader, opt_encdec, opt_crit):
    enc.train()
    dec.train()
    crit.train()
    pbar = tqdm(loader, desc="train", leave=False)
    for cover, _, msg in pbar:
        cover = cover.to(CFG["device"])
        msg = msg.to(CFG["device"])

        # Critic
        set_requires_grad(enc, False)
        set_requires_grad(dec, False)
        crit.zero_grad()
        real_logit = crit(cover)
        fake, _ = enc(cover, msg)
        fake_logit = crit(fake.detach())
        loss_c = -(real_logit.mean() - fake_logit.mean())
        loss_c.backward()
        opt_crit.step()

        # Encoder + Decoder
        set_requires_grad(enc, True)
        set_requires_grad(dec, True)
        opt_encdec.zero_grad()
        fake, residual = enc(cover, msg)
        pred_msg = dec(fake)

        loss_msg = 10.0 * bce(pred_msg, msg) + mse(pred_msg, msg)
        loss_sparsity = 0.1 * F.l1_loss(residual, torch.zeros_like(residual))
        loss_adv = -0.1 * crit(fake).mean()

        # COVER LOSS
        loss_cover = F.l1_loss(fake, cover)

        loss_total = loss_msg + 0.1 * loss_adv + loss_sparsity + loss_cover
        loss_total.backward()
        opt_encdec.step()

        psnr_val = min(
            psnr_metric(
                (cover[0].detach().cpu().numpy() * 255).astype(np.uint8),
                (fake[0].detach().cpu().numpy() * 255).astype(np.uint8),
                data_range=255,
            ),
            55.0,
        )
        pbar.set_postfix(loss_msg=loss_msg.item(), psnr=psnr_val)


# ------------------- VALIDATE -------------------
@torch.no_grad()
def validate(enc, dec, loader):
    enc.eval()
    dec.eval()
    bit_errs, psnrs, ssims = [], [], []
    for cover, _, msg in loader:
        cover, msg = cover.to(CFG["device"]), msg.to(CFG["device"])
        fake, _ = enc(cover, msg)
        pred = dec(fake)
        bit_errs.append((pred.round() != msg).float().mean().item())

        # FIXED: detach() before numpy()
        fake_np = (
            (fake.clamp(0, 1).detach().cpu().numpy() * 255)
            .astype(np.uint8)
            .transpose(0, 2, 3, 1)
        )
        cover_np = (
            (cover.detach().cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        )
        for f, c in zip(fake_np, cover_np):
            psnrs.append(min(psnr_metric(c, f, data_range=255), 55.0))
            ssims.append(ssim_metric(c, f, channel_axis=2, data_range=255, win_size=7))
    return np.mean(bit_errs), np.mean(psnrs), np.mean(ssims)


# ------------------- MAIN -------------------
def main():
    train_loader, val_loader = get_loaders()
    enc = Encoder(CFG["msg_len"]).to(CFG["device"])
    dec = Decoder(CFG["msg_len"]).to(CFG["device"])
    crit = Critic().to(CFG["device"])

    opt_encdec = optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=CFG["lr"],
        betas=(0.5, 0.999),
    )
    opt_crit = optim.Adam(crit.parameters(), lr=CFG["lr"], betas=(0.5, 0.999))

    best = 1.0
    for epoch in range(1, CFG["epochs"] + 1):
        train_one_epoch(enc, dec, crit, train_loader, opt_encdec, opt_crit)
        bit_err, psnr, ssim = validate(enc, dec, val_loader)
        print(
            f"Epoch {epoch:02d} | BitErr {bit_err:.4f} | PSNR {psnr:.2f} | SSIM {ssim:.4f}"
        )
        if bit_err < best:
            best = bit_err
            torch.save(
                {"enc": enc.state_dict(), "dec": dec.state_dict()},
                f"{CFG['save_dir']}/best.pth",
            )
            print("  *** BEST MODEL SAVED ***")
    print("Done → checkpoints/best.pth")


if __name__ == "__main__":
    main()
