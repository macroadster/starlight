# train_stego_selfsupervised.py – Self-supervised neural steganography
import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import cv2
import albumentations as A
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# Simple dataset that just loads clean images
class CleanImageDataset(Dataset):
    def __init__(self, clean_root, transform=None):
        self.clean_root = clean_root
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(clean_root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'))
        ]
        print(f"[Dataset] Found {len(self.image_files)} clean images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.clean_root, self.image_files[idx])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback to a random image if loading fails
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']
        
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        return img

# ------------------- CONFIG -------------------
CFG = {
    "msg_len"    : 100,
    "batch_size" : 4,
    "epochs"     : 30,
    "lr"         : 1e-4,
    "device"     : "cuda" if torch.cuda.is_available() else "cpu",
    "seed"       : 42,
    "save_dir"   : "checkpoints",
    "img_size"   : 256,
}
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
os.makedirs(CFG["save_dir"], exist_ok=True)

# ------------------- TRANSFORMS -------------------
train_tf = A.Compose([
    A.LongestMaxSize(max_size=CFG["img_size"] + 64),
    A.PadIfNeeded(
        min_height=CFG["img_size"], min_width=CFG["img_size"],
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.RandomCrop(width=CFG["img_size"], height=CFG["img_size"]),
    A.HorizontalFlip(p=0.5),
    A.ToFloat(max_value=255),
])

val_tf = A.Compose([
    A.LongestMaxSize(max_size=CFG["img_size"]),
    A.PadIfNeeded(
        min_height=CFG["img_size"], min_width=CFG["img_size"],
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.ToFloat(max_value=255),
])

# ------------------- LOADERS -------------------
def get_loaders():
    train_set = CleanImageDataset(
        clean_root="datasets/sample_submission_2025/clean",
        transform=train_tf,
    )
    val_set = CleanImageDataset(
        clean_root="datasets/val/clean",
        transform=val_tf,
    )
    print(f"[Loader] Train: {len(train_set)} | Val: {len(val_set)}")
    
    return (
        DataLoader(train_set, batch_size=CFG["batch_size"], shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_set,   batch_size=CFG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    )

# ------------------- MODEL (same as before) -------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, norm=True):
        super().__init__()
        p = k // 2
        layers = [nn.Conv2d(cin, cout, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(cout)]
        layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, c): super().__init__(); self.net = nn.Sequential(ConvBlock(c,c), ConvBlock(c,c,norm=False))
    def forward(self, x): return x + self.net(x)

class Encoder(nn.Module):
    def __init__(self, msg_len):
        super().__init__()
        self.msg_proj = nn.Sequential(nn.Linear(msg_len, 64*64), nn.ReLU(inplace=True))
        self.init = ConvBlock(3, 64)
        self.d1 = nn.Sequential(ConvBlock(64,128,s=2), ResBlock(128))
        self.d2 = nn.Sequential(ConvBlock(128,256,s=2), ResBlock(256))
        self.d3 = nn.Sequential(ConvBlock(256,512,s=2), ResBlock(512))
        self.attn = nn.Sequential(nn.Conv2d(512,1,1), nn.Sigmoid())
        self.res_head = nn.Conv2d(512,3,3,padding=1)
    
    def forward(self, cover, msg):
        b = cover.size(0)
        msg_feat = self.msg_proj(msg).view(b,1,64,64)
        msg_feat = F.interpolate(msg_feat, size=cover.shape[2:], mode='bilinear', align_corners=False)
        x = self.init(cover); x = self.d1(x); x = self.d2(x); x = self.d3(x)
        attn = F.interpolate(self.attn(x), size=cover.shape[2:], mode='bilinear', align_corners=False)
        residual = F.interpolate(self.res_head(x), size=cover.shape[2:], mode='bilinear', align_corners=False) * attn
        msg_mod = msg_feat * 2.0 - 1.0
        residual = residual * (1.0 + 3.0 * msg_mod)
        residual = residual * 0.15
        stego = torch.clamp(cover + residual, 0, 1)
        return stego, residual

class Decoder(nn.Module):
    def __init__(self, msg_len):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3,64), ResBlock(64),
            ConvBlock(64,128,s=2), ResBlock(128),
            ConvBlock(128,256,s=2), ResBlock(256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, msg_len), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def set_requires_grad(m, flag):
    for p in m.parameters(): p.requires_grad = flag

bce = nn.BCELoss()
mse = nn.MSELoss()

# ------------------- TRAIN EPOCH -------------------
def train_one_epoch(enc, dec, loader, opt):
    enc.train(); dec.train()
    pbar = tqdm(loader, desc="train", leave=False)
    total_loss = 0
    for cover in pbar:
        cover = cover.to(CFG["device"])
        msg = torch.randint(0, 2, (cover.size(0), CFG["msg_len"])).float().to(CFG["device"])
        
        opt.zero_grad()
        stego, residual = enc(cover, msg)
        pred_msg = dec(stego)
        
        # Losses
        loss_msg = 10.0 * bce(pred_msg, msg) + mse(pred_msg, msg)
        loss_cover = F.l1_loss(stego, cover)
        loss_sparsity = 0.1 * F.l1_loss(residual, torch.zeros_like(residual))
        
        loss_total = loss_msg + loss_cover + loss_sparsity
        loss_total.backward()
        opt.step()
        
        total_loss += loss_total.item()
        pbar.set_postfix(loss=loss_total.item(), msg_acc=1.0-F.binary_cross_entropy(pred_msg, msg).item())
    
    return total_loss / len(loader)

# ------------------- VALIDATE -------------------
@torch.no_grad()
def validate(enc, dec, loader):
    enc.eval(); dec.eval()
    bit_errs, psnrs = [], []
    for cover in loader:
        cover = cover.to(CFG["device"])
        msg = torch.randint(0, 2, (cover.size(0), CFG["msg_len"])).float().to(CFG["device"])
        
        stego, _ = enc(cover, msg)
        pred = dec(stego)
        
        bit_errs.append((pred.round() != msg).float().mean().item())
        
        stego_np = (stego.clamp(0,1).cpu().numpy()*255).astype(np.uint8).transpose(0,2,3,1)
        cover_np = (cover.cpu().numpy()*255).astype(np.uint8).transpose(0,2,3,1)
        for s,c in zip(stego_np, cover_np):
            psnrs.append(min(psnr_metric(c,s,data_range=255), 55.0))
    
    return np.mean(bit_errs), np.mean(psnrs)

# ------------------- MAIN -------------------
def main():
    train_loader, val_loader = get_loaders()
    enc = Encoder(CFG["msg_len"]).to(CFG["device"])
    dec = Decoder(CFG["msg_len"]).to(CFG["device"])
    
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=CFG["lr"], betas=(0.5,0.999))
    
    best = 1.0
    for epoch in range(1, CFG["epochs"]+1):
        train_loss = train_one_epoch(enc, dec, train_loader, opt)
        bit_err, psnr = validate(enc, dec, val_loader)
        print(f"Epoch {epoch:02d} | Loss {train_loss:.4f} | BitErr {bit_err:.4f} | PSNR {psnr:.2f}")
        if bit_err < best:
            best = bit_err
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()}, f"{CFG['save_dir']}/best_selfsup.pth")
            print("  *** BEST MODEL SAVED ***")
    print("Done → checkpoints/best_selfsup.pth")

if __name__ == "__main__":
    main()