# quick_train.py - Fast training for working steganography model
import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2

# Simple dataset
class SimpleDataset(Dataset):
    def __init__(self, clean_root, num_samples=50):
        self.clean_root = clean_root
        files = [f for f in os.listdir(clean_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files = files[:num_samples]  # Limit for fast training
        
    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.clean_root, self.image_files[idx])
        img = cv2.imread(img_path)
        if img is None:
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))  # Resize to fixed size
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

# Simple model
class SimpleEncoder(nn.Module):
    def __init__(self, msg_len=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)
        self.msg_fc = nn.Linear(msg_len, 64*64)
        
    def forward(self, cover, msg):
        b, c, h, w = cover.shape
        msg_feat = self.msg_fc(msg).view(b, 1, 64, 64)
        msg_feat = F.interpolate(msg_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        x = F.relu(self.conv1(cover))
        x = F.relu(self.conv2(x))
        residual = self.conv3(x) * 0.05  # Smaller modification
        
        # Stronger message influence
        msg_influence = msg_feat * 0.2
        residual = residual + msg_influence
        
        stego = torch.clamp(cover + residual, 0, 1)
        return stego, residual

class SimpleDecoder(nn.Module):
    def __init__(self, msg_len=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, msg_len)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

# Training
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataset = SimpleDataset("datasets/val/clean", num_samples=20)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Models
    enc = SimpleEncoder().to(device)
    dec = SimpleDecoder().to(device)
    
    # Optimizer
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    
    # Train for few epochs
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for cover in loader:
            cover = cover.to(device)
            msg = torch.randint(0, 2, (cover.size(0), 100)).float().to(device)
            
            opt.zero_grad()
            stego, _ = enc(cover, msg)
            pred = dec(stego)
            
            # Loss
            msg_loss = F.binary_cross_entropy(pred, msg)
            img_loss = F.mse_loss(stego, cover)
            loss = 10.0 * msg_loss + 0.1 * img_loss  # Emphasize message recovery
            
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pred_binary = (pred > 0.5).float()
            correct += (pred_binary == msg).sum().item()
            total += msg.numel()
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Accuracy={accuracy:.4f}")
        
        if accuracy > 0.95:  # Stop early if good enough
            break
    
    # Save models
    torch.save({
        'enc': enc.state_dict(),
        'dec': dec.state_dict()
    }, 'checkpoints/quick_model.pth')
    print("Model saved to checkpoints/quick_model.pth")

if __name__ == "__main__":
    main()