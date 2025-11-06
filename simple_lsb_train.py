# simple_lsb_train.py - Train model to work with LSB steganography
import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2

# LSB functions
def embed_lsb(image, msg_bits):
    """Embed message bits into image using LSB"""
    img = image.copy()
    h, w, c = img.shape
    
    # Flatten image and embed bits sequentially
    flat_img = img.reshape(-1)
    msg_flat = np.array(msg_bits, dtype=np.uint8)
    
    # Embed in LSB of first channel
    for i in range(min(len(msg_flat), len(flat_img)//3)):
        flat_img[i*3] = (flat_img[i*3] & 0xFE) | msg_flat[i]
    
    return flat_img.reshape(h, w, c)

def extract_lsb(image, msg_len):
    """Extract message bits from image using LSB"""
    img = image.copy()
    flat_img = img.reshape(-1)
    
    bits = []
    for i in range(min(msg_len, len(flat_img)//3)):
        bits.append(flat_img[i*3] & 1)
    
    return np.array(bits, dtype=np.float32)

# Simple neural network to enhance LSB
class LSBDetector(nn.Module):
    def __init__(self, msg_len=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, msg_len),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.encoder(x)

# Dataset
class LSBDataset(Dataset):
    def __init__(self, clean_root, num_samples=20):
        self.clean_root = clean_root
        files = [f for f in os.listdir(clean_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files = files[:num_samples]
        
    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.clean_root, self.image_files[idx])
        img = cv2.imread(img_path)
        if img is None:
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate random message
        msg = np.random.randint(0, 2, 100)
        
        # Embed using LSB
        stego = embed_lsb(img, msg)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        stego_tensor = torch.from_numpy(stego.astype(np.float32) / 255.0).permute(2, 0, 1)
        msg_tensor = torch.from_numpy(msg.astype(np.float32))
        
        return img_tensor, stego_tensor, msg_tensor

# Training
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataset = LSBDataset("datasets/val/clean", num_samples=15)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Model
    model = LSBDetector().to(device)
    
    # Optimizer
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for img, stego, msg in loader:
            img, stego, msg = img.to(device), stego.to(device), msg.to(device)
            
            opt.zero_grad()
            
            # Try to extract from both clean and stego
            pred_clean = model(img)
            pred_stego = model(stego)
            
            # Loss: clean should be random, stego should match message
            loss_clean = F.binary_cross_entropy(pred_clean, torch.rand_like(pred_clean))
            loss_stego = F.binary_cross_entropy(pred_stego, msg)
            
            loss = loss_stego + 0.1 * loss_clean
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pred_binary = (pred_stego > 0.5).float()
            correct += (pred_binary == msg).sum().item()
            total += msg.numel()
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Accuracy={accuracy:.4f}")
        
        if accuracy > 0.9:
            break
    
    # Save model
    torch.save(model.state_dict(), 'checkpoints/lsb_detector.pth')
    print("Model saved to checkpoints/lsb_detector.pth")
    
    # Test
    print("\n=== Testing ===")
    model.eval()
    with torch.no_grad():
        img, stego, msg = dataset[0]
        img, stego, msg = img.unsqueeze(0).to(device), stego.unsqueeze(0).to(device), msg.unsqueeze(0).to(device)
        
        pred = model(stego)
        pred_binary = (pred > 0.5).float()
        
        print(f"Message accuracy: {(pred_binary == msg).float().mean().item():.4f}")
        
        # Save test images
        stego_np = (stego.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite("test_stego_lsb.png", cv2.cvtColor(stego_np, cv2.COLOR_RGB2BGR))
        print("Test stego saved as test_stego_lsb.png")

if __name__ == "__main__":
    main()