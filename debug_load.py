#!/usr/bin/env python3
# debug_load.py - Debug the load.py issues
import os
import torch
import numpy as np
import cv2
from train_stego import Encoder, Decoder, CFG

print("=== Debug Load Issues ===")

# Load models
print("Loading models...")
enc = Encoder(CFG["msg_len"]).cuda().eval()
dec = Decoder(CFG["msg_len"]).cuda().eval()

# Check if checkpoint exists
if os.path.exists("checkpoints/best.pth"):
    print("Loading checkpoint...")
    ckpt = torch.load("checkpoints/best.pth")
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    print("Checkpoint loaded successfully")
else:
    print("No checkpoint found!")
    exit(1)

# Test with simple data
print("\n=== Testing with synthetic data ===")
# Create synthetic cover image
cover = torch.rand(1, 3, 256, 256).cuda()  # Random image
msg = torch.randint(0, 2, (1, CFG["msg_len"])).float().cuda()

print(f"Cover shape: {cover.shape}")
print(f"Message shape: {msg.shape}")
print(f"Message sample: {msg[0, :10]}")

# Forward pass
with torch.no_grad():
    stego, residual = enc(cover, msg)
    pred_msg = dec(stego)
    
print(f"Stego shape: {stego.shape}")
print(f"Residual shape: {residual.shape}")
print(f"Predicted message shape: {pred_msg.shape}")
print(f"Predicted message sample: {pred_msg[0, :10]}")

# Calculate accuracy
pred_binary = (pred_msg > 0.5).float()
accuracy = (pred_binary == msg).float().mean()
print(f"Message accuracy: {accuracy.item():.4f}")

# Test with real image
print("\n=== Testing with real image ===")
cover_path = "datasets/val/clean/clean-0405.png"
if os.path.exists(cover_path):
    print(f"Loading real image: {cover_path}")
    cover_img = cv2.imread(cover_path)
    if cover_img is not None:
        cover_real = cover_img[..., ::-1].astype(np.float32) / 255.0
        cover_real = torch.from_numpy(cover_real).permute(2,0,1).unsqueeze(0).cuda()
        
        # Generate random message
        msg_real = torch.randint(0, 2, (1, CFG["msg_len"])).float().cuda()
        
        with torch.no_grad():
            stego_real, _ = enc(cover_real, msg_real)
            pred_real = dec(stego_real)
            
        pred_binary_real = (pred_real > 0.5).float()
        accuracy_real = (pred_binary_real == msg_real).float().mean()
        
        print(f"Real image message accuracy: {accuracy_real.item():.4f}")
        
        # Save stego image
        stego_np = (stego_real.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        cv2.imwrite("debug_stego.png", stego_np[..., ::-1])
        print("Debug stego image saved as debug_stego.png")
        
        # Test extraction from saved image
        stego_loaded = cv2.imread("debug_stego.png")[..., ::-1].astype(np.float32) / 255.0
        stego_loaded = torch.from_numpy(stego_loaded).permute(2,0,1).unsqueeze(0).cuda()
        
        with torch.no_grad():
            pred_loaded = dec(stego_loaded)
            
        pred_binary_loaded = (pred_loaded > 0.5).float()
        accuracy_loaded = (pred_binary_loaded == msg_real).float().mean()
        
        print(f"Loaded image message accuracy: {accuracy_loaded.item():.4f}")
        print(f"Original vs Loaded match: {torch.allclose(pred_real, pred_loaded, atol=1e-6)}")
        
    else:
        print("Failed to load real image")
else:
    print("Real image not found")

print("\n=== Debug Complete ===")