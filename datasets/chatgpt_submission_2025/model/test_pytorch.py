#!/usr/bin/env python3
"""
Test PyTorch model directly to verify it's working correctly
"""

import torch
import torch.nn.functional as F
# Import the model class directly
import torch
import torch.nn as nn
import torch.nn.functional as F

class SteganographyDetector(nn.Module):
    """Exact model architecture matching the trained detector.pth"""
    
    def __init__(self):
        super(SteganographyDetector, self).__init__()
        
        # Create individual layers to match state dict exactly
        self.rgb_conv_0 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.rgb_conv_1 = nn.ReLU()
        self.rgb_conv_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.rgb_conv_4 = nn.ReLU()
        self.rgb_conv_6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.rgb_conv_7 = nn.ReLU()
        
        # Metadata layers
        self.meta_fc_0 = nn.Linear(3, 32)
        self.meta_fc_1 = nn.ReLU()
        self.meta_fc_2 = nn.Linear(32, 64)
        self.meta_fc_3 = nn.ReLU()
        
        # Classifier layers
        self.classifier_0 = nn.Linear(6336, 256)
        self.classifier_1 = nn.ReLU()
        self.classifier_2 = nn.Dropout(0.5)
        self.classifier_3 = nn.Linear(256, 128)
        self.classifier_4 = nn.ReLU()
        self.classifier_5 = nn.Dropout(0.5)
        self.classifier_6 = nn.Linear(128, 2)
    
    def forward(self, rgb, metadata):
        # RGB forward pass
        x = self.rgb_conv_0(rgb)
        x = self.rgb_conv_1(x)
        x = self.rgb_conv_3(x)
        x = self.rgb_conv_4(x)
        x = self.rgb_conv_6(x)
        x = self.rgb_conv_7(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)
        
        # Metadata forward pass
        m = self.meta_fc_0(metadata)
        m = self.meta_fc_1(m)
        m = self.meta_fc_2(m)
        m = self.meta_fc_3(m)
        
        # Concatenate
        combined = torch.cat([x, m], dim=1)
        
        # Classifier forward pass
        out = self.classifier_0(combined)
        out = self.classifier_1(out)
        out = self.classifier_2(out)
        out = self.classifier_3(out)
        out = self.classifier_4(out)
        out = self.classifier_5(out)
        out = self.classifier_6(out)
        
        return out
    
    def load_custom_state_dict(self, state_dict):
        """Load state dict with custom mapping"""
        # Map the state dict keys to our layer names
        mapping = {}
        for key, value in state_dict.items():
            if key.startswith('rgb_conv.'):
                new_key = key.replace('rgb_conv.', 'rgb_conv_')
                mapping[new_key] = value
            elif key.startswith('meta_fc.'):
                new_key = key.replace('meta_fc.', 'meta_fc_')
                mapping[new_key] = value
            elif key.startswith('classifier.'):
                new_key = key.replace('classifier.', 'classifier_')
                mapping[new_key] = value
            else:
                mapping[key] = value
        
        return self.load_state_dict(mapping, strict=False)
from PIL import Image
import numpy as np
import os

def preprocess_image(img_path):
    """Preprocess image same as inference"""
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).unsqueeze(0)
    return img_tensor

def extract_metadata(img_path):
    """Extract metadata features"""
    features = [0.0, 0.0, 0.0]
    try:
        img = Image.open(img_path)
        exif_bytes = img.info.get('exif')
        if exif_bytes:
            features[0] = 1.0
            features[1] = min(len(exif_bytes) / 1000.0, 10.0)
        
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            with open(img_path, 'rb') as f:
                data = f.read()
            eoi_pos = data.rfind(b'\xff\xd9')
            if eoi_pos >= 0 and eoi_pos + 2 < len(data):
                payload_len = len(data) - (eoi_pos + 2)
                features[2] = min(payload_len / 1000.0, 10.0)
    except:
        pass
    return torch.tensor([features], dtype=torch.float32)

def test_pytorch_model():
    """Test PyTorch model directly"""
    print("Loading PyTorch model...")
    model = SteganographyDetector()
    checkpoint = torch.load('detector.pth', map_location='cpu')
    model.load_custom_state_dict(checkpoint)
    model.eval()
    
    test_images = [
        '../clean/sample_seed_alpha_000.png',
        '../stego/sample_seed_alpha_020.png'
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting {os.path.basename(img_path)}:")
            
            # Preprocess
            rgb_tensor = preprocess_image(img_path)
            metadata_tensor = extract_metadata(img_path)
            
            print(f"  RGB tensor shape: {rgb_tensor.shape}")
            print(f"  Metadata: {metadata_tensor.tolist()}")
            
            # Predict
            with torch.no_grad():
                logits = model(rgb_tensor, metadata_tensor)
                probs = F.softmax(logits, dim=1)
                stego_prob = probs[0][1].item()
                predicted = stego_prob > 0.5
                
                print(f"  Logits: {logits.tolist()}")
                print(f"  Probabilities: {probs.tolist()}")
                print(f"  Stego probability: {stego_prob:.4f}")
                print(f"  Predicted: {'STEGO' if predicted else 'CLEAN'}")
        else:
            print(f"{img_path} not found")

if __name__ == "__main__":
    test_pytorch_model()