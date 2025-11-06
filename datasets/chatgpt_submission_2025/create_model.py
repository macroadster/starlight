#!/usr/bin/env python3
"""
Create and export a simple detector model for ChatGPT submission.
"""

import torch
import torch.nn as nn
import torch.onnx
import numpy as np

class SimpleStegoDetector(nn.Module):
    """Simple CNN-based steganography detector"""
    
    def __init__(self, num_classes=2):  # Binary: clean vs stego
        super(SimpleStegoDetector, self).__init__()
        
        # CNN for RGB input
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7)
        )
        
        # Feature extractor for metadata
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 32),  # exif_present, exif_len, eoi_len
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, rgb, metadata):
        # RGB features
        rgb_features = self.rgb_conv(rgb)
        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        
        # Metadata features
        meta_features = self.meta_fc(metadata)
        
        # Combine and classify
        combined = torch.cat([rgb_features, meta_features], dim=1)
        return self.classifier(combined)

def create_and_export():
    """Create and export model to ONNX"""
    
    # Initialize model
    model = SimpleStegoDetector()
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    rgb_dummy = torch.randn(batch_size, 3, 224, 224)
    metadata_dummy = torch.randn(batch_size, 3)  # [exif_present, exif_len, eoi_len]
    
    # Export to ONNX
    onnx_path = "model/detector.onnx"
    
    try:
        torch.onnx.export(
            model,
            (rgb_dummy, metadata_dummy),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['rgb', 'metadata'],
            output_names=['logits'],
            dynamic_axes={
                'rgb': {0: 'batch_size'},
                'metadata': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Model exported to {onnx_path}")
        
        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        
        # Save PyTorch version too
        torch.save(model.state_dict(), "model/detector.pth")
        print("✅ PyTorch model saved to model/detector.pth")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = create_and_export()
    if success:
        print("🎉 Model creation and export completed!")
    else:
        print("❌ Model creation failed!")