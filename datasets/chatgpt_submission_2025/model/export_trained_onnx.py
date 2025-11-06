#!/usr/bin/env python3
"""
Export the trained ChatGPT 2025 model to ONNX format.

This matches the exact architecture of the trained model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SteganographyDetector(nn.Module):
    """Exact model architecture matching the trained detector.pth"""
    
    def __init__(self):
        super(SteganographyDetector, self).__init__()
        
        # RGB convolution layers (exactly as in trained model)
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # rgb_conv.0
            nn.ReLU(),                                      # rgb_conv.1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # rgb_conv.3
            nn.ReLU(),                                      # rgb_conv.4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # rgb_conv.6
            nn.ReLU()                                       # rgb_conv.7
        )
        
        # Metadata fully connected layers
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 32),    # meta_fc.0
            nn.ReLU(),            # meta_fc.1
            nn.Linear(32, 64),   # meta_fc.2
            nn.ReLU()             # meta_fc.3
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(6336, 256),  # classifier.0
            nn.ReLU(),             # classifier.1
            nn.Dropout(0.5),       # classifier.2
            nn.Linear(256, 128),   # classifier.3
            nn.ReLU(),             # classifier.4
            nn.Dropout(0.5),       # classifier.5
            nn.Linear(128, 2)      # classifier.6
        )
    
    def forward(self, rgb, metadata):
        # Process RGB through conv layers
        rgb_features = self.rgb_conv(rgb)  # [batch, 128, 224, 224]
        rgb_features = F.adaptive_avg_pool2d(rgb_features, (7, 7))  # [batch, 128, 7, 7]
        rgb_features = rgb_features.view(rgb_features.size(0), -1)  # [batch, 128*7*7]
        
        # Process metadata
        meta_features = self.meta_fc(metadata)  # [batch, 64]
        
        # Concatenate features
        combined = torch.cat([rgb_features, meta_features], dim=1)  # [batch, 6336]
        
        # Classify
        logits = self.classifier(combined)  # [batch, 2]
        
        return logits

def export_to_onnx():
    """Export the trained model to ONNX format"""
    
    # Load the trained model
    print("Loading trained model...")
    model = SteganographyDetector()
    checkpoint = torch.load('detector.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create dummy inputs
    rgb_dummy = torch.randn(1, 3, 224, 224)
    metadata_dummy = torch.randn(1, 3)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (rgb_dummy, metadata_dummy),
        'detector.onnx',
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
    
    print("ONNX export completed successfully!")
    
    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load('detector.onnx')
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except Exception as e:
        print(f"ONNX verification failed: {e}")

if __name__ == "__main__":
    export_to_onnx()