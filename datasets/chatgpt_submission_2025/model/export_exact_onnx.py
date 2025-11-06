#!/usr/bin/env python3
"""
Create exact model architecture to match trained state dict
"""

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

def export_to_onnx():
    """Export the trained model to ONNX format"""
    
    # Load the trained model
    print("Loading trained model...")
    model = SteganographyDetector()
    checkpoint = torch.load('detector.pth', map_location='cpu')
    
    # Load with custom mapping
    model.load_custom_state_dict(checkpoint)
    model.eval()
    
    # Create dummy inputs
    rgb_dummy = torch.randn(1, 3, 224, 224)
    metadata_dummy = torch.randn(1, 3)
    
    # Test forward pass
    with torch.no_grad():
        output = model(rgb_dummy, metadata_dummy)
        print(f"Model output shape: {output.shape}")
    
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