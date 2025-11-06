#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX format for Starlight compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import sys
import os

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
    """Export SteganographyDetector to ONNX format"""
    
    # Load the trained model
    model = SteganographyDetector()
    model_path = "model/detector.pth"
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_custom_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    model.eval()
    
    # Create dummy inputs matching the model's expected input format
    batch_size = 1
    
    # RGB images (batch_size, 3, 224, 224)
    rgb_dummy = torch.randn(batch_size, 3, 224, 224)
    
    # Metadata (batch_size, 3) - combining exif, eoi, and one other feature
    metadata_dummy = torch.randn(batch_size, 3)
    
    # Input names for ONNX
    input_names = ['rgb', 'metadata']
    
    # Output name
    output_names = ['logits']
    
    # Export path
    onnx_path = "model/detector.onnx"
    
    try:
        # Test forward pass first
        with torch.no_grad():
            output = model(rgb_dummy, metadata_dummy)
            print(f"Model output shape: {output.shape}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (rgb_dummy, metadata_dummy),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'rgb': {0: 'batch_size'},
                'metadata': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to {onnx_path}")
        
        # Verify the exported model
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print("✅ ONNX model verification passed")
        
        # Test with ONNX Runtime
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path)
        
        # Create test inputs
        test_inputs = {
            'rgb': rgb_dummy.numpy(),
            'metadata': metadata_dummy.numpy()
        }
        
        # Run inference
        outputs = ort_session.run(None, test_inputs)
        print(f"✅ ONNX Runtime test passed - Output shape: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return False

if __name__ == "__main__":
    success = export_to_onnx()
    if success:
        print("🎉 ONNX export completed successfully!")
    else:
        print("❌ ONNX export failed!")