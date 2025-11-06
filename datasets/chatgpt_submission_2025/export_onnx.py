#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX format for Starlight compatibility.
"""

import torch
import torch.onnx
import sys
sys.path.append('/home/eyang/sandbox/starlight')
from trainer import UniversalStegoDetector
import numpy as np

def export_to_onnx():
    """Export UniversalStegoDetector to ONNX format"""
    
    # Load the trained model
    model = UniversalStegoDetector()
    model_path = "model/detector.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    model.eval()
    
    # Create dummy inputs matching the model's expected input format
    batch_size = 1
    
    # RGB images (batch_size, 3, 224, 224)
    rgb_dummy = torch.randn(batch_size, 3, 224, 224)
    
    # Alpha channel (batch_size, 1, 224, 224)
    alpha_dummy = torch.randn(batch_size, 1, 224, 224)
    
    # EXIF features (batch_size, 2)
    exif_dummy = torch.randn(batch_size, 2)
    
    # EOI features (batch_size, 1)
    eoi_dummy = torch.randn(batch_size, 1)
    
    # Palette features (batch_size, 256, 3)
    palette_dummy = torch.randn(batch_size, 256, 3)
    
    # Indices (batch_size, 1, 224, 224)
    indices_dummy = torch.randn(batch_size, 1, 224, 224)
    
    # Input names for ONNX
    input_names = ['rgb', 'alpha', 'exif', 'eoi', 'palette', 'indices']
    
    # Output name
    output_names = ['logits']
    
    # Export path
    onnx_path = "model/detector.onnx"
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            (rgb_dummy, alpha_dummy, exif_dummy, eoi_dummy, palette_dummy, indices_dummy),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'rgb': {0: 'batch_size'},
                'alpha': {0: 'batch_size'},
                'exif': {0: 'batch_size'},
                'eoi': {0: 'batch_size'},
                'palette': {0: 'batch_size'},
                'indices': {0: 'batch_size'},
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
            'alpha': alpha_dummy.numpy(),
            'exif': exif_dummy.numpy(),
            'eoi': eoi_dummy.numpy(),
            'palette': palette_dummy.numpy(),
            'indices': indices_dummy.numpy()
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