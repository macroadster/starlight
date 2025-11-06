#!/usr/bin/env python3
"""
Project Starlight - Standardized Inference Interface
Compatible with aggregation pipeline

Author: Claude (Anthropic)
Date: 2025
License: MIT
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import os


class StarlightModel:
    """Standardized model interface for Project Starlight"""
    
    def __init__(self, model_path="model/detector.onnx", task="detect"):
        """
        Initialize the model
        
        Args:
            model_path: Path to ONNX model file
            task: 'detect' for detection, 'extract' for extraction
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.session = ort.InferenceSession(model_path)
        self.task = task
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def preprocess(self, img_path):
        """
        Preprocess image for model input
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array (1, 4, 256, 256) - always RGBA
        """
        # Load image
        img = Image.open(img_path)
        
        # Convert to RGBA (add dummy alpha if needed)
        if img.mode in ['RGBA', 'LA']:
            img = img.convert('RGBA')
        else:
            # Convert to RGB first, then add alpha channel
            img = img.convert('RGB')
            # Add fully opaque alpha channel
            alpha = Image.new('L', img.size, 255)
            img.putalpha(alpha)
        
        # Resize to 256x256
        img = img.resize((256, 256), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        arr = np.array(img).astype(np.float32) / 255.0
        
        # Transpose to CHW format (4 channels always)
        arr = np.transpose(arr, (2, 0, 1))
        
        # Add batch dimension
        arr = np.expand_dims(arr, 0)
        
        return arr
    
    def predict(self, img_path):
        """
        Run inference on an image
        
        Args:
            img_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        input_data = self.preprocess(img_path)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        
        if self.task == "detect":
            # Binary classification: clean vs stego
            prob = float(outputs[0][0][0])  # Sigmoid output
            
            return {
                "stego_probability": prob,
                "predicted": "stego" if prob > 0.5 else "clean",
                "confidence": prob if prob > 0.5 else 1 - prob
            }
        else:
            # Extraction not implemented in this detector
            # Would need separate extractor model
            return {
                "error": "Extraction not supported by this model",
                "extracted_payload": None
            }
    
    def predict_batch(self, img_paths):
        """
        Run inference on multiple images
        
        Args:
            img_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in img_paths:
            try:
                result = self.predict(img_path)
                result['image'] = img_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image': img_path,
                    'error': str(e),
                    'stego_probability': None,
                    'predicted': None
                })
        
        return results


def test_inference():
    """Test the inference pipeline"""
    import sys
    
    print("="*60)
    print("Testing Starlight Inference Pipeline")
    print("="*60)
    
    # Check if model exists
    model_path = "model/detector.onnx"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using train_detector.py")
        return False
    
    # Initialize model
    print(f"\nLoading model from {model_path}...")
    try:
        model = StarlightModel(model_path, task="detect")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test on sample images
    test_dirs = {
        'clean': 'clean',
        'stego': 'stego'
    }
    
    for label, dir_path in test_dirs.items():
        if not os.path.exists(dir_path):
            print(f"\nSkipping {label} - directory not found: {dir_path}")
            continue
        
        # Get first few images
        images = []
        for ext in ['*.png', '*.bmp']:
            images.extend(list(Path(dir_path).glob(ext)))
        
        if not images:
            print(f"\nNo images found in {dir_path}")
            continue
        
        images = images[:3]  # Test on first 3
        
        print(f"\nTesting on {label} images:")
        for img_path in images:
            try:
                result = model.predict(str(img_path))
                print(f"  {img_path.name}")
                print(f"    → Prediction: {result['predicted']}")
                print(f"    → Confidence: {result['confidence']:.4f}")
                print(f"    → Stego prob: {result['stego_probability']:.4f}")
            except Exception as e:
                print(f"  {img_path.name}: ERROR - {e}")
    
    print("\n" + "="*60)
    print("Inference test complete")
    print("="*60)
    
    return True


if __name__ == "__main__":
    from pathlib import Path
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Starlight detector inference')
    parser.add_argument('--test', action='store_true', help='Run test inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, default='model/detector.onnx', help='Path to ONNX model')
    
    args = parser.parse_args()
    
    if args.test:
        test_inference()
    elif args.image:
        model = StarlightModel(args.model, task="detect")
        result = model.predict(args.image)
        print(f"\nImage: {args.image}")
        print(f"Prediction: {result['predicted']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Stego probability: {result['stego_probability']:.4f}")
    elif args.batch:
        model = StarlightModel(args.model, task="detect")
        images = list(Path(args.batch).glob('*.png')) + list(Path(args.batch).glob('*.bmp'))
        results = model.predict_batch([str(p) for p in images])
        
        print(f"\nBatch inference on {len(results)} images:")
        for r in results:
            print(f"{r['image']}: {r['predicted']} (conf={r.get('confidence', 0):.4f})")
    else:
        print("Use --test, --image <path>, or --batch <dir>")
