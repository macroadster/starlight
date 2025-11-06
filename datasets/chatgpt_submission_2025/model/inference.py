#!/usr/bin/env python3
"""
Standardized inference wrapper for ChatGPT Submission 2025 Steganography Detector

This follows the Starlight project specification for model contributions.
Compatible with the ensemble aggregation system.
"""

import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import json

class StarlightModel:
    """
    Standardized steganography detection model for Starlight project.
    
    Provides binary classification: clean vs stego
    Input: RGB image (any format, auto-converted to 224x224)
    Output: Steganography probability [0,1]
    """
    
    def __init__(self, model_path="model/detector.onnx"):
        """
        Initialize the model.
        
        Args:
            model_path: Path to ONNX model file
        """
        self.session = ort.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Get input shapes from model
        self.rgb_shape = None
        self.meta_shape = None
        
        for inp in self.session.get_inputs():
            if inp.name == 'rgb':
                self.rgb_shape = inp.shape
            elif inp.name == 'metadata':
                self.meta_shape = inp.shape
    
    def preprocess(self, img_path):
        """
        Preprocess image for model inference.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Tuple of (rgb_tensor, metadata_tensor)
        """
        try:
            img = Image.open(img_path)
            
            # Convert RGB and resize
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy and normalize to [0,1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Convert to CHW format and add batch dimension
            rgb_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]
            
            # Extract metadata features
            metadata = self._extract_metadata_features(img_path)
            metadata_tensor = np.array(metadata, dtype=np.float32)[np.newaxis, ...]
            
            return rgb_tensor, metadata_tensor
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image {img_path}: {e}")
    
    def _extract_metadata_features(self, img_path):
        """
        Extract metadata features for steganography detection.
        
        Args:
            img_path: Path to image file
            
        Returns:
            List of 3 metadata features [exif_present, exif_length, eoi_length]
        """
        features = [0.0, 0.0, 0.0]  # [exif_present, exif_length, eoi_length]
        
        try:
            # EXIF features
            img = Image.open(img_path)
            exif_bytes = img.info.get('exif')
            if exif_bytes:
                features[0] = 1.0  # exif_present
                features[1] = min(len(exif_bytes) / 1000.0, 10.0)  # exif_length (normalized)
            
            # EOI (End of Image) features for JPEG
            if img_path.lower().endswith(('.jpg', '.jpeg')):
                try:
                    with open(img_path, 'rb') as f:
                        data = f.read()
                    eoi_pos = data.rfind(b'\xff\xd9')
                    if eoi_pos >= 0 and eoi_pos + 2 < len(data):
                        payload_len = len(data) - (eoi_pos + 2)
                        features[2] = min(payload_len / 1000.0, 10.0)  # eoi_length (normalized)
                except:
                    pass
        except:
            pass
        
        return features
    
    def predict(self, img_path):
        """
        Predict steganography probability for an image.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Dictionary with prediction results:
            - stego_probability: float [0,1]
            - predicted: bool (True if stego, False if clean)
            - confidence: float (max probability)
        """
        try:
            # Preprocess
            rgb_tensor, metadata_tensor = self.preprocess(img_path)
            
            # Prepare inputs for ONNX Runtime
            inputs = {
                'rgb': rgb_tensor.astype(np.float32),
                'metadata': metadata_tensor.astype(np.float32)
            }
            
            # Run inference
            outputs = self.session.run(self.output_names, inputs)
            logits = outputs[0]  # Get logits
            
            # Apply softmax to get probabilities
            logits_array = np.array(logits).flatten()  # Flatten to 1D
            exp_logits = np.exp(logits_array - np.max(logits_array))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Get stego probability (index 1)
            stego_prob = float(probabilities[1])
            predicted = stego_prob > 0.5
            confidence = float(np.max(probabilities))
            
            return {
                "stego_probability": stego_prob,
                "predicted": predicted,
                "confidence": confidence,
                "probabilities": {
                    "clean": float(probabilities[0]),
                    "stego": float(probabilities[1])
                }
            }
            
        except Exception as e:
            return {
                "stego_probability": 0.0,
                "predicted": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def batch_predict(self, img_paths):
        """
        Predict steganography probability for multiple images.
        
        Args:
            img_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in img_paths:
            result = self.predict(img_path)
            results.append(result)
        return results

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model_path = os.path.join(os.path.dirname(__file__), 'detector.onnx')
    model = StarlightModel(model_path)
    
    # Test on a sample image (if available)
    test_image = "../clean/sample_seed_alpha_000.png"  # Adjust path as needed
    
    if os.path.exists(test_image):
        print(f"Testing on {test_image}")
        result = model.predict(test_image)
        
        print("Prediction Results:")
        print(f"  Steganography Probability: {result['stego_probability']:.4f}")
        print(f"  Predicted: {'STEGO' if result['predicted'] else 'CLEAN'}")
        print(f"  Confidence: {result['confidence']:.4f}")
        if 'probabilities' in result:
            print(f"  Probabilities: Clean={result['probabilities']['clean']:.4f}, Stego={result['probabilities']['stego']:.4f}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    else:
        print(f"Test image {test_image} not found")
        print("Model initialized successfully. Ready for inference.")