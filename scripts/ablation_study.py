import os
import copy
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, Any, List, Tuple

# Import starlight utilities
try:
    from starlight_utils import load_unified_input
except ImportError:
    print("Warning: starlight_utils not available")
    load_unified_input = None

# ONNX imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available")

class AblationStudy:
    """Measure impact of removing each stream"""

    def __init__(self, model_path: str):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX runtime required for ablation study")
        self.model = ort.InferenceSession(model_path)
        self.streams = [
            'pixel', 'alpha', 'lsb', 'palette',
            'palette_lsb', 'format', 'content', 'meta'
        ]

    def calculate_fpr(self, test_dataset: str) -> float:
        """Calculate False Positive Rate on clean images"""
        clean_images = list(Path(test_dataset).glob("clean/**/*.png"))
        if not clean_images:
            # Fallback to any png files
            clean_images = list(Path(test_dataset).glob("**/*.png"))[:100]  # Sample

        fp_count = 0
        for img_path in clean_images:
            pred = self.infer(img_path)
            if pred.get('is_steganography', False):
                fp_count += 1
        return (fp_count / len(clean_images)) * 100 if clean_images else 0.0

    def infer(self, img_path: str, zero_streams: List[str] = None) -> Dict[str, Any]:
        """Run inference, optionally zeroing out streams"""
        if not load_unified_input:
            return {"error": "starlight_utils not available"}

        # Load inputs
        pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features = load_unified_input(img_path)

        # Convert to numpy
        lsb_chw = lsb.permute(2, 0, 1) if lsb.dim() == 3 else lsb
        alpha_chw = alpha.unsqueeze(0) if alpha.dim() == 2 else alpha

        inputs = {
            'meta': np.expand_dims(meta.numpy(), 0),
            'alpha': np.expand_dims(alpha_chw.numpy(), 0),
            'lsb': np.expand_dims(lsb_chw.numpy(), 0),
            'palette': np.expand_dims(palette.numpy(), 0),
            'format_features': np.expand_dims(format_features.numpy(), 0),
            'content_features': np.expand_dims(content_features.numpy(), 0),
            'bit_order': np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        }

        # Zero out specified streams
        if zero_streams:
            for stream in zero_streams:
                if stream in inputs:
                    inputs[stream] = np.zeros_like(inputs[stream])

        # Run inference
        try:
            outputs = self.model.run(None, inputs)
            stego_logits = outputs[0]
            method_logits = outputs[1]

            prob = float(1 / (1 + np.exp(-stego_logits[0][0])))
            predicted_method_id = int(np.argmax(method_logits[0]))

            return {
                "stego_probability": prob,
                "predicted_method_id": predicted_method_id,
                "is_steganography": prob > 0.5
            }
        except Exception as e:
            return {"error": str(e)}

    def test_stream_ablation(self, test_dataset: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Systematically remove streams, measure FPR"""
        results = {}

        # Baseline
        full_model_fpr = self.calculate_fpr(test_dataset)
        results['full_model'] = {'fpr': full_model_fpr, 'streams': 8}

        # Ablate each stream
        for stream in self.streams:
            fpr = self.calculate_fpr_with_ablation(test_dataset, [stream])
            impact = fpr - full_model_fpr
            results[f'without_{stream}'] = {'fpr': fpr, 'impact': impact}

        # Sort by impact
        sorted_results = sorted(
            results.items(),
            key=lambda x: abs(x[1].get('impact', 0)),
            reverse=True
        )

        return sorted_results

    def calculate_fpr_with_ablation(self, test_dataset: str, zero_streams: List[str]) -> float:
        """Calculate FPR with specified streams zeroed"""
        clean_images = list(Path(test_dataset).glob("clean/**/*.png"))
        if not clean_images:
            clean_images = list(Path(test_dataset).glob("**/*.png"))[:100]

        fp_count = 0
        for img_path in clean_images:
            pred = self.infer(img_path, zero_streams)
            if pred.get('is_steganography', False):
                fp_count += 1
        return (fp_count / len(clean_images)) * 100 if clean_images else 0.0

def main():
    parser = argparse.ArgumentParser(description='Run ablation study on Starlight model')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--dataset', required=True, help='Path to test dataset')
    parser.add_argument('--output', default='ablation_results.json', help='Output JSON file')

    args = parser.parse_args()

    study = AblationStudy(args.model)
    results = study.test_stream_ablation(args.dataset)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("Ablation study completed. Results saved to", args.output)

if __name__ == '__main__':
    main()