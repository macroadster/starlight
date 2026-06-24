#!/usr/bin/env python3
"""Quantize a trained Starlight .pth model to INT8 and FP16 variants.

Usage:
    python scripts/quantize_model.py --input models/detector_balanced.pth
    python scripts/quantize_model.py --input models/detector_balanced.pth --output-dir models_quantized
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant

# Add project root to path so we can import the model class
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import BalancedStarlightDetector


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def count_params(state_dict):
    return sum(p.numel() for p in state_dict.values() if p.is_floating_point())


def export_fp16(model, output_path):
    """Export model weights in FP16 (half precision)."""
    state_dict = model.state_dict()
    fp16_state = {}
    for k, v in state_dict.items():
        if v.is_floating_point():
            fp16_state[k] = v.half()
        else:
            fp16_state[k] = v
    torch.save(fp16_state, output_path)
    print(f"  FP16 model saved to {output_path}")
    return output_path


def export_int8_weights(model, output_path):
    """Export model with INT8 weight quantization (manual, platform-independent).

    Stores weights as int8 + scale/zero_point per tensor, with non-weight
    tensors kept at original precision. Compatible with all platforms.
    """
    state_dict = model.state_dict()
    quantized_state = {}

    for k, v in state_dict.items():
        # Only quantize weight tensors (not biases, BN params, etc.)
        if v.is_floating_point() and "weight" in k and v.dim() >= 2:
            # Per-tensor symmetric quantization
            abs_max = v.abs().max().clamp(min=1e-8)
            scale = abs_max / 127.0
            v_int8 = (v / scale).round().clamp(-128, 127).to(torch.int8)
            quantized_state[k] = v_int8
            quantized_state[k + ".__scale__"] = scale
        else:
            quantized_state[k] = v

    torch.save(quantized_state, output_path)
    print(f"  INT8 (weight-quantized) model saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Quantize Starlight model to INT8/FP16")
    parser.add_argument(
        "--input", default="models/detector_balanced.pth",
        help="Path to input .pth model file"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same as input)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input model not found: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(args.input)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    fp16_path = os.path.join(output_dir, f"{base_name}_fp16.pth")
    int8_path = os.path.join(output_dir, f"{base_name}_int8.pth")

    # Load the original model
    print(f"Loading model from {args.input}...")
    model = BalancedStarlightDetector(meta_weight=0.3)
    state_dict = torch.load(args.input, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    original_size = get_file_size_mb(args.input)
    original_params = count_params(state_dict)
    print(f"  Original: {original_size:.2f} MB, {original_params:,} params")

    # FP16 export
    print("\nExporting FP16 variant...")
    export_fp16(model, fp16_path)
    fp16_size = get_file_size_mb(fp16_path)

    # INT8 weight quantization
    print("\nExporting INT8 (weight-quantized) variant...")
    export_int8_weights(model, int8_path)
    int8_size = get_file_size_mb(int8_path)

    # Summary
    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<20} {'Size (MB)':<12} {'Reduction':<12}")
    print("-" * 44)
    print(f"{'Original (FP32)':<20} {original_size:<12.2f} {'--':<12}")
    print(f"{'FP16':<20} {fp16_size:<12.2f} {(1 - fp16_size/original_size)*100:<11.1f}%")
    print(f"{'INT8 (dynamic)':<20} {int8_size:<12.2f} {(1 - int8_size/original_size)*100:<11.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
