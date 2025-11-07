#!/usr/bin/env python3
"""
export_to_onnx.py - Export trained PyTorch models to ONNX format
"""

import torch
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add current directory to import modules
sys.path.append(".")

from train import Encoder, Decoder, Critic, CFG


def export_detector_to_onnx(model_path, onnx_path, input_shape=(1, 3, 256, 256)):
    """
    Export critic (detector) model to ONNX format.

    Args:
        model_path (str): Path to PyTorch .pth file
        onnx_path (str): Output path for ONNX file
        input_shape (tuple): Input tensor shape (batch, channels, height, width)
    """
    print(f"Loading detector model from {model_path}...")

    # Initialize detector model
    model = Critic().to(CFG["device"])

    # Load weights
    checkpoint = torch.load(model_path, map_location=CFG["device"])
    if isinstance(checkpoint, dict) and "crit" in checkpoint:
        model.load_state_dict(checkpoint["crit"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and ("enc" in checkpoint and "dec" in checkpoint):
        # This is an encoder-decoder checkpoint, not a critic
        print(
            "Warning: This checkpoint contains encoder/decoder, not a critic detector"
        )
        print("Skipping this model for detector export")
        return
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Creating dummy input...")
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(CFG["device"])

    print(f"Exporting detector to ONNX format: {onnx_path}")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("ONNX export completed!")

    # Verify ONNX model
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")

        # Print model info
        print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
        print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")

    except ImportError:
        print("Warning: onnx package not installed. Cannot verify exported model.")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")


def export_encoder_decoder_to_onnx(
    model_path, onnx_path_detector, onnx_path_extractor, input_shape=(1, 3, 256, 256)
):
    """
    Export encoder-decoder models to ONNX format for detection and extraction.

    Args:
        model_path (str): Path to PyTorch .pth file containing encoder/decoder
        onnx_path_detector (str): Output path for detector ONNX file
        onnx_path_extractor (str): Output path for extractor ONNX file
        input_shape (tuple): Input tensor shape (batch, channels, height, width)
    """
    print(f"Loading encoder-decoder model from {model_path}...")

    # Initialize models
    encoder = Encoder(CFG["msg_len"]).to(CFG["device"])
    decoder = Decoder(CFG["msg_len"]).to(CFG["device"])
    critic = Critic().to(CFG["device"])

    # Load weights
    checkpoint = torch.load(model_path, map_location=CFG["device"])
    if isinstance(checkpoint, dict) and "enc" in checkpoint and "dec" in checkpoint:
        encoder.load_state_dict(checkpoint["enc"])
        decoder.load_state_dict(checkpoint["dec"])
    else:
        print("Warning: Could not find encoder/decoder in checkpoint")
        return

    encoder.eval()
    decoder.eval()
    critic.eval()

    print("Creating dummy input...")
    dummy_input = torch.randn(input_shape).to(CFG["device"])

    # Export detector (critic)
    print(f"Exporting detector to ONNX format: {onnx_path_detector}")
    torch.onnx.export(
        critic,
        (dummy_input,),
        onnx_path_detector,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Export extractor (decoder)
    print(f"Exporting extractor to ONNX format: {onnx_path_extractor}")
    torch.onnx.export(
        decoder,
        (dummy_input,),
        onnx_path_extractor,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("ONNX export completed!")

    # Verify ONNX models
    try:
        import onnx

        for onnx_path in [onnx_path_detector, onnx_path_extractor]:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"✓ ONNX model verification passed: {onnx_path}")
    except ImportError:
        print("Warning: onnx package not installed. Cannot verify exported model.")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")


def main():
    """Main export function."""
    # Model files to export
    models_to_export = [
        {
            "pth_path": "checkpoints/best.pth",
            "onnx_detector": "model/detector.onnx",
            "onnx_extractor": "model/extractor.onnx",
            "description": "LSB steganography detector and extractor",
        }
    ]

    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    for model_info in models_to_export:
        if os.path.exists(model_info["pth_path"]):
            print(f"\n=== Exporting {model_info['description']} ===")
            export_encoder_decoder_to_onnx(
                model_info["pth_path"],
                model_info["onnx_detector"],
                model_info["onnx_extractor"],
            )
        else:
            print(f"Warning: Model file not found: {model_info['pth_path']}")

    print("\n=== Export Complete ===")
    print("ONNX models saved in 'model/' directory")
    print("You can now use these models with inference.py")


if __name__ == "__main__":
    main()
