#!/usr/bin/env python3
"""
Export trained ChatGPT model to ONNX format.
Matches the actual trained model architecture.
"""

import torch
import torch.onnx
import torch.nn as nn
import sys

sys.path.append("/home/eyang/sandbox/starlight")


class ChatGPTStegoDetector(nn.Module):
    """Simple CNN model matching our trained architecture"""

    def __init__(self, num_classes=2):
        super().__init__()

        # RGB CNN backbone
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Metadata processing
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, rgb, metadata):
        # Process RGB
        rgb_features = self.rgb_conv(rgb)
        rgb_features = rgb_features.view(rgb_features.size(0), -1)

        # Process metadata
        meta_features = self.meta_fc(metadata)

        # Concatenate and classify
        combined = torch.cat([rgb_features, meta_features], dim=1)
        logits = self.classifier(combined)

        return logits


def export_to_onnx():
    """Export ChatGPT model to ONNX format"""

    # Load the trained model
    model = ChatGPTStegoDetector()
    model_path = "detector.pth"

    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"✅ Loaded trained model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    model.eval()

    # Create dummy inputs
    batch_size = 1
    rgb_dummy = torch.randn(batch_size, 3, 224, 224)
    metadata_dummy = torch.randn(batch_size, 3)

    # Input/output names
    input_names = ["rgb", "metadata"]
    output_names = ["logits"]

    # Export path
    onnx_path = "detector.onnx"

    try:
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
                "rgb": {0: "batch_size"},
                "metadata": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        print(f"✅ Model exported to {onnx_path}")

        # Verify exported model
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")

        # Test with ONNX Runtime
        import onnxruntime as ort

        ort_session = ort.InferenceSession(onnx_path)

        # Create test inputs
        test_inputs = {"rgb": rgb_dummy.numpy(), "metadata": metadata_dummy.numpy()}

        # Run inference
        outputs = ort_session.run(None, test_inputs)
        print(f"✅ ONNX Runtime test passed - Output shape: {outputs[0].shape}")

        return True

    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")
        return False


if __name__ == "__main__":
    success = export_to_onnx()
    if success:
        print("🎉 ONNX export completed successfully!")
    else:
        print("❌ ONNX export failed!")
