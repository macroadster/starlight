# inference.py
import numpy as np
from PIL import Image
import os
import sys
import json
from typing import Dict, Any

# Optional ONNX imports
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Neural network features disabled.")

# Optional Hugging Face imports
try:
    from transformers import Pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Hugging Face transformers not available. Pipeline features disabled.")

# Add scripts directory to import utilities
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Import unified input loader
try:
    from starlight_utils import load_unified_input
except ImportError as e:
    print(f"Warning: Could not import starlight_utils: {e}")
    load_unified_input = None


class StarlightModel:
    def __init__(
        self,
        detector_path: str = "model/detector.onnx",
        task: str = "detect"
    ):
        self.detector_path = detector_path
        self.task = task

        # Load ONNX model
        if ONNX_AVAILABLE:
            providers = []
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            if 'CoreMLExecutionProvider' in available_providers:
                providers.append('CoreMLExecutionProvider')
            providers.append('CPUExecutionProvider')

            session_options = ort.SessionOptions()
            if 'CUDAExecutionProvider' in providers:
                session_options.enable_mem_pattern = False
            elif 'CoreMLExecutionProvider' in providers:
                session_options.enable_mem_pattern = False

            if os.path.exists(detector_path):
                try:
                    self.detector = ort.InferenceSession(detector_path, sess_options=session_options, providers=providers)
                except Exception as e:
                    print(f"Warning: Could not load detector: {e}")
                    self.detector = None
            else:
                print(f"Warning: Detector model not found at {detector_path}")
                self.detector = None
        else:
            self.detector = None

    def _detect_method_from_filename(self, img_path: str) -> str:
        basename = os.path.basename(img_path)
        parts = basename.split("_")
        if len(parts) >= 3:
            method = parts[-2]  # e.g., alpha, eoi, dct
            return method
        return "lsb"  # Default fallback

    def predict(self, img_path: str, method: str = None) -> Dict[str, Any]:
        if not load_unified_input:
            return {"error": "starlight_utils not available"}

        # Use unified input loader (aligned with scanner.py design)
        pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features = load_unified_input(img_path)

        # Convert to numpy for ONNX and add batch dimension
        # Note: lsb and alpha need to be in CHW format for ONNX
        lsb_chw = lsb.permute(2, 0, 1) if lsb.dim() == 3 else lsb  # (3, 256, 256)
        alpha_chw = alpha.unsqueeze(0) if alpha.dim() == 2 else alpha  # (1, 256, 256)

        inputs = {
            'meta': np.expand_dims(meta.numpy(), 0),
            'alpha': np.expand_dims(alpha_chw.numpy(), 0),
            'lsb': np.expand_dims(lsb_chw.numpy(), 0),
            'palette': np.expand_dims(palette.numpy(), 0),
            'format_features': np.expand_dims(format_features.numpy(), 0),
            'content_features': np.expand_dims(content_features.numpy(), 0),
            'bit_order': np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # Default msb-first
        }

        method = method or self._detect_method_from_filename(img_path)

        if self.task == "detect":
            if self.detector:
                try:
                    outputs = self.detector.run(None, inputs)
                    stego_logits = outputs[0]
                    method_logits = outputs[1]
                    method_id = outputs[2]
                    method_probs = outputs[3]

                    prob = float(1 / (1 + np.exp(-stego_logits[0][0])))  # Sigmoid
                    predicted_method = int(np.argmax(method_logits[0]))

                    return {
                        "image_path": img_path,
                        "stego_probability": prob,
                        "task": self.task,
                        "method": method,
                        "predicted_method_id": predicted_method,
                        "predicted": prob > 0.5
                    }
                except Exception as e:
                    return {"error": f"ONNX inference failed: {e}"}
            else:
                return {"error": "Detector model not loaded"}
        else:
            return {"error": f"Task '{self.task}' not supported in unified design"}


if ONNX_AVAILABLE and load_unified_input:
    class StarlightSteganographyDetectionPipeline:
        def __init__(self, model_path=None, config_path="config.json", **kwargs):
            # Load config
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path, 'r') as f:
                self.config = json.load(f)

            if model_path is None:
                model_path = self.config.get("model_path", "models/detector_balanced.onnx")

            # Load ONNX model
            providers = []
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            if 'CoreMLExecutionProvider' in available_providers:
                providers.append('CoreMLExecutionProvider')
            providers.append('CPUExecutionProvider')

            session_options = ort.SessionOptions()
            if 'CUDAExecutionProvider' in providers:
                session_options.enable_mem_pattern = False
            elif 'CoreMLExecutionProvider' in providers:
                session_options.enable_mem_pattern = False

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            self.model = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

        def __call__(self, image_path, **kwargs):
            sanitized_kwargs, _, _ = self._sanitize_parameters(**kwargs)
            model_inputs = self.preprocess(image_path)
            model_outputs = self._forward(model_inputs)
            return self.postprocess(model_outputs)

        def _sanitize_parameters(self, **kwargs):
            # No specific parameters to sanitize for now
            return {}, {}, {}

        def preprocess(self, image_path):
            if not isinstance(image_path, str) or not os.path.exists(image_path):
                raise ValueError(f"Invalid image_path: {image_path}")

            # Use unified input loader
            try:
                pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features = load_unified_input(image_path)
            except Exception as e:
                raise ValueError(f"Failed to preprocess image {image_path}: {e}")

            # Convert to numpy for ONNX and add batch dimension
            # Note: lsb and alpha need to be in CHW format for ONNX
            lsb_chw = lsb.permute(2, 0, 1) if lsb.dim() == 3 else lsb  # (3, 256, 256)
            alpha_chw = alpha.unsqueeze(0) if alpha.dim() == 2 else alpha  # (1, 256, 256)

            model_inputs = {
                'meta': np.expand_dims(meta.numpy(), 0),
                'alpha': np.expand_dims(alpha_chw.numpy(), 0),
                'lsb': np.expand_dims(lsb_chw.numpy(), 0),
                'palette': np.expand_dims(palette.numpy(), 0),
                'format_features': np.expand_dims(format_features.numpy(), 0),
                'content_features': np.expand_dims(content_features.numpy(), 0),
                'bit_order': np.array([[0.0, 1.0, 0.0]], dtype=np.float32) # Default msb-first
            }

            return model_inputs

        def _forward(self, model_inputs):
            try:
                outputs = self.model.run(None, model_inputs)
                return {
                    'stego_logits': outputs[0],
                    'method_logits': outputs[1],
                }
            except Exception as e:
                raise RuntimeError(f"ONNX inference failed: {e}")

        def postprocess(self, model_outputs):
            stego_logits = model_outputs['stego_logits']
            method_logits = model_outputs['method_logits']

            prob = float(1 / (1 + np.exp(-stego_logits[0][0]))) # Sigmoid
            
            method_probs = np.exp(method_logits[0]) / np.sum(np.exp(method_logits[0]))
            predicted_method_id = int(np.argmax(method_logits[0]))
            predicted_method_name = self.config["id2label"].get(str(predicted_method_id), "unknown")

            return {
                "stego_probability": prob,
                "predicted_method": predicted_method_name,
                "predicted_method_id": predicted_method_id,
                "predicted_method_prob": float(method_probs[predicted_method_id]),
                "is_steganography": prob > 0.5
            }

        def _sanitize_parameters(self, **kwargs):
            # No specific parameters to sanitize for now
            return {}, {}, {}

        def preprocess(self, image_path):
            if not isinstance(image_path, str) or not os.path.exists(image_path):
                raise ValueError(f"Invalid image_path: {image_path}")

            # Use unified input loader
            try:
                pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features = load_unified_input(image_path)
            except Exception as e:
                raise ValueError(f"Failed to preprocess image {image_path}: {e}")

            # Convert to numpy for ONNX and add batch dimension
            # Note: lsb and alpha need to be in CHW format for ONNX
            lsb_chw = lsb.permute(2, 0, 1) if lsb.dim() == 3 else lsb  # (3, 256, 256)
            alpha_chw = alpha.unsqueeze(0) if alpha.dim() == 2 else alpha  # (1, 256, 256)

            model_inputs = {
                'meta': np.expand_dims(meta.numpy(), 0),
                'alpha': np.expand_dims(alpha_chw.numpy(), 0),
                'lsb': np.expand_dims(lsb_chw.numpy(), 0),
                'palette': np.expand_dims(palette.numpy(), 0),
                'format_features': np.expand_dims(format_features.numpy(), 0),
                'content_features': np.expand_dims(content_features.numpy(), 0),
                'bit_order': np.array([[0.0, 1.0, 0.0]], dtype=np.float32) # Default msb-first
            }

            return model_inputs

        def _forward(self, model_inputs):
            try:
                outputs = self.model.run(None, model_inputs)
                return {
                    'stego_logits': outputs[0],
                    'method_logits': outputs[1],
                }
            except Exception as e:
                raise RuntimeError(f"ONNX inference failed: {e}")

        def postprocess(self, model_outputs):
            stego_logits = model_outputs['stego_logits']
            method_logits = model_outputs['method_logits']

            prob = float(1 / (1 + np.exp(-stego_logits[0][0]))) # Sigmoid
            
            method_probs = np.exp(method_logits[0]) / np.sum(np.exp(method_logits[0]))
            predicted_method_id = int(np.argmax(method_logits[0]))
            predicted_method_name = self.config["id2label"].get(str(predicted_method_id), "unknown")

            return {
                "stego_probability": prob,
                "predicted_method": predicted_method_name,
                "predicted_method_id": predicted_method_id,
                "predicted_method_prob": float(method_probs[predicted_method_id]),
                "is_steganography": prob > 0.5
            }

# Convenience functions for specific tasks
def detect_steganography(img_path):
    """Detect steganography using the unified model."""
    model = StarlightModel(task="detect")
    return model.predict(img_path)

def get_starlight_pipeline():
    """
    Initializes and returns the StarlightSteganographyDetectionPipeline.
    Raises ImportError if dependencies are not met.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX runtime library not found. Please install it with 'pip install onnxruntime'.")
    if not load_unified_input:
        raise ImportError("starlight_utils could not be imported. Please ensure the 'scripts' directory is in your Python path.")
    
    return StarlightSteganographyDetectionPipeline()
