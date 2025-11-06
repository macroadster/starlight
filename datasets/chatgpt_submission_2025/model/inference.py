#!/usr/bin/env python3
"""
Method-aware inference for a submitted Starlight model.
Follows the specification in docs/ai_proposal.md
"""

import json
import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import jpegio as jio   # pip install jpegio

class StarlightModel:
    def __init__(self,
                 detector_path: str = "model/detector.onnx",
                 extractor_path: str | None = None,
                 task: str = "detect"):
        self.detector = ort.InferenceSession(detector_path)
        self.extractor = ort.InferenceSession(extractor_path) if extractor_path else None
        self.task = task
        self.input_name = self.detector.get_inputs()[0].name
        self.method_config = self._load_method_config()

    def _load_method_config(self) -> dict:
        cfg_path = os.path.join(os.path.dirname(__file__), "method_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("method_config.json is required")
        with open(cfg_path, "r") as f:
            return json.load(f)

    # -------------------------------------------------
    # method detection -------------------------------------------------
    def _detect_method_from_filename(self, img_path: str) -> str:
        # filenames are `<payload>_<algo>_###.<ext>`
        parts = os.path.basename(img_path).split("_")
        return parts[-2] if len(parts) >= 3 else "lsb"
    # -------------------------------------------------

    # ---------- preprocessing per method ----------
    def preprocess(self, img_path: str, method: str | None = None) -> np.ndarray:
        method = method or self._detect_method_from_filename(img_path)
        cfg = self.method_config.get(method, self.method_config["lsb"])

        # ChatGPT model only supports RGB input, convert all to RGB
        if cfg["mode"] in ["RGB", "RGBA", "P"]:
            return self._preprocess_rgb(img_path, cfg)
        elif cfg["mode"] == "DCT":
            # For DCT images, convert to RGB
            return self._preprocess_rgb(img_path, cfg)
        elif cfg["mode"] == "EXIF":
            # For EXIF analysis, convert to RGB  
            return self._preprocess_rgb(img_path, cfg)
        elif cfg["mode"] == "EOI":
            # For EOI analysis, convert to RGB
            return self._preprocess_rgb(img_path, cfg)
        else:
            # Default to RGB for unknown modes
            return self._preprocess_rgb(img_path, cfg)

    def _preprocess_rgb(self, img_path, cfg):
        # ChatGPT model expects 224x224 input
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        
        # Ensure we have 3D array (H, W, C) before transpose
        if arr.ndim == 2:
            # Grayscale image, convert to 3 channels
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            # Single channel image, duplicate to 3 channels
            arr = np.repeat(arr, 3, axis=2)
        
        arr = np.transpose(arr, (2, 0, 1))          # C×H×W
        result = np.expand_dims(arr, 0)              # 1×C×H×W
        
        # Ensure 4D output
        if result.ndim != 4:
            raise ValueError(f"RGB preprocessing failed: got {result.ndim}D tensor, expected 4D")
        
        return result

    def _preprocess_rgba(self, img_path, cfg):
        # ChatGPT model expects RGB input, convert RGBA to RGB
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)

    def _preprocess_dct(self, img_path):
        jpeg = jio.read(img_path)
        coeffs = jpeg.coef_blocks[0]
        coeffs = (coeffs - coeffs.mean()) / (coeffs.std() + 1e-8)
        padded = np.pad(coeffs.flatten(), (0, 4096), 'constant')[:4096]
        return padded.astype(np.float32).reshape(1, -1)

    def _preprocess_exif(self, img_path):
        img = Image.open(img_path)
        exif = img.info.get("exif", b"")
        data = np.frombuffer(exif, dtype=np.uint8)[:1024]
        padded = np.zeros(1024, dtype=np.float32)
        padded[:len(data)] = data.astype(np.float32) / 255.0
        return padded.reshape(1, -1)

    def _preprocess_eoi(self, img_path):
        with open(img_path, "rb") as f:
            f.seek(-1024, 2)
            tail = f.read()
        eoi_pos = tail.find(b'\xFF\xD9')
        appended = tail[eoi_pos + 2:] if eoi_pos != -1 else tail
        data = np.frombuffer(appended, dtype=np.uint8)[:1024]
        padded = np.zeros(1024, dtype=np.float32)
        padded[:len(data)] = data.astype(np.float32) / 255.0
        return padded.reshape(1, -1)

    def _preprocess_palette(self, img_path, cfg):
        # Convert palette to RGB for ChatGPT model
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)

    def _create_metadata(self, img_path: str) -> np.ndarray:
        """Create metadata features from image"""
        # Simple metadata: [file_size, exif_presence, eoi_presence]
        file_size = os.path.getsize(img_path) / 1e6  # Normalize to MB
        
        # Check for EXIF data
        try:
            img = Image.open(img_path)
            exif_present = 1.0 if img.info.get("exif") else 0.0
        except:
            exif_present = 0.0
        
        # Check for data after EOI marker
        try:
            with open(img_path, "rb") as f:
                f.seek(-100, 2)  # Check last 100 bytes
                tail = f.read()
            eoi_present = 1.0 if b'\xFF\xD9' in tail[:-2] else 0.0
        except:
            eoi_present = 0.0
        
        return np.array([[file_size, exif_present, eoi_present]], dtype=np.float32)

    # -------------------------------------------------
    def predict(self, img_path: str, method: str | None = None) -> dict:
        method = method or self._detect_method_from_filename(img_path)
        input_data = self.preprocess(img_path, method)

        # Create metadata input (3 features: exif, eoi, combined)
        metadata = self._create_metadata(img_path)

        # Detector inference -------------------------------------------------
        input_names = [inp.name for inp in self.detector.get_inputs()]
        input_dict = {}
        
        if 'rgb' in input_names:
            input_dict['rgb'] = input_data.astype(np.float32)
        if 'metadata' in input_names:
            input_dict['metadata'] = metadata.astype(np.float32)
        elif self.input_name in input_dict:
            # Fallback for single input models
            input_dict[self.input_name] = input_data.astype(np.float32)

        outputs = self.detector.run(None, input_dict)
        
        # Handle different output formats
        if len(outputs[0].shape) == 2 and outputs[0].shape[1] == 2:
            # Binary classification with logits
            probs = 1 / (1 + np.exp(-outputs[0]))  # sigmoid
            prob = float(probs[0, 1])  # probability of class 1 (stego)
        else:
            prob = float(np.array(outputs[0]).flatten()[0])

        result = {
            "method": method,
            "stego_probability": prob,
            "predicted": prob > 0.5
        }

        # Extraction (optional) -------------------------------------------------
        if self.task == "extract" and self.extractor:
            out = self.extractor.run(None, {self.extractor.get_inputs()[0].name: input_data.astype(np.float32)})
            out_array = np.array(out[0])
            if out_array.size > 0:
                payload = ''.join(chr(int(b)) for b in np.argmax(out_array, axis=-1).flatten() if int(b) < 128)
                result["extracted_payload"] = payload
            else:
                result["extracted_payload"] = ""

        return result

# Example usage
if __name__ == "__main__":
    model = StarlightModel()
    # Test with a sample image if available
    test_img = "../clean/sample_seed_alpha_000.png"
    if os.path.exists(test_img):
        result = model.predict(test_img)
        print("Prediction result:", result)
    else:
        print("Model initialized. Ready for inference.")