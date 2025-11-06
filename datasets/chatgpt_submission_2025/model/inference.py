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

        if cfg["mode"] == "RGB":
            return self._preprocess_rgb(img_path, cfg)
        elif cfg["mode"] == "RGBA":
            return self._preprocess_rgba(img_path, cfg)
        elif cfg["mode"] == "DCT":
            return self._preprocess_dct(img_path)
        elif cfg["mode"] == "EXIF":
            return self._preprocess_exif(img_path)
        elif cfg["mode"] == "EOI":
            return self._preprocess_eoi(img_path)
        elif cfg["mode"] == "P":
            return self._preprocess_palette(img_path, cfg)
        else:
            raise NotImplementedError(f"Mode {cfg['mode']} not supported")

    def _preprocess_rgb(self, img_path, cfg):
        img = Image.open(img_path).convert("RGB").resize(cfg["resize"])
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))          # C×H×W
        return np.expand_dims(arr, 0)               # 1×C×H×W

    def _preprocess_rgba(self, img_path, cfg):
        img = Image.open(img_path).convert("RGBA").resize(cfg["resize"])
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
        img = Image.open(img_path).convert("P")
        if hasattr(Image, 'Palette') and hasattr(Image.Palette, 'ADAPTIVE'):
            img = Image.open(img_path).convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        data = list(img.getdata())
        bits = "".join(f"{b:08b}" for b in img.tobytes()) + "00000000"
        new_data = []
        i = 0
        for idx in data:
            if i < len(bits):
                idx = (idx & 0xFE) | int(bits[i])
                i += 1
            new_data.append(idx)
        out = Image.new("P", img.size)
        out.putdata(new_data)
        palette = img.getpalette()
        if palette is not None:
            out.putpalette(palette)
        return np.expand_dims(np.array(out).astype(np.float32) / 255.0, 0)

    # -------------------------------------------------
    def predict(self, img_path: str, method: str | None = None) -> dict:
        method = method or self._detect_method_from_filename(img_path)
        input_data = self.preprocess(img_path, method)

        # Detector inference -------------------------------------------------
        outputs = self.detector.run(None, {self.input_name: input_data.astype(np.float32)})
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