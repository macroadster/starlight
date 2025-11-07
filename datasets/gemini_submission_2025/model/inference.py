import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import json
from typing import Dict, Any
import jpegio as jio  # pip install jpegio
import argparse

# --- CONFIGURATION ---
class StarlightModel:
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.method_config = self._load_method_config()
        self.sessions = {} # Cache for loaded models

    def _load_method_config(self) -> Dict:
        config_path = os.path.join(self.model_dir, "method_config.json")
        if not os.path.exists(config_path):
            # Look in parent directory if not in model_dir
            config_path = os.path.join(os.path.dirname(self.model_dir), "method_config.json")
            if not os.path.exists(config_path):
                 raise FileNotFoundError("method_config.json is required")
        with open(config_path) as f:
            return json.load(f)

    def _detect_method_from_filename(self, img_path: str) -> str:
        basename = os.path.basename(img_path)
        parts = basename.split("_")
        if len(parts) >= 3:
            # Assuming filename format: {payload_name}_{algorithm}_{index}.{ext}
            return parts[-2]  # e.g., alpha, eoi, dct
        return "lsb" # Default fallback

    def preprocess(self, img_path: str, method: str) -> np.ndarray:
        config = self.method_config.get(method, self.method_config["lsb"]) # Fallback to lsb config

        if config["mode"] == "RGB":
            return self._preprocess_rgb(img_path, config)
        elif config["mode"] == "RGBA":
            return self._preprocess_rgba(img_path, config)
        elif config["mode"] == "DCT":
            return self._preprocess_dct(img_path)
        elif config["mode"] == "EXIF":
            return self._preprocess_exif(img_path)
        elif config["mode"] == "EOI":
            return self._preprocess_eoi(img_path)
        elif config["mode"] == "P":
            return self._preprocess_palette(img_path, config)
        else:
            raise NotImplementedError(f"Mode {config['mode']} not supported")

    def _preprocess_rgb(self, img_path, config):
        img = Image.open(img_path).convert("RGB").resize(config["resize"])
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)

    def _preprocess_rgba(self, img_path, config):
        img = Image.open(img_path).convert("RGBA").resize(config["resize"])
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # (4, H, W)
        return arr

    def _preprocess_dct(self, img_path):
        jpeg = jio.read(img_path)
        coeffs = jpeg.coef_blocks[0]
        coeffs = (coeffs - coeffs.mean()) / (coeffs.std() + 1e-8)
        padded = np.pad(coeffs.flatten(), (0, 4096), 'constant')[:4096]
        return padded.astype(np.float32).reshape(1, -1)

    def _preprocess_exif(self, img_path):
        img = Image.open(img_path)
        exif = img.info.get("exif")
        data = np.frombuffer(exif or b'', dtype=np.uint8)[:1024]
        padded = np.zeros(1024, dtype=np.float32)
        padded[:len(data)] = data.astype(np.float32) / 255.0
        return padded.reshape(1, -1)

    def _preprocess_eoi(self, img_path):
        with open(img_path, 'rb') as f:
            f.seek(0, 2)  # Go to the end of the file
            file_size = f.tell()
            
            read_size = min(file_size, 1024)
            f.seek(-read_size, 2)
            tail = f.read(read_size)
            
        eoi_pos = tail.find(b'\xFF\xD9')
        
        if eoi_pos != -1:
            appended = tail[eoi_pos + 2:]
        else:
            appended = b'' # If EOI marker not found, assume no appended data
            
        data = np.frombuffer(appended, dtype=np.uint8)[:1024]
        padded = np.zeros(1024, dtype=np.float32)
        padded[:len(data)] = data.astype(np.float32) / 255.0
        return padded.reshape(1, -1) # Reshape to (1, 1024) for (channel, length)

    def _preprocess_palette(self, img_path, config):
        img = Image.open(img_path).convert("P").resize(config["resize"])
        indices = np.array(img)
        palette = np.array(img.getpalette()[:768]).reshape(256, 3) / 255.0
        hist = np.histogram(indices, bins=256, range=(0, 255), density=True)[0]
        combined = np.concatenate([hist, palette.flatten()])
        return combined.astype(np.float32).reshape(1, -1)

    def _postprocess_alpha_extraction(self, model_output: np.ndarray, hint_bytes: bytes, terminator_bytes: bytes) -> str:
        # model_output is (1, 1, H, W) with logits
        # Apply sigmoid and threshold to get binary bits
        bits = (1 / (1 + np.exp(-model_output))).flatten() > 0.5
        
        # Convert bits to bytes
        extracted_bytes = bytearray()
        current_byte = 0
        bit_count = 0
        for bit in bits:
            current_byte |= (bit << bit_count)
            bit_count += 1
            if bit_count == 8:
                extracted_bytes.append(current_byte)
                current_byte = 0
                bit_count = 0
        
        # Search for hint_bytes and terminator_bytes
        full_extracted_data = bytes(extracted_bytes)
        
        if not hint_bytes:
            hint_start = 0
            payload_start = 0
        else:
            hint_start = full_extracted_data.find(hint_bytes)
            if hint_start == -1:
                return ""
            payload_start = hint_start + len(hint_bytes)
        
        if not terminator_bytes:
            terminator_end = len(full_extracted_data)
        else:
            terminator_end = full_extracted_data.find(terminator_bytes, payload_start)
            if terminator_end == -1:
                terminator_end = len(full_extracted_data) # If terminator not found, read till end
        
        return full_extracted_data[payload_start:terminator_end].decode('utf-8', errors='ignore')

    def _postprocess_eoi_extraction(self, model_output: np.ndarray, hint_bytes: bytes, terminator_bytes: bytes) -> str:
        # model_output is (1, 1, 1024) with float values representing bytes
        # Convert float values to bytes (round and clamp)
        extracted_bytes = np.round(model_output).astype(np.uint8).flatten()
        
        # Search for hint_bytes and terminator_bytes
        full_extracted_data = bytes(extracted_bytes)
        
        if not hint_bytes:
            hint_start = 0
            payload_start = 0
        else:
            hint_start = full_extracted_data.find(hint_bytes)
            if hint_start == -1:
                return ""
            payload_start = hint_start + len(hint_bytes)
        
        if not terminator_bytes:
            terminator_end = len(full_extracted_data)
        else:
            terminator_end = full_extracted_data.find(terminator_bytes, payload_start)
            if terminator_end == -1:
                terminator_end = len(full_extracted_data) # If terminator not found, read till end
        
        return full_extracted_data[payload_start:terminator_end].decode('utf-8', errors='ignore')

    def predict(self, img_path: str, task: str, method: str = None) -> Dict[str, Any]:
        method = method or self._detect_method_from_filename(img_path)
        
        # Construct model path and load session on-the-fly
        model_path = os.path.join(self.model_dir, method, f"{task}or.onnx")
        
        if model_path in self.sessions:
            session = self.sessions[model_path]
        elif os.path.exists(model_path):
            session = ort.InferenceSession(model_path)
            self.sessions[model_path] = session
        else:
            return {"error": f"Model not found for method: {method}, task: {task} at {model_path}"}

        # Preprocess and run inference
        input_data = self.preprocess(img_path, method)
        
        # Add batch dimension if not present (some preprocessing steps might already add it)
        if input_data.ndim == 3: # e.g. (C, H, W)
             input_data = np.expand_dims(input_data, 0)
        
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})

        if task == "detect":
            prob = float(outputs[0][0])
            return {
                "method": method,
                "stego_probability": prob,
                "predicted": prob > 0.5
            }
        elif task == "extract":
            config = self.method_config.get(method, self.method_config.get("lsb", {}))
            hint_bytes_hex = config.get("hint_bytes", "")
            terminator_bytes_hex = config.get("terminator_bytes", "")
            
            hint_bytes = bytes.fromhex(hint_bytes_hex) if hint_bytes_hex else b''
            terminator_bytes = bytes.fromhex(terminator_bytes_hex) if terminator_bytes_hex else b''

            if method == "alpha":
                extracted_payload = self._postprocess_alpha_extraction(outputs[0], hint_bytes, terminator_bytes)
            elif method == "eoi":
                extracted_payload = self._postprocess_eoi_extraction(outputs[0], hint_bytes, terminator_bytes)
            else:
                extracted_payload = "Extraction not supported for this method."

            return {
                "method": method,
                "extracted_payload": extracted_payload
            }
        else:
            return {"error": f"Unknown task: {task}"}


def main():
    parser = argparse.ArgumentParser(description="Perform steganography detection or extraction.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--task", type=str, required=True, choices=["detect", "extract"], help="Task to perform (detect or extract).")
    parser.add_argument("--method", type=str, help="Steganography method (e.g., alpha, eoi). If not provided, will attempt to detect from filename.")
    args = parser.parse_args()

    model = StarlightModel()
    result = model.predict(args.image_path, args.task, args.method)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()