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

# Add parent directory to import steganography modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Initialize steganography modules as None
analyze_lsb = None
extract_lsb_simple = None
detect_exif_stego = None
extract_exif_payload = None

try:
    from lsb_steganography import analyze_lsb, extract_lsb_simple
    from exif_steganography import detect_exif_stego, extract_exif_payload
except ImportError as e:
    print(f"Warning: Could not import steganography modules: {e}")


class StarlightModel:
    def __init__(
        self,
        detector_path: str = "model/detector.onnx",
        extractor_path: str = None,
        task: str = "detect",
    ):
        self.detector_path = detector_path
        self.extractor_path = extractor_path
        self.task = task

        # Load ONNX models if available
        self.detector = None
        self.extractor = None
        if ONNX_AVAILABLE:
            # Dynamically select available providers: CUDA > CoreML > CPU
            providers = []
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in available_providers:
                providers.append("CoreMLExecutionProvider")
            providers.append("CPUExecutionProvider")

            session_options = ort.SessionOptions()
            if "CUDAExecutionProvider" in providers:
                session_options.enable_mem_pattern = False  # Optimize for GPU
            elif "CoreMLExecutionProvider" in providers:
                session_options.enable_mem_pattern = False  # Optimize for MPS

            if os.path.exists(detector_path):
                try:
                    self.detector = ort.InferenceSession(
                        detector_path, sess_options=session_options, providers=providers
                    )
                    self.input_name = self.detector.get_inputs()[0].name
                except Exception as e:
                    print(f"Warning: Could not load detector: {e}")
            if extractor_path and os.path.exists(extractor_path):
                try:
                    self.extractor = ort.InferenceSession(
                        extractor_path,
                        sess_options=session_options,
                        providers=providers,
                    )
                except Exception as e:
                    print(f"Warning: Could not load extractor: {e}")

        self.method_config = self._load_method_config()

    def _load_method_config(self) -> Dict:
        # Look for method_config.json in the same directory as this script
        config_path = os.path.join(os.path.dirname(__file__), "method_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError("method_config.json is required")
        with open(config_path) as f:
            return json.load(f)

    def _detect_method_from_filename(self, img_path: str) -> str:
        basename = os.path.basename(img_path)
        parts = basename.split("_")
        if len(parts) >= 3:
            method = parts[-2]  # e.g., alpha, eoi, dct
            if method in self.method_config:
                return method
        return "lsb"  # Default fallback

    def preprocess(self, img_path: str, method: str = None) -> np.ndarray:
        method = method or self._detect_method_from_filename(img_path)
        config = self.method_config.get(method, self.method_config["lsb"])

        if config["mode"] == "RGB":
            return self._preprocess_rgb(img_path, config)
        elif config["mode"] == "RGBA":
            return self._preprocess_rgba(img_path, config)
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
        return np.expand_dims(arr, 0)

    def _preprocess_exif(self, img_path):
        img = Image.open(img_path)
        exif = img.info.get("exif")
        data = np.frombuffer(exif or b"", dtype=np.uint8)[:1024]
        padded = np.zeros(1024, dtype=np.float32)
        padded[: len(data)] = data.astype(np.float32) / 255.0
        return padded.reshape(1, -1)

    def _preprocess_eoi(self, img_path):
        """Preprocess EOI (End of Image) steganography by reading tail bytes"""
        with open(img_path, "rb") as f:
            raw = f.read()

        # Detect format
        if raw.startswith(b"\xff\xd8"):
            format_hint = "jpeg"
        elif raw.startswith(b"\x89PNG"):
            format_hint = "png"
        elif raw.startswith(b"GIF8"):
            format_hint = "gif"
        elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            format_hint = "webp"
        else:
            format_hint = "unknown"

        # Extract post-tail
        if format_hint == "jpeg":
            eoi_pos = raw.rfind(b"\xff\xd9")
            appended = raw[eoi_pos + 2 :] if eoi_pos != -1 else b""
        elif format_hint == "png":
            iend_pos = raw.rfind(b"IEND")
            appended = (
                raw[iend_pos + 12 :] if iend_pos != -1 else b""
            )  # After IEND chunk
        elif format_hint == "gif":
            term_pos = raw.rfind(b";")
            appended = raw[term_pos + 1 :] if term_pos != -1 else b""
        elif format_hint == "webp":
            vp8x_pos = raw.rfind(b"VP8X")
            appended = raw[vp8x_pos + 10 :] if vp8x_pos != -1 else b""
        else:
            appended = b""

        data = np.frombuffer(appended, dtype=np.uint8)[:1024]
        padded = np.zeros(1024, dtype=np.float32)
        padded[: len(data)] = data.astype(np.float32) / 255.0
        return padded.reshape(1, -1)

    def _preprocess_palette(self, img_path, config):
        """Preprocess palette images by converting to RGB"""
        img = Image.open(img_path).convert("RGB").resize(config["resize"])
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)

    def predict(self, img_path: str, method: str = None) -> Dict[str, Any]:
        method = method or self._detect_method_from_filename(img_path)
        input_data = self.preprocess(img_path, method)

        if self.task == "detect":
            if self.detector and self.input_name:
                try:
                    outputs = self.detector.run(None, {self.input_name: input_data})
                    prob = float(outputs[0][0])
                    return {
                        "image_path": img_path,
                        "stego_probability": prob,
                        "task": self.task,
                        "method": method,
                        "predicted": prob > 0.5,
                    }
                except Exception:
                    pass
            # Fallback to rule-based analysis
            prob = self._calculate_overall_probability(img_path)
            return {
                "image_path": img_path,
                "stego_probability": prob,
                "task": self.task,
                "method": method,
                "predicted": prob > 0.5,
            }
        elif self.task == "extract":
            if self.extractor:
                try:
                    outputs = self.extractor.run(
                        None, {self.extractor.get_inputs()[0].name: input_data}
                    )
                    payload = "".join(
                        chr(int(b)) for b in np.argmax(outputs[0], axis=-1)
                    )
                    return {"method": method, "extracted_payload": payload}
                except Exception:
                    pass
            # Fallback to rule-based extraction
            if method in ["lsb", "sequential_lsb"]:
                payload = self._extract_lsb(img_path)
            elif method == "exif":
                payload = self._extract_exif(img_path)
            else:
                payload = ""
            return {"method": method, "extracted_payload": payload}
        else:
            return {"error": f"Unknown task: {self.task}"}

    def _analyze_lsb(self, img_path):
        """Analyze LSB patterns for steganography detection."""
        try:
            if analyze_lsb is None:
                return {"lsb_error": "LSB analysis module not available"}
            stats, _ = analyze_lsb(img_path)

            # Calculate LSB-based probability
            lsb_prob = 0.0
            indicators = []

            # Check for unusual LSB ratios
            for channel, ratio in stats["lsb_ones_ratio"].items():
                if abs(ratio - 0.5) > 0.1:  # Deviation from random
                    lsb_prob += 0.2
                    indicators.append(f"Unusual LSB ratio in {channel}: {ratio:.3f}")

            # Check entropy
            for channel, entropy in stats["lsb_entropy"].items():
                if entropy > 0.95:  # High entropy
                    lsb_prob += 0.1
                    indicators.append(f"High LSB entropy in {channel}: {entropy:.3f}")

            # Try to extract LSB payload
            try:
                if extract_lsb_simple is None:
                    return {"lsb_error": "LSB extraction module not available"}
                extracted_bits = extract_lsb_simple(img_path, 100)
                # Check if extracted bits look like text
                bytes_list = []
                for i in range(0, len(extracted_bits), 8):
                    if i + 8 <= len(extracted_bits):
                        byte_bits = extracted_bits[i : i + 8]
                        byte_val = 0
                        for bit in byte_bits:
                            byte_val = (byte_val << 1) | bit
                        bytes_list.append(byte_val)

                # Try to decode as ASCII
                try:
                    decoded = "".join(
                        [chr(b) if 32 <= b <= 126 else "" for b in bytes_list]
                    )
                    if len(decoded) > 10:  # Meaningful text found
                        lsb_prob += 0.4
                        indicators.append("Extractable LSB text found")
                except:
                    pass
            except:
                pass

            return {
                "lsb_probability": min(lsb_prob, 1.0),
                "lsb_indicators": indicators,
                "lsb_stats": stats,
            }
        except Exception as e:
            return {"lsb_error": str(e)}

    def _analyze_exif(self, img_path):
        """Analyze EXIF data for steganography detection."""
        try:
            if detect_exif_stego is None or extract_exif_payload is None:
                return {"exif_error": "EXIF analysis module not available"}
            detection = detect_exif_stego(img_path)
            extraction = extract_exif_payload(img_path)

            result = {
                "exif_probability": detection["stego_probability"],
                "exif_indicators": detection["indicators"],
            }

            if extraction["payload"]:
                result["exif_extracted_payload"] = extraction["payload"]
                result["exif_payload_size"] = extraction["payload_size"]

            if detection["error"]:
                result["exif_error"] = detection["error"]

            return result
        except Exception as e:
            return {"exif_error": str(e)}

    def _calculate_overall_probability(self, img_path):
        """Calculate overall steganography probability from all methods."""
        lsb_result = self._analyze_lsb(img_path)
        exif_result = self._analyze_exif(img_path)

        probabilities = []

        # LSB probability
        if "lsb_probability" in lsb_result:
            probabilities.append(lsb_result["lsb_probability"] * 0.7)  # Weight: 70%

        # EXIF probability
        if "exif_probability" in exif_result:
            probabilities.append(exif_result["exif_probability"] * 0.3)  # Weight: 30%

        if probabilities:
            return min(sum(probabilities), 1.0)
        else:
            return 0.0

    def _extract_lsb(self, img_path):
        """Extract LSB payload."""
        try:
            if extract_lsb_simple is None:
                return ""
            extracted_bits = extract_lsb_simple(img_path, 800)  # More bits
            bytes_list = []
            for i in range(0, len(extracted_bits), 8):
                if i + 8 <= len(extracted_bits):
                    byte_bits = extracted_bits[i : i + 8]
                    byte_val = 0
                    for bit in byte_bits:
                        byte_val = (byte_val << 1) | bit
                    bytes_list.append(byte_val)
            return "".join([chr(b) if 32 <= b <= 126 else "" for b in bytes_list])
        except:
            return ""

    def _extract_exif(self, img_path):
        """Extract EXIF payload."""
        try:
            if extract_exif_payload is None:
                return ""
            extraction = extract_exif_payload(img_path)
            return extraction.get("payload", "")
        except:
            return ""


# Convenience functions for specific tasks
def detect_steganography(img_path):
    """Detect steganography using all available methods."""
    model = StarlightModel(task="detect")
    return model.predict(img_path)


def extract_payload(img_path, method=None):
    """Extract payload from image."""
    model = StarlightModel(task="extract")
    return model.predict(img_path, method)
