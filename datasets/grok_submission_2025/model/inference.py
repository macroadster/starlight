# inference.py
import numpy as np
from PIL import Image
import os
import sys

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
    def __init__(self, model_path="model/detector.onnx", task="detect"):
        """
        Initialize Starlight model for steganography detection and extraction.

        Args:
            model_path (str): Path to ONNX model file
            task (str): 'detect', 'extract_lsb', or 'extract_exif'
        """
        self.model_path = model_path
        self.task = task

        # Load ONNX model if available
        if ONNX_AVAILABLE and os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(model_path)  # type: ignore
                self.input_name = self.session.get_inputs()[0].name
                self.has_model = True
            except Exception as e:
                print(f"Warning: Could not load ONNX model: {e}")
                self.session = None
                self.input_name = None
                self.has_model = False
        else:
            self.session = None
            self.input_name = None
            self.has_model = False

    def preprocess(self, img_path):
        """Preprocess image for neural network model."""
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)

    def predict(self, img_path):
        """
        Main prediction method supporting multiple steganography detection methods.

        Args:
            img_path (str): Path to image file

        Returns:
            dict: Prediction results
        """
        if not os.path.exists(img_path):
            return {"error": "Image file not found"}

        # Neural network prediction (if model available)
        nn_result = {}
        if self.has_model and self.session is not None:
            try:
                input_data = self.preprocess(img_path)
                outputs = self.session.run(None, {self.input_name: input_data})

                if self.task == "detect":
                    # Apply sigmoid to get probability
                    raw_output = float(outputs[0][0])
                    prob = 1 / (1 + np.exp(-raw_output))  # sigmoid
                    nn_result = {"nn_probability": prob, "nn_predicted": prob > 0.5}
                elif self.task == "extract_lsb":
                    # Neural network LSB extraction
                    bits = outputs[0].round().astype(int)
                    payload_bytes = "".join([chr(int(b)) for b in bits])
                    nn_result = {"nn_extracted_payload": payload_bytes}
            except Exception as e:
                nn_result = {"nn_error": str(e)}

        # LSB analysis (always available)
        lsb_result = self._analyze_lsb(img_path)

        # EXIF analysis (for JPEG images)
        exif_result = self._analyze_exif(img_path)

        # Combine results
        combined_result = {
            "image_path": img_path,
            "task": self.task,
        }

        # Add neural network results
        combined_result.update(nn_result)

        # Add LSB results
        combined_result.update(lsb_result)

        # Add EXIF results
        combined_result.update(exif_result)

        # Overall stego probability (weighted combination)
        combined_result["stego_probability"] = self._calculate_overall_probability(
            combined_result
        )

        return combined_result

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

    def _calculate_overall_probability(self, results):
        """Calculate overall steganography probability from all methods."""
        probabilities = []

        # Neural network probability (most reliable)
        if "nn_probability" in results:
            probabilities.append(results["nn_probability"] * 0.5)  # Weight: 50%

        # LSB probability
        if "lsb_probability" in results:
            probabilities.append(results["lsb_probability"] * 0.3)  # Weight: 30%

        # EXIF probability
        if "exif_probability" in results:
            probabilities.append(results["exif_probability"] * 0.2)  # Weight: 20%

        if probabilities:
            return min(sum(probabilities), 1.0)
        else:
            return 0.0


# Convenience functions for specific tasks
def detect_steganography(img_path):
    """Detect steganography using all available methods."""
    model = StarlightModel(task="detect")
    return model.predict(img_path)


def extract_lsb_payload(img_path):
    """Extract LSB payload from image."""
    model = StarlightModel(task="extract_lsb")
    return model.predict(img_path)


def extract_exif_payload_from_image(img_path):
    """Extract EXIF payload from JPEG image."""
    model = StarlightModel(task="extract_exif")
    return model.predict(img_path)
