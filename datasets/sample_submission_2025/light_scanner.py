import onnxruntime as ort
import torch
from PIL import Image
import numpy as np
import struct
import onnxruntime as ort


def load_dual_input(path):
    # --- Pixel path ---
    img = Image.open(path)
    if img.mode == "P":
        # For palette, keep indices as 1-channel
        img = img.resize((256, 256), resample=0)  # NEAREST
        indices = np.array(img).astype(np.float32) / 255.0  # (256,256)
        pixel = np.stack([indices] * 4, axis=-1)  # (256,256,4), all channels same
    else:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        pixel = (
            np.array(img.resize((256, 256), resample=0)).astype(np.float32) / 255.0
        )  # (256,256,4)
    pixel = torch.from_numpy(pixel).permute(2, 0, 1).unsqueeze(0)  # (1,4,256,256)

    # --- Metadata path ---
    with open(path, "rb") as f:
        raw = f.read()

    # EXIF: find first APP1 (0xFFE1)
    exif = b""
    pos = raw.find(b"\xff\xe1")
    if pos != -1:
        length = struct.unpack(">H", raw[pos + 2 : pos + 4])[0]
        exif = raw[pos + 4 : pos + 4 + length - 2]

    # EOI tail: after last 0xFFD9
    eoi_pos = raw.rfind(b"\xff\xd9")
    tail = raw[eoi_pos + 2 :] if eoi_pos != -1 else b""

    # Combine + pad to 1024
    meta = np.frombuffer(exif + tail, dtype=np.uint8)[:1024]
    meta = np.pad(meta, (0, 1024 - len(meta)), "constant")
    meta = torch.from_numpy(meta.astype(np.float32) / 255.0).unsqueeze(0)  # (1,1024)

    return pixel, meta


class LightScanner:
    def __init__(self, model="models/detector_dual.onnx"):
        self.session = ort.InferenceSession(model)
        self.pixel_in = "pixel"
        self.meta_in = "metadata"

    def scan(self, path):
        pixel, meta = load_dual_input(path)
        outputs = self.session.run(
            None, {self.pixel_in: pixel.numpy(), self.meta_in: meta.numpy()}
        )
        prob, method_id = outputs[0][0], int(outputs[1][0])
        method = ["alpha", "palette", "rgb_lsb", "exif", "eoi"][method_id]
        return {"stego": prob > 0.5, "probability": float(prob), "method": method}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python light_scanner.py <image_path>")
        sys.exit(1)
    scanner = LightScanner()
    result = scanner.scan(sys.argv[1])
    print(result)
