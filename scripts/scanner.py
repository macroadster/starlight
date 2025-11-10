import onnxruntime
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import struct
import torch
import torchvision.transforms as transforms
import time
import json
from starlight_utils import load_multi_input
from starlight_extractor import extraction_functions

def extract_enhanced_metadata_features(image_path):
    """Extract enhanced metadata features for better EXIF detection"""

    with open(image_path, 'rb') as f:
        raw = f.read()

    # Basic features (1024 bytes as before)
    exif = b""
    pos = raw.find(b'\xFF\xE1')
    if pos != -1:
        length = struct.unpack('>H', raw[pos+2:pos+4])[0]
        exif = raw[pos+4:pos+4+length-2]

    # Find format and extract tail
    format_hint = 'jpeg' if raw.startswith(b'\xFF\xD8') else 'auto'
    if format_hint == 'jpeg':
        tail = raw[raw.rfind(b'\xFF\xD9') + 2:] if raw.rfind(b'\xFF\xD9') != -1 else b""
    else:
        tail = b""

    # Enhanced features (next 1024 bytes)
    enhanced_features = np.zeros(1024, dtype=np.uint8)

    # EXIF position and size features
    exif_positions = []
    pos = 0
    while True:
        pos = raw.find(b'\xFF\xE1', pos)
        if pos == -1:
            break
        exif_positions.append(pos)
        pos += 2

    if exif_positions:
        # Store EXIF positions (first 10 positions, 4 bytes each)
        for i, pos in enumerate(exif_positions[:10]):
            if i * 4 + 3 < len(enhanced_features):
                # Convert to unsigned 32-bit
                unsigned_pos = pos & 0xFFFFFFFF
                enhanced_features[i*4:i*4+4] = np.frombuffer(struct.pack('>I', unsigned_pos), dtype=np.uint8)

        # Store EXIF sizes (first 10 sizes, 2 bytes each)
        exif_sizes = []
        for pos in exif_positions[:10]:
            if pos + 4 <= len(raw):
                length = struct.unpack('>H', raw[pos+2:pos+4])[0]
                exif_sizes.append(length)

        for i, size in enumerate(exif_sizes[:10]):
            if i * 2 + 40 < len(enhanced_features):
                # Convert to unsigned 16-bit
                unsigned_size = size & 0xFFFF
                enhanced_features[40+i*2:40+i*2+2] = np.frombuffer(struct.pack('>H', unsigned_size), dtype=np.uint8)

    # JPEG marker analysis (next 100 bytes)
    jpeg_markers = []
    for i in range(0, min(len(raw) - 1, 10000), 2):  # Check first 10KB
        if raw[i] == 0xFF and i + 1 < len(raw):
            marker = raw[i+1]
            if marker != 0x00:
                jpeg_markers.append(marker)

    # Store marker histogram (50 bytes)
    marker_hist = np.zeros(50, dtype=np.uint8)
    for marker in jpeg_markers[:100]:  # First 100 markers
        idx = marker % 50
        if idx < 50:
            marker_hist[idx] = min(marker_hist[idx] + 1, 255)

    enhanced_features[60:110] = marker_hist

    # Tail analysis (next 100 bytes)
    if len(tail) > 0:
        hist = np.histogram(bytearray(tail[:1000]), bins=256)[0]
        tail_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        enhanced_features[110] = min(max(int(tail_entropy), 0), 255)  # Clamp to 0-255
        enhanced_features[111] = min(len(tail), 255)

        # Store first 88 bytes of tail data
        tail_bytes = np.frombuffer(tail[:88], dtype=np.uint8)
        enhanced_features[112:112+len(tail_bytes)] = tail_bytes

    # Combine basic and enhanced features
    basic_bytes = np.frombuffer(exif + tail, dtype=np.uint8)[:1024]
    basic_bytes = np.pad(basic_bytes, (0, 1024 - len(basic_bytes)), 'constant')

    # Return both basic and enhanced features
    return basic_bytes.astype(np.float32) / 255.0, enhanced_features.astype(np.float32) / 255.0

def load_enhanced_multi_input(path, transform=None):
    """Enhanced version of load_multi_input with better metadata features"""
    img = Image.open(path)

    # Augmentation
    rgb_img = img.convert('RGB')
    if transform:
        aug_img = transform(rgb_img)
    else:
        crop = transforms.CenterCrop((256, 256))
        aug_img = crop(rgb_img)

    # Enhanced metadata features
    basic_meta, enhanced_meta = extract_enhanced_metadata_features(path)

    # Combine metadata features (basic + enhanced)
    meta = torch.cat([torch.from_numpy(basic_meta), torch.from_numpy(enhanced_meta)])

    # Alpha path
    if img.mode == 'RGBA':
        alpha_plane = np.array(img.split()[-1]).astype(np.float32) / 255.0
        alpha = torch.from_numpy(alpha_plane).unsqueeze(0)
    else:
        alpha = torch.zeros(1, img.height, img.width)
    alpha = transforms.CenterCrop((256, 256))(alpha)

    # LSB path
    lsb_r = (np.array(aug_img)[:, :, 0] & 1).astype(np.float32)
    lsb_g = (np.array(aug_img)[:, :, 1] & 1).astype(np.float32)
    lsb_b = (np.array(aug_img)[:, :, 2] & 1).astype(np.float32)
    lsb = torch.from_numpy(np.stack([lsb_r, lsb_g, lsb_b], axis=0))

    # Palette path
    if img.mode == 'P':
        palette_bytes = np.array(img.getpalette(), dtype=np.uint8)
        palette_bytes = np.pad(palette_bytes, (0, 768 - len(palette_bytes)), 'constant')
    else:
        palette_bytes = np.zeros(768, dtype=np.uint8)
    palette = torch.from_numpy(palette_bytes.astype(np.float32) / 255.0)

    return meta, alpha, lsb, palette

# --- Worker-specific globals ---
SESSION = None
METHOD_MAP = {0: "alpha", 1: "palette", 2: "lsb.rgb", 3: "exif", 4: "raw"}

def init_worker(model_path):
    """Initializer for each worker process."""
    global SESSION
    SESSION = onnxruntime.InferenceSession(model_path)

def _scan_logic(image_path, session, extract_message=False):
    """The actual scanning logic, independent of the execution context."""
    global METHOD_MAP
    try:
        meta, alpha, lsb, palette = load_enhanced_multi_input(image_path)

        # Add batch dimension for ONNX model
        meta = meta.unsqueeze(0)
        alpha = alpha.unsqueeze(0)
        lsb = lsb.unsqueeze(0)
        palette = palette.unsqueeze(0)

        # Run inference
        input_feed = {
            'meta': meta.numpy(),
            'alpha': alpha.numpy(),
            'lsb': lsb.numpy(),
            'palette': palette.numpy()
        }
        stego_logits, _, method_id, method_probs, _ = session.run(None, input_feed)

        # Process results
        stego_prob = 1 / (1 + np.exp(-stego_logits[0][0]))

        # Check for EXIF content to correct misclassification
        has_exif = False
        try:
            with open(image_path, 'rb') as f:
                raw = f.read()
            if b'\xFF\xE1' in raw:
                pos = raw.find(b'\xFF\xE1')
                if pos != -1:
                    length = struct.unpack('>H', raw[pos+2:pos+4])[0]
                    exif_data = raw[pos+4:pos+4+length-2]
                    if len(exif_data) > 10:  # Has meaningful EXIF
                        has_exif = True
        except:
            pass

        # If model predicted raw/EOI but has EXIF and likely stego, correct to exif
        if method_id[0] == 4 and has_exif and stego_prob > 0.5:
            method_id[0] = 3

        # Method-specific thresholds to improve detection
        thresholds = {0: 0.559, 1: 0.486, 2: 0.724, 3: 0.5, 4: 0.652}  # Optimized for F1 score
        threshold = thresholds.get(method_id[0], 0.8)
        is_stego = stego_prob > threshold

        result = {
            "file_path": str(image_path),
            "is_stego": bool(is_stego),
            "stego_probability": float(stego_prob),
            "method_id": int(method_id[0]),
        }

        if is_stego:
            stego_type = METHOD_MAP.get(method_id[0], "unknown")
            result["stego_type"] = stego_type
            result["confidence"] = float(np.max(method_probs))

            if extract_message:
                # Handle method name mapping for extractor
                extractor_method_name = stego_type
                if stego_type == "lsb.rgb":
                    extractor_method_name = "lsb"
                elif stego_type == "raw":
                    extractor_method_name = "eoi"

                if extractor_method_name in extraction_functions:
                    try:
                        extractor = extraction_functions[extractor_method_name]
                        message, _ = extractor(image_path)
                        if message:
                            result["extracted_message"] = message
                    except Exception as e:
                        result["extraction_error"] = str(e)

        return result
    except Exception as e:
        error_message = f"Could not process {image_path}: {e}"
        return {"file_path": str(image_path), "error": error_message}

def scan_image_worker(image_path):
    """The actual scanning logic that runs in each worker process."""
    global SESSION
    # Extraction is disabled for directory scans for performance
    return _scan_logic(image_path, SESSION, extract_message=False)

class StarlightScanner:
    def __init__(self, model_path, num_workers=4):
        self.model_path = model_path
        self.num_workers = num_workers
        print(f"[INIT] Model path set to: {model_path}")
        print(f"[INIT] Fast scanner configured (workers={num_workers})")

    def scan_file(self, file_path):
        """Scans a single image file."""
        session = onnxruntime.InferenceSession(self.model_path)
        # Extraction is enabled for single file scans
        return _scan_logic(file_path, session, extract_message=True)

    def scan_directory(self, path):
        image_paths = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
        
        print(f"\n[SCANNER] Found {len(image_paths)} images to scan")
        print(f"[SCANNER] Initializing {self.num_workers} parallel workers...")

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=init_worker, initargs=(self.model_path,)) as executor:
            with tqdm(total=len(image_paths), desc="Scanning images") as pbar:
                futures = [executor.submit(scan_image_worker, path) for path in image_paths]
                for future in as_completed(futures):
                    res = future.result()
                    if 'error' in res:
                        # Print errors from the main process to ensure they are visible
                        print(f"Warning: {res['error']}")
                    results.append(res)
                    pbar.update(1)
        return results

def main():
    parser = argparse.ArgumentParser(description="Scan a directory or a single file for steganography.")
    parser.add_argument("path", help="The directory or file to scan.")
    parser.add_argument("--model", default="models/detector_balanced.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers for scanning.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    args = parser.parse_args()

    scanner = StarlightScanner(args.model, num_workers=args.workers)
    start_time = time.time()
    
    if os.path.isfile(args.path):
        results = [scanner.scan_file(args.path)]
    else:
        results = scanner.scan_directory(args.path)
        
    end_time = time.time()

    if args.json:
        print(json.dumps(results, indent=2))
        return

    detected_files = [r for r in results if r.get("is_stego")]
    clean_files = [r for r in results if not r.get("is_stego") and "error" not in r]
    errors = [r for r in results if "error" in r]

    print("\n" + "="*60)
    print("SCAN COMPLETE")
    print("="*60)
    total_scanned = len(results) - len(errors)
    print(f"Total images scanned: {total_scanned}")
    print(f"Steganography detected: {len(detected_files)}")

    if detected_files:
        print("\nDetected files:")
        for i, r in enumerate(detected_files):
            if i < 20:
                message_info = f", Message: '{r['extracted_message'][:50]}...'" if r.get('extracted_message') else ""
                print(f"  - {os.path.basename(r['file_path'])} (Predicted: {r['stego_type']}, Confidence: {r.get('confidence', 0):.1%}{message_info})")
        if len(detected_files) > 20:
            print(f"  ... and {len(detected_files) - 20} more.")

    print(f"\nClean images: {len(clean_files)}")
    if errors:
        print(f"Errors on {len(errors)} images.")

    elapsed_time = end_time - start_time
    images_per_sec = total_scanned / elapsed_time if elapsed_time > 0 else 0
    print(f"Scan time: {elapsed_time:.2f} seconds ({images_per_sec:.1f} images/sec)")

    if detected_files:
        print("\nSteganography types found:")
        type_counts = {}
        for r in detected_files:
            stype = r['stego_type']
            type_counts[stype] = type_counts.get(stype, 0) + 1
        for stype, count in type_counts.items():
            print(f"  {stype}: {count}")

if __name__ == "__main__":
    main()
