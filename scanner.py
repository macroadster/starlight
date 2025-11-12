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
from scripts.starlight_extractor import extraction_functions
from trainer import load_enhanced_multi_input # Import from trainer.py




# --- Worker-specific globals ---
SESSION = None
METHOD_MAP = {0: "alpha", 1: "palette", 2: "lsb.rgb", 3: "exif", 4: "raw"}

def init_worker(model_path):
    """Initializer for each worker process."""
    global SESSION
    import piexif # Re-import piexif in the worker process
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
        # Use a numerically stable sigmoid to avoid overflow warnings
        logit = stego_logits[0][0]
        if logit >= 0:
            stego_prob = 1 / (1 + np.exp(-logit))
        else:
            # Use the equivalent alternative form to avoid overflow for large negative logits
            stego_prob = np.exp(logit) / (1 + np.exp(logit))

        # Method-specific thresholds to improve detection
        thresholds = {0: 0.559, 1: 0.486, 2: 0.724, 3: 0.5, 4: 0.5}  # Optimized for F1 score
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
    def __init__(self, model_path, num_workers=4, quiet=False):
        self.model_path = model_path
        self.num_workers = num_workers
        if not quiet:
            print(f"[INIT] Model path set to: {model_path}")
            print(f"[INIT] Fast scanner configured (workers={num_workers})")

    def scan_file(self, file_path):
        """Scans a single image file."""
        session = onnxruntime.InferenceSession(self.model_path)
        # Extraction is enabled for single file scans
        return _scan_logic(file_path, session, extract_message=True)

    def scan_directory(self, path, quiet=False):
        image_paths = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
        
        if not quiet:
            print(f"\n[SCANNER] Found {len(image_paths)} images to scan")
            print(f"[SCANNER] Initializing {self.num_workers} parallel workers...")

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=init_worker, initargs=(self.model_path,)) as executor:
            progress_bar = tqdm(total=len(image_paths), desc="Scanning images", disable=quiet)
            with progress_bar:
                futures = [executor.submit(scan_image_worker, path) for path in image_paths]
                for future in as_completed(futures):
                    res = future.result()
                    if 'error' in res and not quiet:
                        # Print errors from the main process to ensure they are visible
                        print(f"Warning: {res['error']}")
                    results.append(res)
                    progress_bar.update(1)
        return results

def main():
    parser = argparse.ArgumentParser(description="Scan a directory or a single file for steganography.")
    parser.add_argument("path", help="The directory or file to scan.")
    parser.add_argument("--model", default="models/detector_balanced.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers for scanning.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    args = parser.parse_args()

    scanner = StarlightScanner(args.model, num_workers=args.workers, quiet=args.json)
    start_time = time.time()
    
    if os.path.isfile(args.path):
        results = [scanner.scan_file(args.path)]
    else:
        results = scanner.scan_directory(args.path, quiet=args.json)
        
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
