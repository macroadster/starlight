import onnxruntime
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from starlight_utils import load_multi_input
import time

# --- Worker-specific globals ---
SESSION = None
METHOD_MAP = {0: "alpha", 1: "palette", 2: "lsb.rgb", 3: "exif", 4: "raw"}

def init_worker(model_path):
    """Initializer for each worker process."""
    global SESSION
    SESSION = onnxruntime.InferenceSession(model_path)

def scan_image_worker(image_path):
    """The actual scanning logic that runs in each worker process."""
    global SESSION, METHOD_MAP
    try:
        meta, alpha, lsb, palette = load_multi_input(image_path)
        
        # Add batch dimension for ONNX model
        meta = meta.unsqueeze(0)
        alpha = alpha.unsqueeze(0)
        lsb = lsb.unsqueeze(0)
        palette = palette.unsqueeze(0)

        # Run inference
        input_feed = {
            'metadata': meta.numpy(),
            'alpha': alpha.numpy(),
            'lsb': lsb.numpy(),
            'palette': palette.numpy()
        }
        stego_logits, _, method_id, method_probs = SESSION.run(None, input_feed)
        
        # Process results
        stego_prob = 1 / (1 + np.exp(-stego_logits[0]))
        is_stego = stego_prob > 0.5
        
        result = {
            "file_path": str(image_path),
            "is_stego": bool(is_stego),
            "stego_probability": float(stego_prob),
        }
        
        if is_stego:
            result["stego_type"] = METHOD_MAP.get(method_id[0], "unknown")
            result["confidence"] = float(np.max(method_probs))
            
        return result
    except Exception as e:
        error_message = f"Could not process {image_path}: {e}"
        # This print will appear in the worker's output, may not be visible on main console
        # print(f"Warning: {error_message}") 
        return {"file_path": str(image_path), "error": error_message}

class StarlightScanner:
    def __init__(self, model_path, num_workers=4):
        self.model_path = model_path
        self.num_workers = num_workers
        print(f"[INIT] Model path set to: {model_path}")
        print(f"[INIT] Fast scanner configured (workers={num_workers})")

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
    parser = argparse.ArgumentParser(description="Scan a directory for steganography.")
    parser.add_argument("directory", help="The directory to scan.")
    parser.add_argument("--model", default="models/detector_multi_stream.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers for scanning.")
    args = parser.parse_args()

    scanner = StarlightScanner(args.model, num_workers=args.workers)
    start_time = time.time()
    results = scanner.scan_directory(args.directory)
    end_time = time.time()
    
    detected_files = [r for r in results if r.get("is_stego")]
    clean_files = [r for r in results if not r.get("is_stego") and "error" not in r]
    errors = [r for r in results if "error" in r]

    print("\n" + "="*60)
    print("SCAN COMPLETE")
    print("="*60)
    print(f"Total images scanned: {len(results)}")
    print(f"Steganography detected: {len(detected_files)}")
    
    if detected_files:
        print("\nDetected files:")
        for i, r in enumerate(detected_files):
            if i < 20:
                print(f"  - {os.path.basename(r['file_path'])} (Predicted: {r['stego_type']}, Confidence: {r['confidence']:.1%})")
        if len(detected_files) > 20:
            print(f"  ... and {len(detected_files) - 20} more.")

    print(f"\nClean images: {len(clean_files)}")
    if errors:
        print(f"Errors on {len(errors)} images.")

    elapsed_time = end_time - start_time
    images_per_sec = len(results) / elapsed_time if elapsed_time > 0 else 0
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
