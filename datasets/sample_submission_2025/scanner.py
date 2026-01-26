#!/usr/bin/env python3
"""
Fast Steganography Scanner - Optimized for speed

Key optimizations:
1. Cached ensemble model (singleton)
2. Parallel processing with ThreadPoolExecutor
3. Quick scan mode (detection only)
4. Batch result processing
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

# Import LightScanner
from light_scanner import LightScanner

# Import extraction functions
try:
    from starlight_extractor import (
        extract_alpha,
        extract_palette,
        extract_lsb,
        extract_exif,
        extract_eoi,
    )
except ImportError:
    print("Warning: starlight_extractor not available, extraction disabled")
    extract_alpha = extract_palette = extract_lsb = extract_exif = extract_eoi = (
        lambda *args: ("", 0)
    )

# --- GLOBAL SCANNER CACHE ---
_scanner_instance = None
_scanner_lock = threading.Lock()


def get_scanner_instance():
    """Get or create cached scanner instance (thread-safe singleton)"""
    global _scanner_instance
    if _scanner_instance is None:
        with _scanner_lock:
            if _scanner_instance is None:
                print("[INIT] Creating cached LightScanner (one-time initialization)")
                _scanner_instance = LightScanner()
            else:
                print("[INIT] Using cached LightScanner")
    else:
        print("[INIT] Using cached LightScanner")
    return _scanner_instance


# --- FAST SCANNER ---
class FastStegoScanner:
    def __init__(self, quick_mode=False, max_workers=4, extract_override=False):
        self.quick_mode = quick_mode
        self.max_workers = max_workers
        self.extract_override = extract_override
        print(
            f"[INIT] Fast scanner initialized (quick_mode={quick_mode}, workers={max_workers}, extract_override={extract_override})"
        )
        self.detector = get_scanner_instance()

    def detect_single(self, img_path):
        """Fast detection for a single image"""
        try:
            result = self.detector.scan(str(img_path))
            return {
                "file": str(img_path),
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "predicted_class": "stego" if result["stego"] else "clean",
                "confidence": abs(result["probability"] - 0.5) * 2,
                "ensemble_probability": result["probability"],
                "stego_type": result["method"],
                "is_stego": result["stego"],
            }
        except Exception as e:
            return {"file": str(img_path), "status": "error", "error": str(e)}

    def extract_single(self, img_path, result):
        """Extract messages from a detected stego image"""
        if not result.get("is_stego", False) or (
            self.quick_mode and not self.extract_override
        ):
            return result

        try:
            img_path_str = str(img_path)
            extraction_map = {
                "alpha": lambda: extract_alpha(img_path_str),
                "palette": lambda: extract_palette(img_path_str),
                "lsb": lambda: extract_lsb(img_path_str),
                "exif": lambda: extract_exif(img_path_str),
                "eoi": lambda: extract_eoi(img_path_str),
            }

            extracted_messages = {}
            for algo, extractor_func in extraction_map.items():
                try:
                    message, _ = extractor_func()
                    if message:
                        extracted_messages[algo] = (
                            str(message)[:200] + "..."
                            if len(str(message)) > 200
                            else str(message)
                        )
                except Exception:
                    pass

            result["extracted_message"] = extracted_messages
        except Exception as e:
            result["extraction_error"] = str(e)

        return result

    def scan_file(self, img_path, extract_messages=True):
        """Scan a single file with optional extraction"""
        result = self.detect_single(img_path)
        if result["status"] == "success":
            if extract_messages:
                result = self.extract_single(img_path, result)
        return result

    def scan_directory(self, directory, recursive=True, output_file=None, detail=False):
        """Fast directory scan with parallel processing"""
        directory = Path(directory)
        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".webp",
            ".tiff",
            ".tif",
        }

        # Find all image files
        if recursive:
            image_files = [
                f for f in directory.rglob("*") if f.suffix.lower() in image_extensions
            ]
        else:
            image_files = [
                f for f in directory.glob("*") if f.suffix.lower() in image_extensions
            ]

        print(f"\n[SCANNER] Found {len(image_files)} images to scan")
        print(f"[SCANNER] Using {self.max_workers} parallel workers")

        results = []
        stego_detections = []

        # Parallel processing
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.scan_file, img_path): img_path
                for img_path in image_files
            }

            # Process completed tasks with progress bar
            for future in tqdm(
                as_completed(future_to_path),
                total=len(image_files),
                desc="Scanning images",
            ):
                result = future.result()
                results.append(result)

                if result.get("is_stego", False):
                    stego_detections.append((Path(result["file"]), result))

        scan_time = time.time() - start_time

        # Display results
        if detail and stego_detections:
            print(f"\n{'='*60}")
            print(f"DETECTIONS FOUND")
            print(f"{'='*60}")
            for img_path, result in stego_detections:
                print(f"\n[FOUND] {img_path.name}")
                print(f"  Path: {img_path}")
                print(
                    f"  Type: {result['predicted_class']} (confidence: {result['confidence']:.2%})"
                )
                print(f"  Method: {result['stego_type']}")
                if result.get("extracted_message"):
                    for algo, msg in result["extracted_message"].items():
                        print(f"  Message ({algo}): {msg}")

        # Summary
        print(f"\n{'='*60}")
        print(f"SCAN COMPLETE")
        print(f"{'='*60}")
        print(f"Total images scanned: {len(image_files)}")
        print(f"Steganography detected: {len(stego_detections)}")
        print(f"Clean images: {len(image_files) - len(stego_detections)}")
        print(
            f"Scan time: {scan_time:.2f} seconds ({len(image_files)/scan_time:.1f} images/sec)"
        )

        # Type breakdown
        type_counts = {}
        for result in results:
            if result.get("is_stego", False):
                stego_type = result.get("stego_type", "unknown")
                type_counts[stego_type] = type_counts.get(stego_type, 0) + 1

        if type_counts:
            print(f"\nSteganography types found:")
            for stego_type, count in sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {stego_type}: {count}")

        # Save results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Fast Steganography Scanner")
    parser.add_argument("target", help="Image file or directory to scan")
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Quick scan: skip extraction (default for directories)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan subdirectories",
    )
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument(
        "--detail", action="store_true", help="Display detailed detection results"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract messages from detected stego images (overrides quick mode)",
    )

    args = parser.parse_args()

    # Check if target exists
    if not os.path.exists(args.target):
        print(f"Error: Target not found: {args.target}")
        sys.exit(1)

    target_path = Path(args.target)

    if target_path.is_file():
        # For single files, default to extraction (not quick)
        scanner = FastStegoScanner(
            quick_mode=False, max_workers=args.workers, extract_override=args.extract
        )
        print(f"\n[SCANNING] Single file: {args.target}")
        # For single files, perform extraction by default (unless --quick specified)
        extract_by_default = True  # Since quick_mode=False
        result = scanner.scan_file(target_path, extract_messages=extract_by_default)

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"File: {result['file']}")
        if result["status"] == "success":
            print(
                f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.2%})"
            )
            print(f"Method: {result['stego_type']}")
            if result.get("extracted_message"):
                extracted = result["extracted_message"]
                if isinstance(extracted, dict):
                    for algo_name, msg_content in extracted.items():
                        print(f"  Message ({algo_name}): {msg_content}")
                else:
                    print(f"  Message: {extracted}")
            if result.get("extraction_error"):
                print(f"  Extraction Error: {result['extraction_error']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif target_path.is_dir():
        scanner = FastStegoScanner(
            quick_mode=True, max_workers=args.workers, extract_override=args.extract
        )
        print(f"\n[SCANNING] Directory: {args.target}")
        # For directory scans, use quick mode by default (no extraction)
        results = scanner.scan_directory(
            args.target,
            recursive=args.recursive,
            output_file=args.output,
            detail=args.detail,
        )

    else:
        print(f"Error: Invalid target (not a file or directory)")
        sys.exit(1)


if __name__ == "__main__":
    main()
