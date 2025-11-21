import argparse
import os
import sys
from tqdm import tqdm

# Add the parent directory to the Python path to allow importing from 'scanner'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner import StarlightScanner

def run_regression(clean_dir, model_path, num_workers):
    """
    Scans a directory of clean images and calculates the false positive rate.
    """
    if not os.path.isdir(clean_dir):
        print(f"Error: Directory not found at '{clean_dir}'")
        return

    scanner = StarlightScanner(model_path, num_workers=num_workers, quiet=True)
    results = scanner.scan_directory(clean_dir, quiet=True)

    total_scanned = 0
    false_positives = 0

    for result in tqdm(results, desc="Analyzing results"):
        if "error" not in result:
            total_scanned += 1
            if result.get("is_stego"):
                false_positives += 1
                print(f"  False positive: {result['file_path']}")

    if total_scanned > 0:
        fp_rate = (false_positives / total_scanned) * 100
        print(f"\nFalse Positive Rate: {fp_rate:.2f}% ({false_positives}/{total_scanned})")
    else:
        print("\nNo images were scanned.")

def main():
    parser = argparse.ArgumentParser(description="Run a regression test to calculate the false positive rate on a clean dataset.")
    parser.add_argument("clean_dir", help="Path to the directory of clean images.")
    parser.add_argument("--model", default="models/detector_balanced.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers for scanning.")
    args = parser.parse_args()

    run_regression(args.clean_dir, args.model, args.workers)

if __name__ == "__main__":
    main()
