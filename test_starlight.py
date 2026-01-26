#!/usr/bin/env python3
"""
scan_datasets.py - Comprehensive dataset scanning script
Loops through all submission datasets and validation set to provide accuracy summaries.
Directly calls scanner functions to preserve progress bars and real-time output.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Import scanner module
from scanner import StarlightScanner


def scan_directory_and_parse(scanner, dir_path, dataset_name, show_details=False):
    """Scan a directory and return parsed results."""
    if not os.path.isdir(dir_path):
        print(f"⚠️  Directory not found: {dir_path}")
        return None

    print(f"📁 Scanning {dataset_name}...")

    # Call scanner directly - progress bar will show naturally
    results = scanner.scan_directory(dir_path, quiet=False)

    if not results:
        print(f"   ⚠️  No results returned")
        return None

    # Parse results
    total_files = len(results)
    detected_files = [r for r in results if r.get("is_stego")]
    clean_files = [r for r in results if not r.get("is_stego") and "error" not in r]
    errors = [r for r in results if "error" in r]

    detected_count = len(detected_files)
    error_count = len(errors)

    # Calculate percentage
    rate = (detected_count * 100.0 / total_files) if total_files > 0 else 0.0

    # Show details if requested
    if show_details:
        print(f"   📊 Total files: {total_files}")
        print(f"   ✅ Detected: {detected_count}")
        print(f"   📈 Rate: {rate:.1f}%")
        if error_count > 0:
            print(f"   ⚠️  Errors: {error_count}")
        else:
            print(f"   ✅ No errors")

        # Show detected files if any
        if detected_files:
            print(f"   🔍 Detected files:")
            for i, r in enumerate(detected_files[:10]):
                filename = os.path.basename(r["file_path"])
                stego_type = r.get("stego_type", "unknown")
                confidence = r.get("confidence", 0)
                print(f"     - {filename} ({stego_type}, {confidence:.1%})")
            if len(detected_files) > 10:
                print(f"     ... and {len(detected_files) - 10} more")
        print()

    return {
        "dataset_name": dataset_name,
        "total_files": total_files,
        "detected": detected_count,
        "rate": rate,
        "errors": error_count,
        "results": results,
        "detected_files": detected_files,
    }


def calculate_fp_rate_by_type(clean_results):
    """Calculate false positive rate grouped by steganography type."""
    # Collect false positives by type they were misidentified as
    fp_by_type = defaultdict(lambda: {"count": 0, "total": 0})

    for scan_data in clean_results:
        results = scan_data.get("results", [])
        total_scanned = len(results)
        fp_count = len(scan_data.get("detected_files", []))

        # For each detected file in the clean directory, track what type it was misidentified as
        for detected in scan_data.get("detected_files", []):
            stego_type = detected.get("stego_type", "unknown")
            fp_by_type[stego_type]["count"] += 1

        # Add total files to each type's count for averaging
        if fp_count > 0:
            for detected in scan_data.get("detected_files", []):
                stego_type = detected.get("stego_type", "unknown")
                fp_by_type[stego_type]["total"] += total_scanned

    # Calculate rates
    fp_rates = {}
    for stego_type, data in fp_by_type.items():
        rate = (data["count"] * 100.0 / data["total"]) if data["total"] > 0 else 0.0
        fp_rates[stego_type] = {
            "false_positives": data["count"],
            "total_clean_files": data["total"],
            "rate": rate,
        }

    return fp_rates


def calculate_fn_rate_by_type(stego_results):
    """Calculate false negative rate grouped by actual steganography type."""
    # Collect false negatives by actual steganography type
    fn_by_type = defaultdict(lambda: {"detected": 0, "total": 0})

    for scan_data in stego_results:
        results = scan_data.get("results", [])
        detected_files = {r["file_path"] for r in scan_data.get("detected_files", [])}

        # Count files by actual steganography type (from filename)
        for result in results:
            if "error" in result:
                continue

            filepath = result["file_path"]
            filename = os.path.basename(filepath)

            # Extract steganography type from filename
            actual_stego_type = "unknown"
            for stego_method in ["alpha", "palette", "lsb.rgb", "exif", "raw", "eoi"]:
                if stego_method in filename.lower():
                    actual_stego_type = stego_method
                    break

            fn_by_type[actual_stego_type]["total"] += 1

            # Check if this file was detected (true positive)
            if filepath not in detected_files:
                # This is a false negative
                fn_by_type[actual_stego_type]["detected"] += 1

    # Calculate rates (false negatives are undetected files)
    fn_rates = {}
    for stego_type, data in fn_by_type.items():
        total_files = data["total"]
        undetected_files = data["detected"]  # These are false negatives
        detected_files = total_files - undetected_files
        fn_rate = (undetected_files * 100.0 / total_files) if total_files > 0 else 0.0

        fn_rates[stego_type] = {
            "false_negatives": undetected_files,
            "true_positives": detected_files,
            "total_stego_files": total_files,
            "fn_rate": fn_rate,
            "detection_rate": 100.0 - fn_rate,
        }

    return fn_rates


def print_summary_table(title, results_list, metric_name):
    """Print a summary table for results."""
    print()
    print(f"{title}:")
    print(
        "┌─────────────────────────────────┬─────────┬─────────────┬───────────┬─────────┐"
    )

    if "False Pos" in metric_name:
        print(
            "│ Dataset                         │ Files   │ False Pos   │ Rate (%)  │ Errors  │"
        )
    else:
        print(
            "│ Dataset                         │ Files   │ Detected    │ Rate (%)  │ Errors  │"
        )

    print(
        "├─────────────────────────────────┼─────────┼─────────────┼───────────┼─────────┤"
    )

    for item in results_list:
        dataset = item["dataset_name"][:31].ljust(31)
        total = item["total_files"]
        detected = item["detected"]
        rate = item["rate"]
        errors = item["errors"]
        print(
            f"│ {dataset} │ {total:7d} │ {detected:11d} │ {rate:9.1f} │ {errors:7d} │"
        )

    print(
        "└─────────────────────────────────┴─────────┴─────────────┴───────────┴─────────┘"
    )


def print_fp_rate_by_type_table(fp_rates):
    """Print false positive rate breakdown by steganography type."""
    if not fp_rates:
        return

    print()
    print("❌ FALSE POSITIVE RATE BY STEGANOGRAPHY TYPE:")
    print("┌──────────────┬─────────────────┬────────────────┬───────────┐")
    print("│ Stego Type   │ False Positives │ Total Scanned  │ Rate (%)  │")
    print("├──────────────┼─────────────────┼────────────────┼───────────┤")

    for stego_type in sorted(fp_rates.keys()):
        data = fp_rates[stego_type]
        fp_count = data["false_positives"]
        total = data["total_clean_files"]
        rate = data["rate"]
        stego_type_str = stego_type[:12].ljust(12)
        print(f"│ {stego_type_str} │ {fp_count:15d} │ {total:14d} │ {rate:9.1f} │")

    print("└──────────────┴─────────────────┴────────────────┴───────────┘")


def print_fn_rate_by_type_table(fn_rates):
    """Print false negative rate breakdown by steganography type."""
    if not fn_rates:
        return

    print()
    print("🔍 FALSE NEGATIVE RATE BY STEGANOGRAPHY TYPE:")
    print(
        "┌──────────────┬──────────────────┬─────────────────┬─────────────┬───────────┐"
    )
    print(
        "│ Stego Type   │ False Negatives  │ True Positives  │ Total Files │ FN Rate   │"
    )
    print(
        "├──────────────┼──────────────────┼─────────────────┼─────────────┼───────────┤"
    )

    for stego_type in sorted(fn_rates.keys()):
        data = fn_rates[stego_type]
        fn_count = data["false_negatives"]
        tp_count = data["true_positives"]
        total = data["total_stego_files"]
        fn_rate = data["fn_rate"]
        stego_type_str = stego_type[:12].ljust(12)
        print(
            f"│ {stego_type_str} │ {fn_count:16d} │ {tp_count:15d} │ {total:11d} │ {fn_rate:9.1f} │"
        )

    print(
        "└──────────────┴──────────────────┴─────────────────┴─────────────┴───────────┘"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Scan datasets and provide comprehensive accuracy summaries"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="models/detector_balanced.pth",
        help="Path to PyTorch model file (default: models/detector_balanced.pth)",
    )
    parser.add_argument(
        "-d",
        "--details",
        action="store_true",
        help="Show detailed file-by-file results",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

    # Set workers to CPU count if not specified
    if args.workers is None:
        args.workers = os.cpu_count() or 8

    # Check if model exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    print("=" * 60)
    print("🔍 Project Starlight - Dataset Scanning Tool")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Details: {args.details}")
    print()

    # Initialize scanner
    scanner = StarlightScanner(args.model, num_workers=args.workers, quiet=False)

    # Arrays to store results
    clean_results = []
    stego_results = []

    print("🔍 Discovering datasets...")
    print()

    # Find all submission directories
    datasets_path = Path("datasets")
    submission_dirs = sorted(datasets_path.glob("*_submission_*"))

    if not submission_dirs:
        print("⚠️  No submission directories found in datasets/")

    for submission_dir in submission_dirs:
        dataset_name = submission_dir.name

        # Scan clean directory
        clean_scan = scan_directory_and_parse(
            scanner, submission_dir / "clean", f"{dataset_name} (clean)", args.details
        )
        if clean_scan:
            clean_results.append(clean_scan)

        # Scan stego directory
        stego_scan = scan_directory_and_parse(
            scanner, submission_dir / "stego", f"{dataset_name} (stego)", args.details
        )
        if stego_scan:
            stego_results.append(stego_scan)

    # Also scan validation set if it exists
    val_path = datasets_path / "val"
    if val_path.is_dir():
        val_clean_scan = scan_directory_and_parse(
            scanner, val_path / "clean", "validation (clean)", args.details
        )
        if val_clean_scan:
            clean_results.append(val_clean_scan)

        val_stego_scan = scan_directory_and_parse(
            scanner, val_path / "stego", "validation (stego)", args.details
        )
        if val_stego_scan:
            stego_results.append(val_stego_scan)

    # Print summary tables
    print()
    print("=" * 60)
    print("📊 SUMMARY RESULTS")
    print("=" * 60)

    if clean_results:
        print_summary_table(
            "🧹 CLEAN DIRECTORIES (False Positives)", clean_results, "False Pos"
        )

    if stego_results:
        print_summary_table(
            "🎯 STEGO DIRECTORIES (Detection)", stego_results, "Detected"
        )

    # Calculate overall statistics
    print()
    print("📈 OVERALL PERFORMANCE:")

    total_clean_files = sum(r["total_files"] for r in clean_results)
    total_clean_fps = sum(r["detected"] for r in clean_results)
    total_stego_files = sum(r["total_files"] for r in stego_results)
    total_stego_detected = sum(r["detected"] for r in stego_results)

    overall_fp_rate = (
        (total_clean_fps * 100.0 / total_clean_files) if total_clean_files > 0 else 0.0
    )
    overall_detection_rate = (
        (total_stego_detected * 100.0 / total_stego_files)
        if total_stego_files > 0
        else 0.0
    )

    print("┌─────────────────────────────────┬───────────┬─────────────┐")
    print("│ Metric                          │ Count     │ Rate (%)    │")
    print("├─────────────────────────────────┼───────────┼─────────────┤")
    print(
        f"│ Total Clean Files               │ {total_clean_files:9d} │ {overall_fp_rate:11.2f} │"
    )
    print(
        f"│ Total Stego Files               │ {total_stego_files:9d} │ {overall_detection_rate:11.2f} │"
    )
    print("├─────────────────────────────────┼───────────┼─────────────┤")
    print(
        f"│ False Positives (Clean)         │ {total_clean_fps:9d} │ {overall_fp_rate:11.2f} │"
    )
    print(
        f"│ True Positives (Stego)          │ {total_stego_detected:9d} │ {overall_detection_rate:11.2f} │"
    )
    print("└─────────────────────────────────┴───────────┴─────────────┘")

    # Calculate and print false positive rate by steganography type
    fp_rates = calculate_fp_rate_by_type(clean_results)
    if fp_rates:
        print_fp_rate_by_type_table(fp_rates)

    # Calculate and print false negative rate by steganography type
    fn_rates = calculate_fn_rate_by_type(stego_results)
    if fn_rates:
        print_fn_rate_by_type_table(fn_rates)

    # Performance assessment
    print()
    print("🎯 PERFORMANCE ASSESSMENT:")

    if overall_fp_rate < 1.0:
        print("✅ \033[32mFalse positive rate: EXCELLENT (< 1%)\033[0m")
    elif overall_fp_rate < 5.0:
        print("✅ \033[33mFalse positive rate: GOOD (< 5%)\033[0m")
    else:
        print("❌ \033[31mFalse positive rate: NEEDS IMPROVEMENT (> 5%)\033[0m")

    if overall_detection_rate > 95.0:
        print("✅ \033[32mDetection rate: EXCELLENT (> 95%)\033[0m")
    elif overall_detection_rate > 85.0:
        print("✅ \033[33mDetection rate: GOOD (> 85%)\033[0m")
    else:
        print("❌ \033[31mDetection rate: NEEDS IMPROVEMENT (< 85%)\033[0m")

    print()
    print("=" * 60)
    print("✅ Scan completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
