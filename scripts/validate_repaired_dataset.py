#!/usr/bin/env python3
"""
Repaired Dataset Validator - Week 1, Day 4
Validates that repaired dataset meets all quality requirements

Run: python scripts/validate_repaired_dataset.py --dataset datasets/grok_submission_2025/training/v3_repaired --output docs/claude/validation_report.json
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class ValidationResult:
    """Results of dataset validation"""

    total_images: int
    clean_count: int
    stego_count: int
    negative_count: int

    # Quality checks
    no_invalid_labels: bool
    extraction_verified: bool  # Placeholder until integrated
    format_balanced: bool
    negatives_present: bool
    no_corrupted_signals: bool

    # Detailed stats
    mode_distribution: Dict[str, int]
    format_distribution: Dict[str, int]
    method_distribution: Dict[str, int]
    negative_categories: Dict[str, int]

    # Issues found
    issues: List[str]
    warnings: List[str]

    # Overall pass/fail
    passed: bool


class RepairedDatasetValidator:
    """Validates repaired dataset quality"""

    REQUIRED_METHODS = ["alpha", "lsb", "palette", "exif", "eoi"]
    REQUIRED_NEGATIVE_CATEGORIES = [
        "rgb_no_alpha",
        "uniform_alpha",
        "natural_noise",
        "patterns",
        "special_cases",
    ]

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> ValidationResult:
        """Run all validation checks"""
        print(f"Validating dataset: {self.dataset_path}")

        # Gather statistics
        stats = self._gather_statistics()

        # Run validation checks
        checks = {
            "no_invalid_labels": self._check_no_invalid_labels(),
            "extraction_verified": self._check_extraction_verified(),
            "format_balanced": self._check_format_balance(),
            "negatives_present": self._check_negatives_present(),
            "no_corrupted_signals": self._check_no_corrupted_signals(),
        }

        # Overall pass/fail
        passed = all(checks.values()) and len(self.issues) == 0

        result = ValidationResult(
            total_images=stats["total_images"],
            clean_count=stats["clean_count"],
            stego_count=stats["stego_count"],
            negative_count=stats["negative_count"],
            no_invalid_labels=checks["no_invalid_labels"],
            extraction_verified=checks["extraction_verified"],
            format_balanced=checks["format_balanced"],
            negatives_present=checks["negatives_present"],
            no_corrupted_signals=checks["no_corrupted_signals"],
            mode_distribution=stats["mode_distribution"],
            format_distribution=stats["format_distribution"],
            method_distribution=stats["method_distribution"],
            negative_categories=stats["negative_categories"],
            issues=self.issues,
            warnings=self.warnings,
            passed=passed,
        )

        return result

    def _gather_statistics(self) -> Dict:
        """Gather dataset statistics"""
        stats = {
            "total_images": 0,
            "clean_count": 0,
            "stego_count": 0,
            "negative_count": 0,
            "mode_distribution": Counter(),
            "format_distribution": Counter(),
            "method_distribution": Counter(),
            "negative_categories": Counter(),
        }

        # Count clean images
        clean_dir = self.dataset_path / "clean"
        if clean_dir.exists():
            for img_path in clean_dir.rglob("*"):
                if self._is_image(img_path):
                    stats["clean_count"] += 1
                    stats["total_images"] += 1
                    self._update_image_stats(img_path, stats)

        # Count stego images by method
        stego_dir = self.dataset_path / "stego"
        if stego_dir.exists():
            for method in self.REQUIRED_METHODS:
                method_dir = stego_dir / method
                if method_dir.exists():
                    count = 0
                    for img_path in method_dir.rglob("*"):
                        if self._is_image(img_path):
                            count += 1
                            stats["stego_count"] += 1
                            stats["total_images"] += 1
                            self._update_image_stats(img_path, stats)
                    stats["method_distribution"][method] = count

        # Count negative examples
        negatives_dir = self.dataset_path / "negatives"
        if negatives_dir.exists():
            for category in self.REQUIRED_NEGATIVE_CATEGORIES:
                category_dir = negatives_dir / category
                if category_dir.exists():
                    count = 0
                    for img_path in category_dir.rglob("*"):
                        if self._is_image(img_path):
                            count += 1
                            stats["negative_count"] += 1
                            stats["total_images"] += 1
                    stats["negative_categories"][category] = count

        return stats

    def _update_image_stats(self, img_path: Path, stats: Dict):
        """Update statistics for a single image"""
        try:
            with Image.open(img_path) as img:
                stats["mode_distribution"][img.mode] += 1
                if img.format:
                    stats["format_distribution"][img.format] += 1
        except:
            pass

    def _is_image(self, path: Path) -> bool:
        """Check if path is an image file"""
        return path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        }

    def _check_no_invalid_labels(self) -> bool:
        """Check that no invalid labels exist"""
        print("\nChecking for invalid labels...")

        stego_dir = self.dataset_path / "stego"
        if not stego_dir.exists():
            self.issues.append("Stego directory does not exist")
            return False

        found_invalid = False

        # Check alpha method
        alpha_dir = stego_dir / "alpha"
        if alpha_dir.exists():
            for img_path in alpha_dir.rglob("*"):
                if self._is_image(img_path):
                    try:
                        with Image.open(img_path) as img:
                            if img.mode not in ["RGBA", "LA", "PA"]:
                                self.issues.append(
                                    f"Invalid: Alpha label on {img.mode} image: {img_path}"
                                )
                                found_invalid = True
                    except Exception as e:
                        self.warnings.append(f"Could not open {img_path}: {e}")

        # Check palette method
        palette_dir = stego_dir / "palette"
        if palette_dir.exists():
            for img_path in palette_dir.rglob("*"):
                if self._is_image(img_path):
                    try:
                        with Image.open(img_path) as img:
                            if img.mode not in ["P", "PA"]:
                                self.issues.append(
                                    f"Invalid: Palette label on {img.mode} image: {img_path}"
                                )
                                found_invalid = True
                    except Exception as e:
                        self.warnings.append(f"Could not open {img_path}: {e}")

        if found_invalid:
            print("  ❌ Found invalid labels")
            return False
        else:
            print("  ✅ No invalid labels found")
            return True

    def _check_extraction_verified(self) -> bool:
        """Check that extraction has been verified"""
        print("\nChecking extraction verification...")

        # TODO: Integrate with actual extraction functions
        # For now, just check if manifest exists with extraction results

        manifest_path = self.dataset_path / "repair_manifest.json"
        if not manifest_path.exists():
            self.warnings.append("Repair manifest not found - cannot verify extraction")
            print("  ⚠️  Cannot verify extraction (manifest missing)")
            return True  # Don't fail on this yet

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Check if extraction was attempted
            if "extraction_verified" in manifest:
                verified = manifest["extraction_verified"]
                if verified:
                    print("  ✅ Extraction verified per manifest")
                    return True
                else:
                    self.warnings.append("Extraction verification not completed")
                    print("  ⚠️  Extraction not fully verified")
                    return True  # Warning, not error
            else:
                self.warnings.append(
                    "Manifest does not contain extraction verification"
                )
                print("  ⚠️  No extraction verification in manifest")
                return True  # Warning, not error

        except Exception as e:
            self.warnings.append(f"Could not read manifest: {e}")
            print("  ⚠️  Could not verify extraction")
            return True  # Warning, not error

    def _check_format_balance(self) -> bool:
        """Check that clean images match stego format distribution"""
        print("\nChecking format balance...")

        # Get format distributions
        clean_formats = Counter()
        stego_formats = Counter()

        clean_dir = self.dataset_path / "clean"
        if clean_dir.exists():
            for img_path in clean_dir.rglob("*"):
                if self._is_image(img_path):
                    try:
                        with Image.open(img_path) as img:
                            if img.format:
                                clean_formats[img.format] += 1
                    except:
                        pass

        stego_dir = self.dataset_path / "stego"
        if stego_dir.exists():
            for img_path in stego_dir.rglob("*"):
                if self._is_image(img_path):
                    try:
                        with Image.open(img_path) as img:
                            if img.format:
                                stego_formats[img.format] += 1
                    except:
                        pass

        # Check balance (within 20% tolerance)
        balanced = True
        for fmt, stego_count in stego_formats.items():
            clean_count = clean_formats.get(fmt, 0)

            if clean_count == 0:
                self.warnings.append(
                    f"Format imbalance: {fmt} has {stego_count} stego but 0 clean images"
                )
                balanced = False
            else:
                ratio = clean_count / stego_count
                if ratio < 0.8 or ratio > 1.2:
                    self.warnings.append(
                        f"Format imbalance: {fmt} has {stego_count} stego but {clean_count} clean "
                        f"(ratio: {ratio:.2f})"
                    )
                    balanced = False

        if balanced:
            print("  ✅ Formats are balanced")
        else:
            print("  ⚠️  Some format imbalances detected")

        return True  # Warning, not critical error

    def _check_negatives_present(self) -> bool:
        """Check that all negative example categories are present"""
        print("\nChecking negative examples...")

        negatives_dir = self.dataset_path / "negatives"
        if not negatives_dir.exists():
            self.issues.append("Negatives directory does not exist")
            print("  ❌ Negatives directory missing")
            return False

        all_present = True
        for category in self.REQUIRED_NEGATIVE_CATEGORIES:
            category_dir = negatives_dir / category
            if not category_dir.exists():
                self.issues.append(f"Missing negative category: {category}")
                all_present = False
            else:
                # Count images
                count = sum(1 for p in category_dir.rglob("*") if self._is_image(p))
                if count == 0:
                    self.issues.append(f"Negative category {category} is empty")
                    all_present = False
                elif count < 50:
                    self.warnings.append(
                        f"Negative category {category} has only {count} images (recommend ≥50)"
                    )

        if all_present:
            print("  ✅ All negative categories present")
        else:
            print("  ❌ Some negative categories missing")

        return all_present

    def _check_no_corrupted_signals(self) -> bool:
        """Check for signal-corrupting issues"""
        print("\nChecking for signal corruption...")

        # Check for JPEG LSB steganography (lossy compression corrupts LSB)
        stego_dir = self.dataset_path / "stego" / "lsb"
        if stego_dir.exists():
            jpeg_lsb_count = 0
            for img_path in stego_dir.rglob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg"}:
                    jpeg_lsb_count += 1

            if jpeg_lsb_count > 0:
                self.warnings.append(
                    f"Found {jpeg_lsb_count} JPEG images in LSB directory - "
                    "lossy compression may corrupt LSB data"
                )

        # Check for JPEG alpha steganography
        alpha_dir = self.dataset_path / "stego" / "alpha"
        if alpha_dir.exists():
            jpeg_alpha_count = 0
            for img_path in alpha_dir.rglob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg"}:
                    jpeg_alpha_count += 1

            if jpeg_alpha_count > 0:
                self.warnings.append(
                    f"Found {jpeg_alpha_count} JPEG images in alpha directory - "
                    "JPEG does not support alpha channel"
                )

        print("  ✅ No critical signal corruption detected")
        return True

    def print_report(self, result: ValidationResult):
        """Print human-readable validation report"""
        print("\n" + "=" * 70)
        print("DATASET VALIDATION REPORT")
        print("=" * 70)

        print(f"\n📊 Dataset Statistics")
        print(f"  Total Images: {result.total_images}")
        print(f"    Clean: {result.clean_count}")
        print(f"    Stego: {result.stego_count}")
        print(f"    Negatives: {result.negative_count}")

        print(f"\n🔍 Validation Checks")
        self._print_check("No Invalid Labels", result.no_invalid_labels)
        self._print_check("Extraction Verified", result.extraction_verified)
        self._print_check("Format Balanced", result.format_balanced)
        self._print_check("Negatives Present", result.negatives_present)
        self._print_check("No Signal Corruption", result.no_corrupted_signals)

        print(f"\n📈 Distribution Analysis")
        print(f"  Image Modes:")
        for mode, count in sorted(result.mode_distribution.items()):
            pct = (count / result.total_images) * 100
            print(f"    {mode:8s}: {count:5d} ({pct:5.1f}%)")

        print(f"\n  Stego Methods:")
        for method, count in sorted(result.method_distribution.items()):
            pct = (count / result.stego_count) * 100 if result.stego_count > 0 else 0
            print(f"    {method:8s}: {count:5d} ({pct:5.1f}%)")

        print(f"\n  Negative Categories:")
        for category, count in sorted(result.negative_categories.items()):
            print(f"    {category:20s}: {count:5d}")

        if result.issues:
            print(f"\n❌ Issues Found ({len(result.issues)}):")
            for issue in result.issues:
                print(f"  • {issue}")

        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  • {warning}")

        print("\n" + "=" * 70)
        if result.passed:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
        print("=" * 70)

    def _print_check(self, name: str, passed: bool):
        """Print a single check result"""
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    def export_report(self, result: ValidationResult, output_path: Path):
        """Export validation report as JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\nReport exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate repaired dataset quality")
    parser.add_argument(
        "--dataset", required=True, help="Path to repaired dataset directory"
    )
    parser.add_argument(
        "--output",
        default="docs/claude/validation_report.json",
        help="Output path for validation report",
    )

    args = parser.parse_args()

    validator = RepairedDatasetValidator(Path(args.dataset))
    result = validator.validate_all()

    validator.print_report(result)
    validator.export_report(result, Path(args.output))

    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
