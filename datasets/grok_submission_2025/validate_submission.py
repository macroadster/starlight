#!/usr/bin/env python3
"""
validate_submission.py - Complete validation of Starlight submission
Validates both LSB and EXIF models and their integration
"""

import os
import json
import numpy as np
from pathlib import Path
from model.inference import StarlightModel
from aggregate_models import SuperStarlightDetector


def validate_onnx_models():
    """Validate ONNX model files exist and are loadable"""
    print("=== ONNX Model Validation ===")

    required_files = ["model/detector.onnx", "model/extractor.onnx"]

    all_valid = True
    for file_path in required_files:
        if os.path.exists(file_path):
            try:
                # Try to load with ONNX runtime
                import onnxruntime as ort

                session = ort.InferenceSession(file_path)
                print(f"✓ {file_path} - Loaded successfully")
                print(f"  Inputs: {[inp.name for inp in session.get_inputs()]}")
                print(f"  Outputs: {[out.name for out in session.get_outputs()]}")
            except Exception as e:
                print(f"✗ {file_path} - Failed to load: {e}")
                all_valid = False
        else:
            print(f"✗ {file_path} - File not found")
            all_valid = False

    return all_valid


def validate_steganography_modules():
    """Validate LSB and EXIF steganography modules"""
    print("\n=== Steganography Module Validation ===")

    try:
        from lsb_steganography import analyze_lsb

        print("✓ LSB steganography module imported successfully")

        # Test LSB analysis
        test_img = "clean/README_lsb_000.png"
        if os.path.exists(test_img):
            stats, _ = analyze_lsb(test_img)
            print(f"✓ LSB analysis works on {test_img}")
            print(f"  LSB ratios: {stats['lsb_ones_ratio']}")

    except Exception as e:
        print(f"✗ LSB module error: {e}")
        return False

    try:
        from exif_steganography import detect_exif_stego

        print("✓ EXIF steganography module imported successfully")

        # Test EXIF detection
        test_img = "stego/README_exif_000.jpeg"
        if os.path.exists(test_img):
            detection = detect_exif_stego(test_img)
            print(f"✓ EXIF detection works on {test_img}")
            print(f"  Detection result: {detection.get('stego_probability', 0):.3f}")

    except Exception as e:
        print(f"✗ EXIF module error: {e}")
        return False

    return True


def validate_inference_interface():
    """Validate standardized inference interface"""
    print("\n=== Inference Interface Validation ===")

    try:
        model = StarlightModel()
        print("✓ StarlightModel initialized successfully")

        # Test on different image types
        test_cases = [
            ("clean/README_lsb_000.png", "PNG"),
            ("stego/README_exif_000.jpeg", "JPEG"),
        ]

        for img_path, img_type in test_cases:
            if os.path.exists(img_path):
                result = model.predict(img_path)
                required_keys = ["image_path", "stego_probability", "task"]

                missing_keys = [key for key in required_keys if key not in result]
                if not missing_keys:
                    print(
                        f"✓ Inference works on {img_type}: {result['stego_probability']:.3f}"
                    )
                else:
                    print(f"✗ Missing keys in {img_type} result: {missing_keys}")
                    return False

    except Exception as e:
        print(f"✗ Inference interface error: {e}")
        return False

    return True


def validate_ensemble():
    """Validate ensemble model functionality"""
    print("\n=== Ensemble Model Validation ===")

    try:
        model_configs = [
            {"task": "detect", "method": "neural", "weight": 1.0},
            {"task": "detect", "method": "lsb", "weight": 1.0},
            {"task": "detect", "method": "exif", "weight": 1.0},
        ]

        ensemble = SuperStarlightDetector(model_configs)
        print("✓ Ensemble model created successfully")

        # Test ensemble prediction
        test_img = "clean/README_lsb_000.png"
        if os.path.exists(test_img):
            result = ensemble.predict(test_img)
            required_keys = ["ensemble_probability", "predicted", "stego_type"]

            missing_keys = [key for key in required_keys if key not in result]
            if not missing_keys:
                print(
                    f"✓ Ensemble prediction works: {result['ensemble_probability']:.3f}"
                )
                print(f"  Weights: {[f'{w:.3f}' for w in result['model_weights']]}")
            else:
                print(f"✗ Missing keys in ensemble result: {missing_keys}")
                return False

    except Exception as e:
        print(f"✗ Ensemble model error: {e}")
        return False

    return True


def validate_dataset_structure():
    """Validate dataset structure follows Starlight architecture"""
    print("\n=== Dataset Structure Validation ===")

    required_dirs = ["clean", "stego", "model"]

    required_files = [
        "model/inference.py",
        "model/model_card.md",
        "model/requirements.txt",
        "README.md",
    ]

    all_valid = True

    # Check directories
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_valid = False

    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            all_valid = False

    # Check for paired clean/stego files
    clean_files = set(os.listdir("clean")) if os.path.isdir("clean") else set()
    stego_files = set(os.listdir("stego")) if os.path.isdir("stego") else set()

    # Remove .json files from stego for comparison
    stego_images = {f for f in stego_files if not f.endswith(".json")}

    common_files = clean_files & stego_images
    print(f"✓ Found {len(common_files)} paired clean/stego images")

    return all_valid


def run_comprehensive_test():
    """Run comprehensive test on all available images"""
    print("\n=== Comprehensive Detection Test ===")

    # Get all test images
    test_images = []

    clean_dir = Path("clean")
    stego_dir = Path("stego")

    if clean_dir.exists():
        for img_file in clean_dir.glob("*.png"):
            test_images.append((str(img_file), "Clean PNG"))
        for img_file in clean_dir.glob("*.jpeg"):
            test_images.append((str(img_file), "Clean JPEG"))

    if stego_dir.exists():
        for img_file in stego_dir.glob("*.png"):
            test_images.append((str(img_file), "Stego PNG"))
        for img_file in stego_dir.glob("*.jpeg"):
            test_images.append((str(img_file), "Stego JPEG"))

    if not test_images:
        print("✗ No test images found")
        return False

    # Test with ensemble
    model_configs = [
        {"task": "detect", "method": "neural", "weight": 1.0},
        {"task": "detect", "method": "lsb", "weight": 1.0},
        {"task": "detect", "method": "exif", "weight": 1.0},
    ]

    ensemble = SuperStarlightDetector(model_configs)

    results = []
    for img_path, label in test_images[:10]:  # Test first 10 images
        try:
            result = ensemble.predict(img_path)
            results.append(
                {
                    "label": label,
                    "path": img_path,
                    "probability": float(result["ensemble_probability"]),
                    "predicted": bool(result["predicted"]),
                    "type": result["stego_type"],
                }
            )
            print(
                f"{label:12s} - Prob: {result['ensemble_probability']:.3f}, "
                f"Pred: {result['predicted']}, Type: {result['stego_type']}"
            )
        except Exception as e:
            print(f"✗ Error testing {img_path}: {e}")

    # Save comprehensive results
    output = {
        "validation_summary": {
            "total_images_tested": len(results),
            "average_probability": np.mean([r["probability"] for r in results]),
            "positive_predictions": sum([r["predicted"] for r in results]),
        },
        "detailed_results": results,
    }

    with open("model/validation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(
        "\n✓ Comprehensive test complete. Results saved to model/validation_results.json"
    )
    return True


def main():
    """Main validation function"""
    print("Starlight Submission Validation")
    print("=" * 50)

    validation_results = {
        "onnx_models": validate_onnx_models(),
        "steganography_modules": validate_steganography_modules(),
        "inference_interface": validate_inference_interface(),
        "ensemble_model": validate_ensemble(),
        "dataset_structure": validate_dataset_structure(),
    }

    # Run comprehensive test if basic validation passes
    if all(validation_results.values()):
        validation_results["comprehensive_test"] = run_comprehensive_test()

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    for test_name, passed in validation_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title():25s} - {status}")

    overall_success = all(validation_results.values())
    print(
        f"\nOVERALL RESULT: {'✓ VALIDATION PASSED' if overall_success else '✗ VALIDATION FAILED'}"
    )

    if overall_success:
        print("\n🎉 Your Starlight submission is ready!")
        print("Both LSB and EXIF models have been successfully verified and merged.")
    else:
        print("\n⚠️  Please fix the failed validation items before submission.")

    return overall_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
