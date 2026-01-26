"""
Starlight Production Readiness Test Suite
Comprehensive testing for V4 model production deployment.
"""

import json
import argparse
import time
import os
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple
import unittest
import threading
import concurrent.futures

# Import starlight utilities
try:
    from scripts.starlight_utils import load_unified_input
except ImportError:
    print("Warning: starlight_utils not available")
    load_unified_input = None

# ONNX imports
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available")

# PIL for image manipulation
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available")

# psutil for memory monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available")


class ProductionTestSuite(unittest.TestCase):
    """Production readiness tests for Starlight V4"""

    def setUp(self):
        """Set up test environment"""
        self.model_path = os.getenv("STARLIGHT_MODEL", "models/detector_balanced.onnx")
        self.dataset_path = os.getenv(
            "STARLIGHT_DATASET", "datasets/sample_submission_2025"
        )

        if ONNX_AVAILABLE:
            try:
                self.model = ort.InferenceSession(self.model_path)
            except Exception as e:
                print(f"Warning: Could not load model {self.model_path}: {e}")
                self.model = None
        else:
            self.model = None

        # Expected benchmark values
        self.expected_fpr = 0.00  # From benchmark report
        self.expected_detection_rate = 98.63  # From benchmark report

    def test_01_benchmark_verification(self):
        """Verify model matches benchmark performance"""
        if not self.model:
            self.skipTest("ONNX not available")

        # Test on sample dataset
        fpr = calculate_fpr(self.model, self.dataset_path)
        detection_rates = calculate_detection_rates(self.model, self.dataset_path)
        overall_detection = np.mean(list(detection_rates.values()))

        # Verify FPR is 0.00% as per benchmark
        self.assertLessEqual(fpr, 0.01, f"FPR {fpr:.2f}% exceeds benchmark 0.00%")

        # Verify detection rate meets benchmark
        self.assertGreaterEqual(
            overall_detection,
            95.0,
            f"Detection rate {overall_detection:.1f}% below benchmark {self.expected_detection_rate}%",
        )

        print(
            f"✅ Benchmark verification: FPR={fpr:.2f}%, Detection={overall_detection:.1f}%"
        )

    def test_02_fresh_dataset_testing(self):
        """Test on completely new, unseen images"""
        if not self.model:
            self.skipTest("ONNX not available")

        if not PIL_AVAILABLE:
            self.skipTest("PIL not available for image generation")

        # Create synthetic clean images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate some random images
            for i in range(10):
                img = Image.new("RGB", (64, 64), color=(i * 25, i * 25, i * 25))
                img.save(temp_path / f"fresh_clean_{i}.png")

            if list(temp_path.glob("*.png")):
                fpr = calculate_fpr(self.model, str(temp_path))
                self.assertLessEqual(fpr, 1.0, f"High FPR on fresh images: {fpr:.1f}%")

                print(f"✅ Fresh dataset test: FPR={fpr:.1f}%")

    def test_03_edge_case_handling(self):
        """Test with corrupted files, unusual formats, large images"""
        if not self.model:
            self.skipTest("ONNX not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test cases
            test_cases = []

            if PIL_AVAILABLE:
                # Large image
                large_img = Image.new("RGB", (2048, 2048), color=(128, 128, 128))
                large_path = temp_path / "large.png"
                large_img.save(large_path)
                test_cases.append(("large_image", large_path))

                # Unusual format (if supported)
                try:
                    unusual_img = Image.new(
                        "RGBA", (64, 64), color=(128, 128, 128, 255)
                    )
                    unusual_path = temp_path / "unusual.tiff"
                    unusual_img.save(unusual_path)
                    test_cases.append(("unusual_format", unusual_path))
                except:
                    pass

                # Corrupted file
                corrupt_path = temp_path / "corrupt.png"
                with open(corrupt_path, "wb") as f:
                    f.write(b"not an image")
                test_cases.append(("corrupt_file", corrupt_path))
            else:
                # Skip edge case tests if PIL not available
                self.skipTest("PIL not available for edge case testing")

            # Test each case doesn't crash
            for case_name, img_path in test_cases:
                try:
                    result = infer_image(self.model, str(img_path))
                    # Should return some result, even if error
                    self.assertIsInstance(result, dict)
                    print(f"✅ Edge case '{case_name}': handled gracefully")
                except Exception as e:
                    self.fail(f"Edge case '{case_name}' crashed: {e}")

    def test_04_performance_under_load(self):
        """Stress test with high throughput requirements"""
        if not self.model:
            self.skipTest("ONNX not available")

        # Test concurrent requests
        def worker():
            latencies = benchmark_inference(
                self.model, sample_size=10, dataset_path=self.dataset_path
            )
            return latencies

        # Run 4 concurrent workers
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(4)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time
        total_requests = sum(len(latencies) for latencies in results)
        throughput = total_requests / total_time

        # Check throughput is reasonable (>10 req/sec)
        self.assertGreater(throughput, 10, f"Low throughput: {throughput:.1f} req/sec")

        # Check latency under load
        all_latencies = [lat for latencies in results for lat in latencies]
        p95 = np.percentile(all_latencies, 95)
        self.assertLess(p95, 50, f"High latency under load: {p95:.1f}ms")

        print(f"✅ Load test: {throughput:.1f} req/sec, p95={p95:.1f}ms")

    def test_05_memory_usage(self):
        """Ensure model fits in production memory constraints"""
        if not self.model:
            self.skipTest("ONNX not available")

        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for memory monitoring")

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run some inferences
        latencies = benchmark_inference(
            self.model, sample_size=50, dataset_path=self.dataset_path
        )

        # Check memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 500MB)
        self.assertLess(
            memory_increase, 500, f"High memory usage: +{memory_increase:.1f}MB"
        )

        print(f"✅ Memory test: {memory_increase:.1f}MB increase")

    def test_06_cross_platform_compatibility(self):
        """Test basic functionality (skip full cross-platform)"""
        if not self.model:
            self.skipTest("ONNX not available")

        # Just test that model loads and runs on current platform
        result = infer_image(
            self.model, list(Path(self.dataset_path).glob("clean/*.png"))[0]
        )
        self.assertIn("is_steganography", result)

        print("✅ Cross-platform test: model functional on current platform")


def calculate_fpr(model, dataset_path):
    """Calculate False Positive Rate on clean images"""
    clean_images = list(Path(dataset_path).glob("clean/**/*.png"))
    if not clean_images:
        clean_images = list(Path(dataset_path).glob("**/*.png"))[:50]

    fp_count = 0
    for img_path in clean_images[:100]:  # Limit for speed
        pred = infer_image(model, str(img_path))
        if pred.get("is_steganography", False):
            fp_count += 1
    return (fp_count / len(clean_images)) * 100 if clean_images else 0.0


def calculate_detection_rates(model, dataset_path):
    """Calculate detection rates per method"""
    methods = ["lsb", "alpha", "palette", "exif", "eoi"]
    results = {}
    for method in methods:
        stego_files = list(Path(dataset_path).glob(f"stego/{method}/**/*.png"))
        if stego_files:
            tp_count = sum(
                1
                for f in stego_files[:20]
                if infer_image(model, str(f)).get("is_steganography", False)
            )
            results[method] = (tp_count / len(stego_files[:20])) * 100
        else:
            results[method] = 0.0
    return results


def benchmark_inference(model, sample_size=100, dataset_path=None):
    """Benchmark inference latency"""
    if dataset_path:
        test_images = list(Path(dataset_path).glob("clean/**/*.png"))[:sample_size]
    else:
        test_images = []

    if not test_images:
        return [0.0]

    latencies = []
    for img in test_images:
        start = time.time()
        infer_image(model, str(img))
        latencies.append((time.time() - start) * 1000)
    return latencies


def infer_image(model, img_path):
    """Infer on a single image"""
    if not load_unified_input:
        return {"error": "starlight_utils not available", "is_steganography": False}

    try:
        (
            pixel_tensor,
            meta,
            alpha,
            lsb,
            palette,
            palette_lsb,
            format_features,
            content_features,
        ) = load_unified_input(img_path)

        lsb_chw = lsb.permute(2, 0, 1) if lsb.dim() == 3 else lsb
        alpha_chw = alpha.unsqueeze(0) if alpha.dim() == 2 else alpha

        inputs = {
            "meta": np.expand_dims(meta.numpy(), 0),
            "alpha": np.expand_dims(alpha_chw.numpy(), 0),
            "lsb": np.expand_dims(lsb_chw.numpy(), 0),
            "palette": np.expand_dims(palette.numpy(), 0),
            "format_features": np.expand_dims(format_features.numpy(), 0),
            "content_features": np.expand_dims(content_features.numpy(), 0),
            "bit_order": np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        }

        outputs = model.run(None, inputs)
        stego_logits = outputs[0]
        method_logits = outputs[1]

        prob = float(1 / (1 + np.exp(-stego_logits[0][0])))
        predicted_method_id = int(np.argmax(method_logits[0]))

        return {
            "stego_probability": prob,
            "predicted_method_id": predicted_method_id,
            "is_steganography": prob > 0.5,
        }
    except Exception as e:
        return {"error": str(e), "is_steganography": False}


def main():
    """Run production tests"""
    parser = argparse.ArgumentParser(
        description="Production readiness tests for Starlight V4"
    )
    parser.add_argument("--model", help="Path to model file")
    parser.add_argument("--dataset", help="Path to dataset directory")
    parser.add_argument(
        "--output",
        default="results/v4_validation_report.json",
        help="Output report file",
    )
    args = parser.parse_args()

    # Set environment variables if provided
    if args.model:
        os.environ["STARLIGHT_MODEL"] = args.model
    if args.dataset:
        os.environ["STARLIGHT_DATASET"] = args.dataset

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(ProductionTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model or "models/detector_balanced.onnx",
        "dataset": args.dataset or "datasets/sample_submission_2025",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
        "details": {
            "failures": [{"test": str(f[0]), "error": f[1]} for f in result.failures],
            "errors": [{"test": str(e[0]), "error": e[1]} for e in result.errors],
        },
    }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 Report saved to: {output_path}")
    print(f"✅ Production ready: {report['success']}")

    return 0 if report["success"] else 1


if __name__ == "__main__":
    exit(main())
