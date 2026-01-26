import json
import argparse
from datetime import datetime
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any, List

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


class RegressionTestSuite:
    def __init__(self, baseline_path):
        with open(baseline_path) as f:
            self.baseline = json.load(f)

    def test_fpr_stability(self, model, test_data):
        """FPR should not exceed baseline + 0.02%"""
        current_fpr = calculate_fpr(model, test_data)
        baseline_fpr = self.baseline["fpr"]
        max_fpr = baseline_fpr + 0.02

        assert (
            current_fpr < max_fpr
        ), f"FPR regression: {current_fpr:.2f}% > {max_fpr:.2f}%"

        return {
            "test": "FPR Stability",
            "current": current_fpr,
            "baseline": baseline_fpr,
        }

    def test_detection_rates(self, model, test_data):
        """All methods should maintain >95% detection"""
        rates = calculate_detection_rates(model, test_data)

        for method, rate in rates.items():
            assert rate > 95, f"{method} detection dropped to {rate:.1f}%"

        return {"test": "Detection Rates", "rates": rates}

    def test_inference_speed(self, model, test_data):
        """Latency p95 should stay <5ms"""
        latencies = benchmark_inference(model, sample_size=100, dataset_path=test_data)
        p95 = np.percentile(latencies, 95)

        assert p95 < 5.0, f"Inference p95: {p95:.1f}ms"

        return {"test": "Inference Speed", "p95_ms": p95}

    def test_format_consistency(self, model, test_data):
        """Ensure no format-specific regressions"""
        formats = ["jpeg", "png", "gif", "webp", "bmp"]

        for fmt in formats:
            fpr = calculate_fpr_by_format(model, test_data, fmt)
            baseline = self.baseline["fpr_by_format"].get(fmt, 0.1)

            assert fpr < baseline + 0.02, f"FPR regression on {fmt}: {fpr:.2f}%"

        return {"test": "Format Consistency", "formats_tested": len(formats)}

    def test_negatives_resistance(self, model, negative_dataset):
        """Model should not trigger on negative examples"""
        if isinstance(negative_dataset, str):
            negative_images = list(Path(negative_dataset).glob("**/*.png"))[
                :100
            ]  # Sample
        else:
            negative_images = negative_dataset

        fp_count = sum(
            1
            for img in negative_images
            if infer_image(model, str(img)).get("is_steganography", False)
        )
        fp_rate = (fp_count / len(negative_images)) * 100

        assert fp_rate < 0.5, f"FP on negatives: {fp_rate:.1f}%"

        return {"test": "Negatives Resistance", "fp_rate": fp_rate}

    def run_all_tests(self, model, test_data, negatives_data=None):
        """Execute full regression suite"""
        tests = [
            ("FPR Stability", lambda: self.test_fpr_stability(model, test_data)),
            ("Detection Rates", lambda: self.test_detection_rates(model, test_data)),
            ("Inference Speed", lambda: self.test_inference_speed(model, test_data)),
            (
                "Format Consistency",
                lambda: self.test_format_consistency(model, test_data),
            ),
        ]

        if negatives_data:
            tests.append(
                (
                    "Negatives Resistance",
                    lambda: self.test_negatives_resistance(model, negatives_data),
                )
            )

        results = {"timestamp": datetime.now().isoformat(), "tests": {}}

        for test_name, test_fn in tests:
            try:
                result = test_fn()
                results["tests"][test_name] = {"status": "PASS", "details": result}
            except AssertionError as e:
                results["tests"][test_name] = {"status": "FAIL", "error": str(e)}

        return results


def calculate_fpr(model, test_data):
    """Calculate False Positive Rate on clean images"""
    clean_images = list(Path(test_data).glob("clean/**/*.png"))
    if not clean_images:
        clean_images = list(Path(test_data).glob("**/*.png"))[:100]

    fp_count = 0
    for img_path in clean_images:
        pred = infer_image(model, img_path)
        if pred.get("is_steganography", False):
            fp_count += 1
    return (fp_count / len(clean_images)) * 100 if clean_images else 0.0


def calculate_detection_rates(model, test_data):
    """Calculate detection rates per method"""
    methods = ["lsb", "alpha", "palette", "exif", "eoi"]
    results = {}
    for method in methods:
        stego_files = list(Path(test_data).glob(f"stego/{method}/**/*.png"))
        tp_count = sum(
            1
            for f in stego_files
            if infer_image(model, f).get("is_steganography", False)
        )
        results[method] = (tp_count / len(stego_files)) * 100 if stego_files else 0.0
    return results


def benchmark_inference(model, sample_size=1000, dataset_path=None):
    """Benchmark inference latency"""
    if dataset_path:
        test_images = list(Path(dataset_path).glob("clean/**/*.png"))[:sample_size]
    else:
        # Fallback, assume some images exist
        test_images = list(
            Path("datasets/sample_submission_2025/clean/").glob("*.png")
        )[:sample_size]

    latencies = []
    for img in test_images:
        start = time.time()
        infer_image(model, str(img))
        latencies.append((time.time() - start) * 1000)
    return latencies if latencies else [0.0]


def calculate_fpr_by_format(model, test_data, fmt):
    """Calculate FPR for specific format"""
    clean_images = list(Path(test_data).glob(f"clean/**/*.{fmt}"))
    fp_count = sum(
        1
        for img in clean_images
        if infer_image(model, img).get("is_steganography", False)
    )
    return (fp_count / len(clean_images)) * 100 if clean_images else 0.0


def infer_image(model, img_path):
    """Infer on a single image"""
    if not load_unified_input:
        return {"error": "starlight_utils not available"}

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
        return {"error": str(e)}


def load_model(model_path):
    """Load ONNX model"""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX runtime required")
    return ort.InferenceSession(model_path)


def load_dataset(dataset_path):
    """Load dataset (just return path for now)"""
    return dataset_path


def main():
    parser = argparse.ArgumentParser(description="Regression tests for Starlight")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", default="tests/regression_baselines.json")
    parser.add_argument("--negatives", help="Negative examples dataset")
    args = parser.parse_args()

    suite = RegressionTestSuite(args.baseline)
    results = suite.run_all_tests(
        load_model(args.model),
        load_dataset(args.dataset),
        load_dataset(args.negatives) if args.negatives else None,
    )

    # Print results
    print("\n=== REGRESSION TEST RESULTS ===")
    for test_name, result in results["tests"].items():
        status = result["status"]
        print(f"{test_name:30} {status}")

    # Save results
    with open(
        f"results/regression_reports/{datetime.now().isoformat()}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)

    # Exit with appropriate code
    failed = [t for t in results["tests"].values() if t["status"] == "FAIL"]
    if failed:
        print(f"\n❌ {len(failed)} test(s) failed")
        exit(1)
    else:
        print("\n✅ All regression tests passed")
        exit(0)


if __name__ == "__main__":
    main()
