import json
import argparse
import time
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from scanner import StarlightScanner
from monitoring.metrics_collector import MetricsCollector
from monitoring.dashboard import DashboardGenerator

class PerformanceMonitor:
    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.scanner = StarlightScanner(model_path, num_workers=1, quiet=True)  # quiet for monitoring
        self.collector = MetricsCollector()

    def calculate_fpr(self, clean_images):
        """False Positive Rate on clean images"""
        fp_count = 0
        for img_path in clean_images:
            result = self.scanner.scan_file(img_path)
            if result.get("is_stego", False):
                fp_count += 1
        return (fp_count / len(clean_images)) * 100 if clean_images else 0

    def calculate_detection_rates(self, stego_images_by_method):
        """True Positive Rate per method"""
        results = {}
        for method, images in stego_images_by_method.items():
            tp_count = sum(1 for img in images if self.scanner.scan_file(img).get("is_stego", False))
            results[method] = (tp_count / len(images)) * 100 if images else 0
        return results

    def dataset_quality_metrics(self):
        """Balance and diversity of training data"""
        stego_dir = self.dataset_path / 'stego'
        if not stego_dir.exists():
            return {'balance': {}, 'entropy': 0, 'total': 0}

        counts = {}
        for method_dir in stego_dir.iterdir():
            if method_dir.is_dir():
                counts[method_dir.name] = len(list(method_dir.glob('**/*.png')))

        total = sum(counts.values())
        balance = {k: v/total*100 for k, v in counts.items()} if total > 0 else {}

        # Shannon entropy for diversity
        entropy = -sum((p/100) * math.log2(p/100) for p in balance.values() if p > 0)

        return {'balance': balance, 'entropy': entropy, 'total': total}

    def inference_performance(self, test_images):
        """Latency and throughput benchmarks"""
        if not test_images:
            return {'p50_ms': 0, 'p95_ms': 0, 'p99_ms': 0, 'throughput_imgs_per_sec': 0}

        latencies = []
        for img in test_images[:100]:  # limit to 100 for speed
            start = time.time()
            self.scanner.scan_file(img)
            latencies.append((time.time() - start) * 1000)

        return {
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'throughput_imgs_per_sec': len(latencies) / sum(latencies) * 1000
        }

    def check_regressions(self, metrics, baseline):
        """Alert if metrics degrade"""
        alerts = []

        # FPR regression
        if metrics['fpr'] > baseline['fpr'] * 1.1:
            alerts.append(f"⚠️  FPR increased: {metrics['fpr']:.2f}% vs {baseline['fpr']:.2f}%")

        # Per-method detection
        for method in baseline.get('detection_rates', {}):
            if method in metrics['detection_rates'] and metrics['detection_rates'][method] < baseline['detection_rates'][method] * 0.95:
                alerts.append(f"⚠️  {method.upper()} detection dropped")

        # Latency regression
        if metrics['performance']['p95_ms'] > baseline['performance']['p95_ms'] * 1.2:
            alerts.append(f"⚠️  Inference latency increased")

        return alerts

    def generate_report(self):
        """Produce full metrics JSON"""
        # Get clean images
        clean_dir = self.dataset_path / 'clean'
        clean_images = list(clean_dir.glob('**/*.png')) if clean_dir.exists() else []

        # Get stego images by method
        stego_dir = self.dataset_path / 'stego'
        stego_images_by_method = {}
        if stego_dir.exists():
            for method_dir in stego_dir.iterdir():
                if method_dir.is_dir():
                    stego_images_by_method[method_dir.name] = list(method_dir.glob('**/*.png'))

        # Sample test images for performance
        test_images = clean_images[:50] if len(clean_images) > 50 else clean_images

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'fpr': self.calculate_fpr(clean_images),
            'detection_rates': self.calculate_detection_rates(stego_images_by_method),
            'dataset_quality': self.dataset_quality_metrics(),
            'performance': self.inference_performance(test_images)
        }

        # Log to collector
        self.collector.log_metrics(metrics)

        return metrics



def main():
    parser = argparse.ArgumentParser(description='Monitor Starlight performance')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--baseline', help='Path to baseline metrics JSON')
    parser.add_argument('--output', default='metrics.json', help='Output metrics file')
    parser.add_argument('--update-dashboard', action='store_true', help='Update the dashboard markdown')

    args = parser.parse_args()

    monitor = PerformanceMonitor(args.model, args.dataset)
    metrics = monitor.generate_report()

    # Save metrics
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Check regressions
    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        alerts = monitor.check_regressions(metrics, baseline)
        for alert in alerts:
            print(alert)

    if args.update_dashboard:
        dashboard_gen = DashboardGenerator()
        dashboard_gen.update_dashboard(metrics)

    print(f"✅ Metrics updated: {args.output}")

if __name__ == '__main__':
    main()