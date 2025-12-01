#!/usr/bin/env python3
"""
Comprehensive Production Validation Script for V4
Tests model performance on large, diverse dataset to verify production readiness

Requirements:
- 10,000+ diverse clean images
- All 5 steganography methods represented in stego dataset
- Breakdown by format, source, and edge cases

Author: Claude (Anthropic)
Date: November 30, 2025
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import time
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner import StarlightScanner


class ProductionValidator:
    """Comprehensive validation for production readiness"""
    
    def __init__(self, model_path, num_workers=8):
        self.scanner = StarlightScanner(model_path, num_workers=num_workers, quiet=True)
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'workers': num_workers
            },
            'clean_validation': {},
            'stego_validation': {},
            'performance_metrics': {},
            'edge_cases': {},
            'breakdown_analysis': {}
        }
    
    def validate_clean_dataset(self, clean_dir, min_images=10000):
        """
        Validate false positive rate on large clean dataset
        
        Args:
            clean_dir: Path to directory with clean images
            min_images: Minimum required images for production validation
        """
        print(f"\n{'='*80}")
        print(f"CLEAN DATASET VALIDATION")
        print(f"{'='*80}")
        print(f"Target: {min_images}+ images, FP rate < 0.1%")
        print(f"Directory: {clean_dir}\n")
        
        if not os.path.exists(clean_dir):
            print(f"❌ ERROR: Clean directory not found: {clean_dir}")
            return False
        
        # Scan directory
        start_time = time.time()
        results = self.scanner.scan_directory(clean_dir, quiet=False)
        scan_duration = time.time() - start_time
        
        # Analyze results
        total_images = len([r for r in results if 'error' not in r])
        false_positives = [r for r in results if r.get('is_stego', False)]
        errors = [r for r in results if 'error' in r]
        
        fp_rate = (len(false_positives) / total_images * 100) if total_images > 0 else 0
        
        # Breakdown by format
        format_breakdown = self._analyze_by_format(results, false_positives)
        
        # Breakdown by method (for FPs)
        method_breakdown = self._analyze_fp_methods(false_positives)
        
        # Performance metrics
        images_per_sec = total_images / scan_duration if scan_duration > 0 else 0
        
        # Store results
        self.results['clean_validation'] = {
            'total_images': total_images,
            'false_positives': len(false_positives),
            'fp_rate_percent': round(fp_rate, 4),
            'errors': len(errors),
            'scan_duration_seconds': round(scan_duration, 2),
            'images_per_second': round(images_per_sec, 2),
            'format_breakdown': format_breakdown,
            'fp_method_breakdown': method_breakdown,
            'meets_target': fp_rate < 0.1 and total_images >= min_images
        }
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"CLEAN VALIDATION RESULTS")
        print(f"{'='*80}")
        print(f"Total Images Scanned: {total_images:,}")
        print(f"False Positives: {len(false_positives)}")
        print(f"False Positive Rate: {fp_rate:.4f}%")
        print(f"Errors: {len(errors)}")
        print(f"Scan Duration: {scan_duration:.2f} seconds")
        print(f"Throughput: {images_per_sec:.2f} images/sec")
        
        # Target assessment
        if total_images < min_images:
            print(f"\n⚠️  WARNING: Dataset too small ({total_images} < {min_images})")
        
        if fp_rate < 0.05:
            print(f"\n✅ EXCELLENT: FP rate {fp_rate:.4f}% (target: <0.1%)")
        elif fp_rate < 0.1:
            print(f"\n✅ PASS: FP rate {fp_rate:.4f}% (target: <0.1%)")
        elif fp_rate < 0.5:
            print(f"\n⚠️  ACCEPTABLE: FP rate {fp_rate:.4f}% (acceptable for some use cases)")
        else:
            print(f"\n❌ FAIL: FP rate {fp_rate:.4f}% too high for production")
        
        # Format breakdown
        if format_breakdown:
            print(f"\nFormat Breakdown:")
            for fmt, stats in sorted(format_breakdown.items()):
                fp_count = stats['false_positives']
                total = stats['total']
                fp_pct = (fp_count / total * 100) if total > 0 else 0
                print(f"  {fmt:8s}: {fp_count:4d} FP / {total:5d} total ({fp_pct:.2f}%)")
        
        # Method breakdown for FPs
        if method_breakdown:
            print(f"\nFalse Positive Methods:")
            for method, count in sorted(method_breakdown.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(false_positives) * 100) if false_positives else 0
                print(f"  {method:12s}: {count:4d} ({pct:.1f}%)")
        
        # List some false positives for analysis
        if false_positives:
            print(f"\nSample False Positives (first 10):")
            for i, fp in enumerate(false_positives[:10]):
                filename = os.path.basename(fp['file_path'])
                method = fp.get('stego_type', 'unknown')
                conf = fp.get('confidence', 0) * 100
                print(f"  {i+1:2d}. {filename:40s} → {method:12s} ({conf:.1f}%)")
        
        return fp_rate < 0.1 and total_images >= min_images
    
    def validate_stego_dataset(self, stego_dir, min_per_method=100):
        """
        Validate detection rate on steganography dataset
        
        Args:
            stego_dir: Path to directory with stego images
            min_per_method: Minimum samples per method for validation
        """
        print(f"\n{'='*80}")
        print(f"STEGO DATASET VALIDATION")
        print(f"{'='*80}")
        print(f"Target: {min_per_method}+ samples per method, detection rate > 95%")
        print(f"Directory: {stego_dir}\n")
        
        if not os.path.exists(stego_dir):
            print(f"❌ ERROR: Stego directory not found: {stego_dir}")
            return False
        
        # Scan directory
        start_time = time.time()
        results = self.scanner.scan_directory(stego_dir, quiet=False)
        scan_duration = time.time() - start_time
        
        # Load ground truth from JSON sidecars
        ground_truth = self._load_ground_truth(stego_dir)
        
        # Analyze results
        total_images = len([r for r in results if 'error' not in r])
        detected = [r for r in results if r.get('is_stego', False)]
        missed = [r for r in results if not r.get('is_stego', False) and 'error' not in r]
        
        detection_rate = (len(detected) / total_images * 100) if total_images > 0 else 0
        
        # Method-specific analysis
        method_stats = self._analyze_by_method(results, ground_truth)
        
        # Store results
        self.results['stego_validation'] = {
            'total_images': total_images,
            'detected': len(detected),
            'missed': len(missed),
            'detection_rate_percent': round(detection_rate, 2),
            'scan_duration_seconds': round(scan_duration, 2),
            'method_breakdown': method_stats,
            'meets_target': detection_rate > 95.0
        }
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"STEGO VALIDATION RESULTS")
        print(f"{'='*80}")
        print(f"Total Stego Images: {total_images:,}")
        print(f"Detected: {len(detected)}")
        print(f"Missed: {len(missed)}")
        print(f"Detection Rate: {detection_rate:.2f}%")
        print(f"Scan Duration: {scan_duration:.2f} seconds")
        
        # Target assessment
        if detection_rate > 98:
            print(f"\n✅ EXCELLENT: Detection rate {detection_rate:.2f}% (target: >95%)")
        elif detection_rate > 95:
            print(f"\n✅ PASS: Detection rate {detection_rate:.2f}% (target: >95%)")
        else:
            print(f"\n❌ FAIL: Detection rate {detection_rate:.2f}% below target (>95%)")
        
        # Method breakdown
        if method_stats:
            print(f"\nMethod-Specific Performance:")
            for method, stats in sorted(method_stats.items()):
                total = stats['total']
                detected_count = stats['detected']
                rate = (detected_count / total * 100) if total > 0 else 0
                status = "✅" if rate > 95 else "⚠️" if rate > 90 else "❌"
                print(f"  {status} {method:12s}: {detected_count:4d}/{total:4d} ({rate:.1f}%)")
        
        # List some missed detections for analysis
        if missed:
            print(f"\nSample Missed Detections (first 10):")
            for i, miss in enumerate(missed[:10]):
                filename = os.path.basename(miss['file_path'])
                truth = ground_truth.get(filename, {})
                method = truth.get('method', 'unknown')
                print(f"  {i+1:2d}. {filename:40s} (expected: {method})")
        
        return detection_rate > 95.0
    
    def validate_performance(self, test_dir, target_images=1000):
        """
        Validate performance benchmarks (latency, throughput)
        
        Args:
            test_dir: Directory with test images
            target_images: Number of images for benchmark
        """
        print(f"\n{'='*80}")
        print(f"PERFORMANCE BENCHMARK")
        print(f"{'='*80}")
        print(f"Target: >20 images/sec, <50ms per image")
        print(f"Test images: {target_images}\n")
        
        if not os.path.exists(test_dir):
            print(f"❌ ERROR: Test directory not found: {test_dir}")
            return False
        
        # Get sample images
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        test_images = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    test_images.append(os.path.join(root, file))
                    if len(test_images) >= target_images:
                        break
            if len(test_images) >= target_images:
                break
        
        if len(test_images) < 100:
            print(f"⚠️  WARNING: Not enough test images ({len(test_images)} < 100)")
        
        test_images = test_images[:target_images]
        
        # Run benchmark
        print(f"Benchmarking on {len(test_images)} images...")
        start_time = time.time()
        
        # Use scanner directory scan for realistic performance
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy test images to temp dir
            for i, img in enumerate(test_images):
                shutil.copy(img, os.path.join(tmpdir, f"test_{i}{os.path.splitext(img)[1]}"))
            
            # Scan
            results = self.scanner.scan_directory(tmpdir, quiet=True)
        
        scan_duration = time.time() - start_time
        
        # Calculate metrics
        images_per_sec = len(test_images) / scan_duration if scan_duration > 0 else 0
        ms_per_image = (scan_duration / len(test_images) * 1000) if test_images else 0
        
        # Store results
        self.results['performance_metrics'] = {
            'test_images': len(test_images),
            'scan_duration_seconds': round(scan_duration, 2),
            'images_per_second': round(images_per_sec, 2),
            'ms_per_image': round(ms_per_image, 2),
            'meets_target_throughput': images_per_sec > 20,
            'meets_target_latency': ms_per_image < 50
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"PERFORMANCE RESULTS")
        print(f"{'='*80}")
        print(f"Test Images: {len(test_images):,}")
        print(f"Total Duration: {scan_duration:.2f} seconds")
        print(f"Throughput: {images_per_sec:.2f} images/sec")
        print(f"Latency: {ms_per_image:.2f} ms/image")
        
        # Target assessment
        if images_per_sec > 30:
            print(f"\n✅ EXCELLENT: Throughput {images_per_sec:.2f} img/s (target: >20)")
        elif images_per_sec > 20:
            print(f"\n✅ PASS: Throughput {images_per_sec:.2f} img/s (target: >20)")
        else:
            print(f"\n⚠️  BELOW TARGET: Throughput {images_per_sec:.2f} img/s (target: >20)")
        
        if ms_per_image < 30:
            print(f"✅ EXCELLENT: Latency {ms_per_image:.2f} ms (target: <50ms)")
        elif ms_per_image < 50:
            print(f"✅ PASS: Latency {ms_per_image:.2f} ms (target: <50ms)")
        else:
            print(f"⚠️  BELOW TARGET: Latency {ms_per_image:.2f} ms (target: <50ms)")
        
        return images_per_sec > 20 and ms_per_image < 50
    
    def generate_report(self, output_path):
        """Generate comprehensive validation report"""
        
        # Calculate overall status
        clean_pass = self.results.get('clean_validation', {}).get('meets_target', False)
        stego_pass = self.results.get('stego_validation', {}).get('meets_target', False)
        perf_pass = self.results.get('performance_metrics', {}).get('meets_target_throughput', False)
        
        overall_pass = clean_pass and stego_pass and perf_pass
        
        self.results['overall_assessment'] = {
            'production_ready': overall_pass,
            'clean_validation_pass': clean_pass,
            'stego_validation_pass': stego_pass,
            'performance_pass': perf_pass
        }
        
        # Add recommendations
        recommendations = []
        
        if not clean_pass:
            fp_rate = self.results.get('clean_validation', {}).get('fp_rate_percent', 0)
            if fp_rate > 0.5:
                recommendations.append("CRITICAL: False positive rate too high for production. Review special case logic.")
            elif fp_rate > 0.1:
                recommendations.append("WARNING: False positive rate above target. Consider additional validation.")
        
        if not stego_pass:
            det_rate = self.results.get('stego_validation', {}).get('detection_rate_percent', 0)
            recommendations.append(f"CRITICAL: Detection rate {det_rate:.1f}% below 95% target. Review method-specific performance.")
        
        if not perf_pass:
            throughput = self.results.get('performance_metrics', {}).get('images_per_second', 0)
            recommendations.append(f"WARNING: Performance {throughput:.1f} img/s below target. Optimize inference pipeline.")
        
        if overall_pass:
            recommendations.append("SUCCESS: All validation targets met. V4 is production-ready.")
        
        self.results['recommendations'] = recommendations
        
        # Write JSON report
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"FINAL ASSESSMENT")
        print(f"{'='*80}")
        print(f"Production Ready: {'✅ YES' if overall_pass else '❌ NO'}")
        print(f"  Clean Validation: {'✅' if clean_pass else '❌'}")
        print(f"  Stego Detection: {'✅' if stego_pass else '❌'}")
        print(f"  Performance: {'✅' if perf_pass else '❌'}")
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
        print(f"\nFull report saved to: {output_path}")
        print(f"{'='*80}\n")
        
        return overall_pass
    
    # Helper methods
    
    def _analyze_by_format(self, results, false_positives):
        """Analyze results by image format"""
        format_stats = defaultdict(lambda: {'total': 0, 'false_positives': 0})
        
        fp_paths = {r['file_path'] for r in false_positives}
        
        for result in results:
            if 'error' in result:
                continue
            
            path = result['file_path']
            ext = os.path.splitext(path)[1].lower().replace('.', '')
            
            format_stats[ext]['total'] += 1
            if path in fp_paths:
                format_stats[ext]['false_positives'] += 1
        
        return dict(format_stats)
    
    def _analyze_fp_methods(self, false_positives):
        """Analyze false positives by predicted method"""
        method_counts = Counter()
        for fp in false_positives:
            method = fp.get('stego_type', 'unknown')
            method_counts[method] += 1
        return dict(method_counts)
    
    def _load_ground_truth(self, stego_dir):
        """Load ground truth from JSON sidecars"""
        ground_truth = {}
        
        for root, _, files in os.walk(stego_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        
                        # Get corresponding image filename
                        img_filename = file[:-5]  # Remove .json
                        method = data.get('embedding', {}).get('technique', 'unknown')
                        ground_truth[img_filename] = {'method': method}
                    except:
                        continue
        
        return ground_truth
    
    def _analyze_by_method(self, results, ground_truth):
        """Analyze detection rate by steganography method"""
        method_stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'missed': []})
        
        for result in results:
            if 'error' in result:
                continue
            
            filename = os.path.basename(result['file_path'])
            
            # Get ground truth method
            truth = ground_truth.get(filename, {})
            method = truth.get('method', 'unknown')
            
            method_stats[method]['total'] += 1
            if result.get('is_stego', False):
                method_stats[method]['detected'] += 1
            else:
                method_stats[method]['missed'].append(filename)
        
        return dict(method_stats)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive V4 Production Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validation with 10K clean images
  python validate_production.py \\
    --clean datasets/production_validation/clean \\
    --stego datasets/production_validation/stego \\
    --model models/detector_balanced.pth \\
    --output validation_results.json

  # Quick validation on smaller dataset
  python validate_production.py \\
    --clean datasets/val/clean \\
    --stego datasets/val/stego \\
    --model models/detector_balanced.pth \\
    --min-clean 1000 \\
    --output quick_validation.json

Expected Directory Structure:
  datasets/production_validation/
  ├── clean/          # 10,000+ diverse clean images
  │   ├── photos/
  │   ├── screenshots/
  │   ├── generated/
  │   └── ...
  └── stego/          # 500+ stego images with JSON sidecars
      ├── alpha/
      ├── lsb/
      ├── palette/
      ├── exif/
      └── eoi/
        """
    )
    
    parser.add_argument('--clean', required=True, help='Path to clean images directory')
    parser.add_argument('--stego', required=True, help='Path to stego images directory')
    parser.add_argument('--model', default='models/detector_balanced.pth', 
                       help='Path to model file (PTH or ONNX)')
    parser.add_argument('--output', default='production_validation_results.json',
                       help='Output JSON report path')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--min-clean', type=int, default=10000,
                       help='Minimum clean images for validation')
    parser.add_argument('--min-per-method', type=int, default=100,
                       help='Minimum samples per stego method')
    parser.add_argument('--skip-performance', action='store_true',
                       help='Skip performance benchmarking')
    
    args = parser.parse_args()
    
    # Validate model exists
    if not os.path.exists(args.model):
        print(f"❌ ERROR: Model not found: {args.model}")
        return 1
    
    # Create validator
    print(f"{'='*80}")
    print(f"STARLIGHT V4 PRODUCTION VALIDATION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Clean Dataset: {args.clean}")
    print(f"Stego Dataset: {args.stego}")
    print(f"Output Report: {args.output}")
    
    validator = ProductionValidator(args.model, num_workers=args.workers)
    
    # Run validations
    clean_pass = validator.validate_clean_dataset(args.clean, min_images=args.min_clean)
    stego_pass = validator.validate_stego_dataset(args.stego, min_per_method=args.min_per_method)
    
    if not args.skip_performance:
        perf_pass = validator.validate_performance(args.clean, target_images=1000)
    
    # Generate report
    overall_pass = validator.generate_report(args.output)
    
    # Exit code
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
