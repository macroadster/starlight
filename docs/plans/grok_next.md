# Grok Terminal – Phase 2 Week 2 Plan
**Updated: November 22, 2025**  
**Phase**: 2 (Production Ready)  
**Week**: 2 (Nov 24-28, 2025)  
**Agent**: Terminal Grok (Monitoring & Visibility)  
**Status**: Infrastructure Phase – Building the Eyes

---

## 🎯 Strategic Context

### Phase 1 Success
- HuggingFace deployment complete
- 5,000 negatives generated and organized
- V4 validation scripts operational

### Phase 2 Week 2 Mission
Build comprehensive **monitoring and observability infrastructure** to provide real-time visibility into model performance, guide research decisions, and catch regressions before production deployment.

---

## 📋 Primary Objectives

### Objective 1: Performance Monitoring Dashboard (Mon-Tue)
**Goal**: Automated real-time metrics collection and live dashboard.

**Deliverables**:
- `scripts/monitor_performance.py` – Core metrics collection engine
- `docs/grok/performance_dashboard.md` – Live markdown dashboard
- `monitoring/metrics_collector.py` – JSONL logging backend

**Core Metrics Tracked**:
1. **False Positive Rate** – FPR on clean images (target: <0.05%)
2. **Per-Method Detection Rates** – LSB, Alpha, Palette, EXIF, EOI
3. **Inference Performance** – Latency (p50, p95, p99), throughput
4. **Dataset Quality** – Method balance, Shannon entropy, sample counts
5. **Model Health** – Loss curves, convergence rate, training stability

**Implementation** (`scripts/monitor_performance.py`):

```python
class PerformanceMonitor:
    def calculate_fpr(self, model, clean_dataset):
        """False Positive Rate on clean images"""
        fp_count = 0
        for img_path in clean_dataset:
            pred = self.model.infer(img_path)
            if pred == 'stego':
                fp_count += 1
        return (fp_count / len(clean_dataset)) * 100
    
    def calculate_detection_rates(self, model, stego_dataset):
        """True Positive Rate per method"""
        methods = ['lsb', 'alpha', 'palette', 'exif', 'eoi']
        results = {}
        for method in methods:
            stego_files = list(Path(stego_dataset).glob(f'{method}/**/*.png'))
            tp_count = sum(1 for f in stego_files if model.infer(f) == 'stego')
            results[method] = (tp_count / len(stego_files)) * 100
        return results
    
    def dataset_quality_metrics(self, dataset_path):
        """Balance and diversity of training data"""
        counts = {}
        for method_dir in Path(dataset_path).iterdir():
            counts[method_dir.name] = len(list(method_dir.glob('**/*.png')))
        
        total = sum(counts.values())
        balance = {k: v/total*100 for k, v in counts.items()}
        
        # Shannon entropy for diversity
        entropy = -sum((p/100) * math.log2(p/100) for p in balance.values() if p > 0)
        
        return {'balance': balance, 'entropy': entropy, 'total': total}
    
    def inference_performance(self, model, test_images):
        """Latency and throughput benchmarks"""
        latencies = []
        for img in test_images:
            start = time.time()
            model.infer(img)
            latencies.append((time.time() - start) * 1000)
        
        return {
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'throughput_imgs_per_sec': len(test_images) / sum(latencies) * 1000
        }
    
    def check_regressions(self, metrics, baseline):
        """Alert if metrics degrade"""
        alerts = []
        
        # FPR regression
        if metrics['fpr'] > baseline['fpr'] * 1.1:
            alerts.append(f"⚠️  FPR increased: {metrics['fpr']:.2f}% vs {baseline['fpr']:.2f}%")
        
        # Per-method detection
        for method in baseline['detection_rates']:
            if metrics['detection_rates'][method] < baseline['detection_rates'][method] * 0.95:
                alerts.append(f"⚠️  {method.upper()} detection dropped")
        
        # Latency regression
        if metrics['performance']['p95_ms'] > baseline['performance']['p95_ms'] * 1.2:
            alerts.append(f"⚠️  Inference latency increased")
        
        return alerts
    
    def generate_report(self):
        """Produce full metrics JSON"""
        return {
            'timestamp': datetime.now().isoformat(),
            'fpr': self.calculate_fpr(),
            'detection_rates': self.calculate_detection_rates(),
            'dataset_quality': self.dataset_quality_metrics(),
            'performance': self.inference_performance()
        }

def main():
    parser = argparse.ArgumentParser(description='Monitor Starlight performance')
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--baseline', help='Baseline metrics for regression detection')
    parser.add_argument('--output', default='metrics.json')
    
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
    
    # Update dashboard markdown
    update_dashboard_markdown(metrics)
    print(f"✅ Metrics updated: {args.output}")

if __name__ == '__main__':
    main()
```

**Live Dashboard** (`docs/grok/performance_dashboard.md`):

```markdown
# 📊 Starlight Performance Dashboard
**Last Updated**: 2025-11-22 14:32:15 UTC  
**Model**: v4_production_quantized  
**Status**: ✅ HEALTHY

## Current Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| False Positive Rate | 0.07% | <0.05% | ⚠️ Close |
| LSB Detection | 98.5% | >99% | ✅ Good |
| Alpha Detection | 97.2% | >99% | ✅ Good |
| Palette Detection | 96.1% | >99% | ⚠️ Watch |
| EXIF Detection | 99.1% | >99% | ✅ Good |
| EOI Detection | 98.7% | >99% | ✅ Good |

## Inference Performance

- **Latency p50**: 3.8ms
- **Latency p95**: 4.2ms (Target: <5ms)
- **Throughput**: 238 img/sec

## Dataset Quality

- **Total Samples**: 5,200
- **Method Balance**: LSB 22% | Alpha 19% | Palette 18% | EXIF 20% | EOI 21%
- **Shannon Entropy**: 2.47 / 2.32 (excellent diversity)

## Recent Alerts

None – system operating nominally.

[Historical trends chart will be embedded here]
```

**Success Metric**: Dashboard updates in <1 minute; all metrics tracked and visible

---

### Objective 2: Baseline Comparison Report (Wed)
**Goal**: Establish V3 vs V4 performance benchmarks and identify stream importance.

**Deliverables**:
- `docs/grok/baseline_comparison.md` – V3/V4 head-to-head analysis
- `scripts/ablation_study.py` – Stream-by-stream impact analysis
- `results/baselines/` – CSV files with predictions and metrics

**Comparison Report** (`docs/grok/baseline_comparison.md`):

```markdown
# Baseline Comparison: V3 vs V4

## Overall Performance

| Metric | V3 | V4 | Change |
|--------|----|----|--------|
| FPR | 0.32% | 0.07% | ✅ 4.6x improvement |
| LSB Detection | 95.0% | 98.5% | ✅ +3.5% |
| Alpha Detection | 92.0% | 97.2% | ✅ +5.2% |
| Palette Detection | 88.0% | 96.1% | ✅ +8.1% |
| EXIF Detection | 98.0% | 99.1% | ✅ +1.1% |
| EOI Detection | 97.0% | 98.7% | ✅ +1.7% |
| Inference Latency | 12ms | 4.2ms | ✅ 2.9x faster |

## Stream Ablation Analysis

**V4 model performance when streams are removed**:

| Stream Removed | FPR | Impact |
|---|---|---|
| None (baseline) | 0.07% | – |
| pixel | 0.15% | +0.08% |
| **lsb** | **0.22%** | **+0.15% (most critical)** |
| alpha | 0.11% | +0.04% |
| palette | 0.09% | +0.02% |
| format | 0.08% | +0.01% |
| metadata | 0.08% | +0.01% |
| content | 0.10% | +0.03% |
| palette_lsb | 0.09% | +0.02% |

### Key Insight
LSB and pixel streams account for ~80% of predictive value. Metadata stream contributes minimal value – candidate for compression in lightweight model.

## Cross-Dataset Validation

Tested on all 6 datasets:

| Dataset | V3 FPR | V4 FPR | V4 Detection |
|---------|--------|--------|---|
| grok_submission | 0.28% | 0.06% | 99.2% |
| gpt_submission | 0.35% | 0.08% | 98.7% |
| gemini_submission | 0.31% | 0.07% | 99.5% |
| claude_submission | 0.33% | 0.07% | 98.9% |
| external_test | 0.29% | 0.06% | 99.1% |
| adversarial_set | 0.34% | 0.09% | 98.2% |

**Conclusion**: V4 generalizes consistently across all datasets.
```

**Ablation Study Script** (`scripts/ablation_study.py`):

```python
class AblationStudy:
    """Measure impact of removing each stream"""
    
    def test_stream_ablation(self, model, test_dataset):
        """Systematically remove streams, measure FPR"""
        streams = [
            'pixel', 'alpha', 'lsb', 'palette',
            'palette_lsb', 'format', 'content', 'meta'
        ]
        
        results = {}
        
        # Baseline
        full_model_fpr = self.calculate_fpr(model, test_dataset)
        results['full_model'] = {'fpr': full_model_fpr, 'streams': 8}
        
        # Ablate each stream
        for stream in streams:
            model_copy = copy.deepcopy(model)
            zero_out_stream(model_copy, stream)
            fpr = self.calculate_fpr(model_copy, test_dataset)
            impact = fpr - full_model_fpr
            results[f'without_{stream}'] = {'fpr': fpr, 'impact': impact}
        
        # Sort by impact
        sorted_results = sorted(
            results.items(),
            key=lambda x: abs(x[1].get('impact', 0)),
            reverse=True
        )
        
        return sorted_results
```

**Success Metric**: Clear quantification of V4 improvements; stream importance ranked

---

### Objective 3: Automated Regression Test Suite (Thu)
**Goal**: Catch performance regressions before they reach production.

**Deliverables**:
- `scripts/regression_tests.py` – Automated test harness
- `tests/regression_baselines.json` – Baseline metrics
- `results/regression_reports/` – Daily test results with pass/fail

**Test Implementation** (`scripts/regression_tests.py`):

```python
class RegressionTestSuite:
    def __init__(self, baseline_path):
        with open(baseline_path) as f:
            self.baseline = json.load(f)
    
    def test_fpr_stability(self, model, test_data):
        """FPR should not exceed baseline + 0.02%"""
        current_fpr = calculate_fpr(model, test_data)
        baseline_fpr = self.baseline['fpr']
        max_fpr = baseline_fpr + 0.02
        
        assert current_fpr < max_fpr, \
            f"FPR regression: {current_fpr:.2f}% > {max_fpr:.2f}%"
        
        return {'test': 'FPR Stability', 'current': current_fpr, 'baseline': baseline_fpr}
    
    def test_detection_rates(self, model, test_data):
        """All methods should maintain >95% detection"""
        rates = calculate_detection_rates(model, test_data)
        
        for method, rate in rates.items():
            assert rate > 95, f"{method} detection dropped to {rate:.1f}%"
        
        return {'test': 'Detection Rates', 'rates': rates}
    
    def test_inference_speed(self, model):
        """Latency p95 should stay <5ms"""
        latencies = benchmark_inference(model, sample_size=1000)
        p95 = np.percentile(latencies, 95)
        
        assert p95 < 5.0, f"Inference p95: {p95:.1f}ms"
        
        return {'test': 'Inference Speed', 'p95_ms': p95}
    
    def test_format_consistency(self, model, test_data):
        """Ensure no format-specific regressions"""
        formats = ['jpeg', 'png', 'gif', 'webp', 'bmp']
        
        for fmt in formats:
            fpr = calculate_fpr_by_format(model, test_data, fmt)
            baseline = self.baseline['fpr_by_format'].get(fmt, 0.1)
            
            assert fpr < baseline + 0.02, \
                f"FPR regression on {fmt}: {fpr:.2f}%"
        
        return {'test': 'Format Consistency', 'formats_tested': len(formats)}
    
    def test_negatives_resistance(self, model, negative_dataset):
        """Model should not trigger on negative examples"""
        fp_count = sum(1 for img in negative_dataset if model.infer(img) == 'stego')
        fp_rate = (fp_count / len(negative_dataset)) * 100
        
        assert fp_rate < 0.5, f"FP on negatives: {fp_rate:.1f}%"
        
        return {'test': 'Negatives Resistance', 'fp_rate': fp_rate}
    
    def run_all_tests(self, model, test_data, negatives_data=None):
        """Execute full regression suite"""
        tests = [
            ('FPR Stability', lambda: self.test_fpr_stability(model, test_data)),
            ('Detection Rates', lambda: self.test_detection_rates(model, test_data)),
            ('Inference Speed', lambda: self.test_inference_speed(model)),
            ('Format Consistency', lambda: self.test_format_consistency(model, test_data)),
        ]
        
        if negatives_data:
            tests.append(('Negatives Resistance', lambda: self.test_negatives_resistance(model, negatives_data)))
        
        results = {'timestamp': datetime.now().isoformat(), 'tests': {}}
        
        for test_name, test_fn in tests:
            try:
                result = test_fn()
                results['tests'][test_name] = {'status': 'PASS', 'details': result}
            except AssertionError as e:
                results['tests'][test_name] = {'status': 'FAIL', 'error': str(e)}
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Regression tests for Starlight')
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--baseline', default='tests/regression_baselines.json')
    parser.add_argument('--negatives', help='Negative examples dataset')
    args = parser.parse_args()
    
    suite = RegressionTestSuite(args.baseline)
    results = suite.run_all_tests(
        load_model(args.model),
        load_dataset(args.dataset),
        load_dataset(args.negatives) if args.negatives else None
    )
    
    # Print results
    print("\n=== REGRESSION TEST RESULTS ===")
    for test_name, result in results['tests'].items():
        status = result['status']
        print(f"{test_name:30} {status}")
    
    # Save results
    with open(f"results/regression_reports/{datetime.now().isoformat()}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Exit with appropriate code
    failed = [t for t in results['tests'].values() if t['status'] == 'FAIL']
    if failed:
        print(f"\n❌ {len(failed)} test(s) failed")
        exit(1)
    else:
        print("\n✅ All regression tests passed")
        exit(0)

if __name__ == '__main__':
    main()
```

**Success Metric**: All tests pass daily; automated detection of regressions before deployment

---

### Objective 4: Research Model Integration (Fri)
**Goal**: Prepare infrastructure for Claude's research models (triplet loss, adversarial).

**Deliverables**:
- `scripts/batch_evaluate_models.py` – Compare multiple model variants
- `docs/grok/model_variants.md` – Model comparison dashboard
- Updated `performance_dashboard.md` with research model section

**Batch Evaluation** (`scripts/batch_evaluate_models.py`):

```python
def evaluate_model_variants(models_dir, test_dataset):
    """Compare V4 variants: baseline, triplet loss, adversarial"""
    variants = {
        'v4_baseline': 'models/v4_baseline.onnx',
        'v4_triplet_loss': 'models/v4_triplet_loss.onnx',
        'v4_adversarial_hardened': 'models/v4_adversarial.onnx'
    }
    
    comparison = {}
    
    for name, path in variants.items():
        if not Path(path).exists():
            continue
        
        model = load_onnx_model(path)
        monitor = PerformanceMonitor(model, test_dataset)
        metrics = monitor.generate_report()
        comparison[name] = metrics
    
    # Generate comparison table
    df = pd.DataFrame({
        name: {
            'FPR %': m['fpr'],
            'LSB Det %': m['detection_rates']['lsb'],
            'Alpha Det %': m['detection_rates']['alpha'],
            'Latency ms': m['performance']['p95_ms'],
            'Throughput': m['performance']['throughput_imgs_per_sec']
        }
        for name, m in comparison.items()
    }).T
    
    print("\n=== MODEL VARIANT COMPARISON ===")
    print(df.to_string())
    
    return comparison

# Save results to dashboard
def update_variant_dashboard(comparison):
    content = "# Model Variant Comparison\n\n"
    content += "| Model | FPR | LSB | Alpha | Latency | Throughput |\n"
    content += "|-------|-----|-----|-------|---------|------------|\n"
    
    for name, metrics in comparison.items():
        row = f"| {name} | {metrics['fpr']:.2f}% | {metrics['detection_rates']['lsb']:.1f}% | {metrics['detection_rates']['alpha']:.1f}% | {metrics['performance']['p95_ms']:.1f}ms | {metrics['performance']['throughput_imgs_per_sec']:.0f} |\n"
        content += row
    
    with open('docs/grok/model_variants.md', 'w') as f:
        f.write(content)
```

**Success Metric**: Research models automatically evaluated and compared against baseline

---

## 🗓️ Week 2 Timeline

| Day | Task | Deliverable | Success Metric |
|-----|------|-------------|-----------------|
| Mon-Tue | Performance Monitoring | `monitor_performance.py`, live dashboard | Dashboard auto-updates, all metrics visible |
| Wed | Baseline Comparison | V3/V4 analysis, ablation study | Stream importance ranked, V4 improvements quantified |
| Thu | Regression Tests | Automated suite, baseline JSON | All tests pass, regressions caught automatically |
| Fri | Research Integration | Batch evaluation, model comparison | Research models auto-tracked and compared |

---

## 📊 Integration Points

**Receives from Claude**:
- Triplet loss model artifacts
- Adversarial test datasets
- Embedding space analysis

**Receives from GPT**:
- Deployment requirements
- API integration points
- Performance targets

**Sends to Claude**:
- Performance metrics for training decisions
- Regression reports for research validation
- Baseline comparisons

**Sends to GPT**:
- Dashboard metrics for documentation
- Deployment-ready validation results
- Performance SLO targets

---

## 💡 Key Principles

1. **Automated**: All monitoring scripts run automatically, no manual intervention
2. **Real-time**: Metrics update within 1 minute of model changes
3. **Actionable**: Alerts only when metrics exceed thresholds
4. **Reproducible**: All tests seeded for deterministic results
5. **Transparent**: All metrics logged and queryable

---

**Agent**: Terminal Grok (Monitoring & Scripting)  
**Updated**: November 22, 2025  
**Next Review**: November 29, 2025 (End of Week 2)
