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

## Research Model Section

### Model Variants Comparison
See [docs/grok/model_variants.md](model_variants.md) for detailed comparison of research models including triplet loss and adversarial variants.

### Research Metrics
- **Triplet Loss Model FPR**: Pending evaluation
- **Adversarial Hardened Model Latency**: Pending evaluation
- **Embedding Space Analysis**: Pending from Claude

## Recent Alerts

None – system operating nominally.

[Historical trends chart will be embedded here]