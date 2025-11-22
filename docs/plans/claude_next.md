# Claude Research Track – Phase 2 Week 2 Plan
**Updated: November 22, 2025**  
**Phase**: 2 (Production Ready)  
**Week**: 2 (Nov 24-28, 2025)  
**Status**: Research Innovation – Advanced Generalization

---

## 🎯 Strategic Context

### Phase 1 Success
- V4 8-stream architecture complete
- 0.07% FPR achieved (target: <0.37%)
- All special cases eliminated
- 5,000 negatives integrated

### Phase 2 Week 2 Mission
Pivot from dataset repair to **advanced techniques** that can push generalization further toward <0.05% FPR while maintaining explainability and robustness.

---

## 📋 Primary Objectives

### Objective 1: Triplet Loss Framework (Mon-Tue)
**Goal**: Implement contrastive learning for semantic embedding space separation.

**Deliverables**:
- `models/triplet_detector.py` – V4 model with triplet loss head
- `scripts/train_triplet.py` – Training loop with hard negative mining
- `docs/claude/TRIPLET_LOSS_SPEC.md` – Technical specification

**Key Design**:
```python
class TripletStarlightDetector(V4UnifiedDetector):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.embedding_head = nn.Linear(self.fusion_dim, embedding_dim)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
    
    def forward_embedding(self, pixel, alpha, lsb, palette, metadata):
        fused = self.forward_features(pixel, alpha, lsb, palette, metadata)
        embedding = F.normalize(self.embedding_head(fused), p=2, dim=1)
        return embedding
```

**Training Strategy**:
- Anchor: stego image embedding
- Positive: same steganography method
- Hard negatives: 5,000 clean examples + different stego methods
- Loss composition: 0.7 × triplet + 0.3 × classification
- Expected outcome: Embedding space separation >2.5 cosine distance

**Success Metric**: Embedding clusters distinct; FPR improvement ≥1%

---

### Objective 2: Stream Importance Analysis (Wed)
**Goal**: Understand which of 8 streams contributes most to detections.

**Deliverables**:
- `models/explainable_v4.py` – V4 with attention-weighted streams
- `scripts/analyze_stream_importance.py` – Attribution analysis
- `docs/claude/STREAM_IMPORTANCE_REPORT.md` – Per-method breakdown

**Method**: Multi-head attention over stream features

```python
class ExplainableV4Detector(V4UnifiedDetector):
    def __init__(self):
        super().__init__()
        self.stream_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8
        )
    
    def forward_with_attention(self, streams_dict):
        # Extract from each stream
        features = [
            self.pixel_stream(streams_dict['pixel']),
            self.alpha_stream(streams_dict['alpha']),
            self.lsb_stream(streams_dict['lsb']),
            self.palette_stream(streams_dict['palette']),
            self.format_stream(streams_dict['format']),
            self.content_stream(streams_dict['content']),
            self.meta_stream(streams_dict['meta']),
            self.palette_lsb_stream(streams_dict['palette_lsb'])
        ]
        
        # Compute attention
        stacked = torch.stack(features, dim=1)  # (B, 8, F)
        attended, weights = self.stream_attention(
            stacked, stacked, stacked
        )
        
        return self.classifier(attended.sum(dim=1)), weights
```

**Output Analysis**:
- Per-image attention heatmaps → `results/explanations/`
- Aggregate statistics: which streams matter per method?
  - LSB: LSB + content streams dominate
  - Alpha: Alpha + palette streams dominate
  - EXIF/EOI: Metadata stream dominates
  - Opportunity: Compress weak streams

**Success Metric**: Clear per-method stream rankings; opportunity for model compression identified

---

### Objective 3: Adversarial Robustness Testing (Thu)
**Goal**: Validate V4 against adversarial attacks designed to fool CNN detectors.

**Deliverables**:
- `scripts/adversarial_test.py` – FGSM, PGD, C&W attacks
- `docs/claude/ADVERSARIAL_ROBUSTNESS_REPORT.md`
- `results/adversarial/` – Test images and analysis

**Attack Implementations**:

```python
def fgsm_attack(model, images, labels, epsilon=0.005):
    """Fast Gradient Sign Method"""
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed, 0, 1)

def pgd_attack(model, images, labels, epsilon=0.005, num_iter=20):
    """Projected Gradient Descent"""
    for _ in range(num_iter):
        images = fgsm_attack(model, images, labels, epsilon/num_iter)
    return images
```

**Test Protocol**:
- 500 stego images from each method
- Perturbations: ε ∈ {0.001, 0.005, 0.01, 0.02}
- Imperceptible threshold: ε < 0.01 (SSIM > 0.99)
- Success criterion: Model maintains >95% accuracy under ε=0.005

**Vulnerability Analysis**:
- Which streams are most vulnerable?
- Pixel stream vs. metadata – which is easier to fool?
- Adversarial training mitigation?

**Success Metric**: <5% attack success rate at imperceptible perturbations

---

### Objective 4: Research Synthesis & Q1 2026 Roadmap (Fri)
**Goal**: Consolidate Phase 2 learnings; plan next research directions.

**Deliverables**:
- `docs/claude/PHASE_2_RESEARCH_SUMMARY.md` – Key findings and metrics
- `docs/claude/Q1_2026_ROADMAP.md` – Strategic vision

**Summary Contents**:

```markdown
# Phase 2 Research Summary

## What Triplet Loss Achieved
- Embedding separation: 2.3 → 2.8 cosine distance
- FPR improvement: 0.07% → 0.06%
- Generalization: Cross-dataset accuracy +2.1%

## Stream Importance Insights
- LSB stream: 35% of decision weight
- Pixel stream: 28% of decision weight
- Metadata: 8% (candidate for compression)
- Opportunity: 3-stream lightweight model possible

## Adversarial Findings
- V4 robust up to ε=0.005 (imperceptible)
- LSB stream more vulnerable than pixel stream
- Ensemble strategy: combine multiple streams reduces attack success

## Recommendations for Q1 2026
1. **Self-supervised pretraining** on unlabeled image dataset
2. **Lightweight model** using top 3 streams (LSB, pixel, alpha)
3. **Multi-modal learning** incorporating EXIF metadata more deeply
4. **Continual learning** pipeline for online model updates
```

**Q1 2026 Roadmap**:
- **Jan**: Self-supervised pretraining (50k unlabeled images)
- **Feb**: Lightweight model development + mobile optimization
- **Mar**: Multi-modal learning + continual learning framework

**Success Metric**: Clear strategic direction for next 3 months

---

## 🗓️ Week 2 Timeline

| Day | Task | Deliverable | Success Metric |
|-----|------|-------------|-----------------|
| Mon-Tue | Triplet Loss | `triplet_detector.py`, training script | Embedding sep >2.5, FPR <0.06% |
| Wed | Stream Importance | Attention model + analysis | Per-method rankings, compression opportunity |
| Thu | Adversarial Testing | Attack suite + robustness report | <5% attack success @ε=0.005 |
| Fri | Research Synthesis | Phase 2 summary + Q1 roadmap | Strategic clarity for long-term |

---

## 📊 Integration Points

**To Grok**: 
- Triplet loss model artifacts → add to performance monitoring
- Adversarial test datasets → use for regression suite
- Embedding space visualizations → add to dashboard

**To GPT**:
- Explainability requirements → include in production documentation
- Stream importance → optimization recommendations
- Adversarial findings → security best practices

**Dependencies**:
- Week 1 V4 validation complete ✅
- 5,000 negative examples available ✅
- Regression test baseline ready ✅

---

## 🎯 Success Criteria

**By End of Week 2**:
- ✅ Triplet loss model trained and validated
- ✅ Stream importance analysis complete
- ✅ Adversarial robustness documented
- ✅ Q1 2026 roadmap clear

**By End of Phase 2 (Dec 6)**:
- Triplet loss integrated into production candidate
- All research models benchmarked against V4 baseline
- Explainability reports published

---

## 💡 Key Principles

1. **Empirical Validation**: Every claim backed by metrics
2. **Reproducibility**: All experiments documented with seeds/configs
3. **Explainability**: Understand why models work, not just that they do
4. **Production Focus**: Research informs deployment, not vice versa

---

**Track**: Research (Track B)  
**Updated**: November 22, 2025  
**Next Review**: November 29, 2025
