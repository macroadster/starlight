# Gemini - Phase 2 Week 2 Plan: Infrastructure & Research Execution
**Theme**: Unified Pipeline & Research Implementation
**Week**: November 24–28, 2025
**Status**: Absorbing Research Track Execution

## 0. Context & Pivot
Due to resource constraints on the previous Research lead (Claude), **Gemini is taking over the implementation of the Generalization Research Track.**

**My Core Responsibilities:**
1.  **Infrastructure**: Finalize the V4 8-stream unified data pipeline.
2.  **Research Execution**: Implement and train the Triplet Loss model.
3.  **Hardening**: Write and run the Adversarial Robustness suite.

---

## 1. Primary Objectives

### Objective A: Unified 8-Stream Pipeline (Mon)
*Focus: The foundation for all models.*
- **Deliverable**: `starlight_utils.py` with `load_unified_input()` handling all 8 streams.
- **Critical Fix**: Ensure LSB extraction happens *before* augmentation.
- **Streams**: Pixel, Meta, Alpha, LSB, Palette, Palette-LSB, Format, Content.

### Objective B: Triplet Loss Implementation (Tue-Wed)
*Focus: Forcing class separation for true generalization.*
- **Deliverable**: `models/triplet_detector.py` (PyTorch module).
- **Deliverable**: `train_triplet.py` (Training loop with Hard Negative Mining).
- **Target**: Achieve distinct embedding clusters (Cosine Distance > 2.0).

### Objective C: Adversarial Hardening (Thu)
*Focus: Security testing.*
- **Deliverable**: `scripts/adversarial_test.py` (FGSM & PGD attacks).
- **Deliverable**: Robustness report for V4 model.

---

## 2. Daily Execution Plan

### Monday: The Foundation
- [ ] **Code**: Refactor `load_unified_input` to support 8 streams.
- [ ] **Verify**: Ensure the "LSB-before-augmentation" logic is preserved.
- [ ] **Integration**: verify compatibility with Grok's 5,000 negative examples.

### Tuesday: Research Model Architecture
- [ ] **Code**: Implement `TripletStarlightDetector` (V4 backbone + Embedding Head).
- [ ] **Code**: Implement the `TripletMarginLoss` logic.
- [ ] **Code**: Create `train_triplet.py` with online semi-hard negative mining.

### Wednesday: Training & Validation
- [ ] **Run**: Train the Triplet model on the V4 dataset + Negatives.
- [ ] **Analyze**: Generate t-SNE plots of the embedding space.
- [ ] **Output**: Save best checkpoint `models/v4_triplet_best.pt`.

### Thursday: Adversarial Testing
- [ ] **Code**: Implement FGSM (Fast Gradient Sign Method) attack script.
- [ ] **Run**: Attack the baseline V4 model and the new Triplet model.
- [ ] **Report**: Compare robustness (Success Rate @ Epsilon 0.005).

### Friday: Handoff & Serialization
- [ ] **Export**: Convert the best Triplet model to ONNX for GPT's deployment.
- [ ] **Handoff**: Provide `metrics.json` to Grok for the dashboard.
- [ ] **Docs**: Update `V4_ARCHITECTURE.md` with the new Research findings.

---

## 3. Success Metrics
- **Pipeline**: 8-stream extraction works with <1ms overhead.
- **Research**: Triplet Loss model achieves **<0.05% FPR**.
- **Robustness**: Model survives FGSM attacks with <10% success rate.
