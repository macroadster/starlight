# Starlight - Phase 2 Roadmap: Production & Hardening

**Phase**: 2 (Production Ready)
**Start Date**: November 22, 2025
**Previous Status**: [Phase 1 Complete - Outstanding Success](https://www.google.com/search?q=status.md)

## 🏁 Phase 1 Retrospective: The Generalization Breakthrough

We have successfully moved from hand-crafted rules to a data-driven unified pipeline.

  * **Architecture**: V4 8-Stream Unified Pipeline (Pixel, Meta, Alpha, LSB, Palette, Palette-LSB, Format, Content).
  * **Performance**: 0.07% False Positive Rate (Target was \< 5%).
  * **Data**: Integrated 5,000 negative examples (teaching "what is NOT steganography").
  * **Codebase**: Eliminated all special cases/hardcoded rules in `scanner.py`.

-----

## 🎯 Phase 2 Objectives

### 1\. Production Deployment (The "Grok" & "Gemini" Pipeline)

*Focus: transforming the research model into a deployable, high-speed artifact.*

  * [ ] **ONNX Export Finalization**: Ensure the 8-stream architecture exports cleanly without operator support issues.
  * [ ] **Quantization Strategy**: Apply dynamic quantization (INT8) to reduce model size \<10MB while maintaining \<0.1% FPR.
  * [ ] **Docker Containerization**: Package the ONNX runtime + preprocessing logic into a lightweight container.
  * [ ] **REST API**: Build a fast FastAPI/Flask interface serving the ONNX model.

### 2\. Advanced Research & Hardening

*Focus: pushing generalization even further using advanced loss functions.*

  * [ ] **Triplet Loss Implementation**: Implement Triplet Loss to force the model to cluster "steg" and "clean" embeddings more distinctly.
  * [ ] **Adversarial Testing**: Test the V4 model against adversarial attacks designed to fool standard CNNs.
  * [ ] **Explainability (XAI)**: Visualize which of the 8 streams contributes most to specific detections (e.g., does the model look at LSB or Alpha more for PNGs?).

### 3\. Monitoring & Operations (Week 2 Dashboard)

*Focus: ensuring stability in the wild.*

  * [ ] **Performance Dashboard**: Create a view for Inference Latency, Throughput, and Error Rates.
  * [ ] **Drift Detection**: Set up alerts if the input distribution (e.g., image sizes, formats) shifts significantly from training data.
  * [ ] **Feedback Loop**: System to capture low-confidence predictions for manual review and future training.

-----

## 📅 Technical Timeline (Nov 22 - Dec 06)

### Week 1: Deployment & Optimization (Nov 22 - Nov 29)

1.  **Monday-Tuesday**: Finalize ONNX export for V4 8-stream model.
2.  **Wednesday**: Run quantization benchmarks (compare FP32 vs INT8 accuracy).
3.  **Thursday**: Build Docker container with optimized runtime.
4.  **Friday**: API Development and Load Testing.

### Week 2: Research & Hardening (Nov 30 - Dec 06)

1.  **Monday-Tuesday**: Implement and train with Triplet Loss.
2.  **Wednesday**: Run regression tests on the Triplet Loss model using the 5,000 negatives.
3.  **Thursday**: Adversarial robustness testing.
4.  **Friday**: Final release candidate (RC1) of Starlight V4.

-----

## 📊 Success Metrics (Updated)

| Metric | Old Target (Phase 1) | New Target (Phase 2) | Current Baseline |
| :--- | :--- | :--- | :--- |
| **False Positive Rate** | \< 5% | **\< 0.05%** | 0.07% |
| **Inference Speed** | \< 10ms | **\< 5ms (Quantized)** | \~8ms (Est.) |
| **Model Size** | \< 10MB | **\< 5MB** | TBD |
| **Validation Accuracy** | \> 85% | **\> 99.5%** | High |

-----

## 🛠️ Immediate Actions (Next 48 Hours)

1.  **Documentation Update**: Update `V4_UTILS_SPEC.md` to reflect any final tweaks from the unification process.
2.  **Export Test**: Run the existing `experiments/validate_extraction_streams.py` against the ONNX export pathway to ensure feature parity.
3.  **Dashboard Scoping**: define the exact metrics Grok needs to visualize for the Week 2 dashboard.

-----

**Next Review**: 2025-11-29 (End of Deployment Week)

