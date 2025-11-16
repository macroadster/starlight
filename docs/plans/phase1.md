# ✅ **REVISED `phase1.md` — Unified Cross-Agent Plan**

# (Updated to match GPT-Next + Claude feedback)

# ============================================

Below is the **fully revised Phase 1 plan**, consistent with the new consensus structure and the updated GPT-Next + Grok roles.

You can replace the current `docs/phase1.md` with this version.

---

# **Phase 1 — Unified Cross-Agent Plan (Revised 2025-11-15)**

### **Objective:** Enable generalization without special cases by repairing datasets, stabilizing the multi-stream architecture, and preparing for V3 unified model training.

---

# **1. Phase 1 Overview**

Phase 1 fixes all root causes preventing generalization:

* Incorrect labels
* Impossible labels (alpha-in-RGB, palette-on-RGB)
* Missing negative counterexamples
* Lack of multi-format metadata alignment
* Preprocessing inconsistencies (augment before LSB extraction, resizing, etc.)
* Incomplete EXIF/EOI pipelines

**Goal:** Achieve a dataset + architecture foundation stable enough to train V3 unified multi-stream model without special-case logic.

---

# **2. Track Responsibilities (Aligned With Claude’s Model)**

```
Track A — Production (Stable)
    Grok (support only)
Track B — Research (Active)
    Claude, Gemini, Grok
Track C — Documentation & Coordination
    GPT-Next (Phase 1 only)
```

---

# **3. Phase 1 Tasks (Unified)**

## **3.1 Dataset Reconstruction (Critical Path)**

Owned by: **Claude + Gemini + Grok**

### Required fixes:

1. Remove all impossible labels
2. Re-extract every stego sample → verify bitmatch
3. Validate PNG/GIF/WebP palette and index streams
4. Standardize LSB extraction (before augmentation)
5. No resizing (only padding)
6. Format-matched cleans for every stego method
7. Generate negative counterexamples (Grok, Week 1–2)
8. Produce manifest_v3.jsonl (signed and validated)

### Output:

```
data/manifest_v3.jsonl
data/consistency_report.md
```

---

## **3.2 Unified Multi-Stream Architecture (V3)**

Owned by: **Claude + Gemini**

Streams:

1. Pixel
2. Alpha
3. LSB
4. Palette
5. Metadata (EXIF/EOI)
6. Format Signature

**Constraints:**

* ONNX-exportable
* Quantization-safe
* No custom ops

### Output:

```
models/v3_architecture_spec.md
```

---

## **3.3 Negative Counterexample Generator**

Owned by: **Grok (Week 1–2)**

### Requires:

* Schema reviewed by Claude
* Verified against manifest rules
* 100-sample prototype validated
* Extraction-verified full sample set

### Purpose:

Teach model format impossibilities and prevent special cases.

### Output:

```
data/negatives_v1/
docs/data/negative_schema.md
```

---

## **3.4 Cross-Dataset Evaluation**

Owned by: **Gemini**

Evaluate against all 22k images:

* FP, FN, precision/recall
* per-method recall
* per-format stability
* throughput under CPU only

Output:

```
reports/v3_validation_metrics.json
```

---

## **3.5 Hugging Face Deployment (support Track A + B)**

Owned by: **Grok**

Week 1:

* Deploy production model
* Deploy research model
* Publish model card + minimal demo (NO dashboard yet)

Week 2:

* Add dashboard
* Add evaluation notebooks

---

## **3.6 Consensus Restructure (Track C)**

Owned by: **GPT-Next (Phase 1 Only)**

See GPT-Next final plan above.

---

# **4. Milestones**

### **Week 1**

* HF deployment (Grok)
* Consensus restructure draft (GPT-Next)
* Negative sample schema (Grok + Claude)
* Dataset extraction validation start (Gemini)

### **Week 2**

* 5,000 negative counterexamples
* Finalized V3 architecture spec
* Updated dataset manifest v3

### **Week 3**

* Full V3 training pipeline ready
* Initial unified model training (prototype)

---

# **5. Phase 1 Success Criteria**

### Dataset:

* 100% extraction-verified stego samples
* No impossible labels
* Balanced across methods and formats
* Negative examples integrated

### Architecture:

* V3 spec stable, agreed by Claude & Gemini
* ONNX graph validated
* Feature streams clean

### Coordination:

* New ai_consensus.md clear, concise, status-first
* All historical content properly archived
* Navigation correct

---

# **6. Notes**

Phase 1 must complete before any attempt to eliminate special-case rules.
Generalization cannot succeed without dataset integrity + unified architecture.

---

# END OF PHASE 1 DOCUMENT

