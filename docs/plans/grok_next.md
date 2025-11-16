# Grok's Next Week Plan: Project Starlight (Week of Nov 17–21, 2025)

**File:** `docs/plans/grok_next.md`  
**Author:** Grok 4 (xAI)  
**Last Updated:** 2025-11-15T15:36:00 PST  
**Status:** **REFRESHED** — EOI/EXIF multi-format complete; V3 lightweight model trained; Hugging Face export pending; **Strategic Shift to Avoid Rule-Based Workarounds**

---

## Executive Summary
**This is a full refresh of `grok_next.md` based on `status.md` (2025-11-15), `phase1.md`, and your directive to avoid hybrid scanner and rule-based special cases.**

**Your Concern**:  
Hybrid scanner risks turning AI into mechanical rules, undermining the goal of building AI with **common sense**. Special cases are workarounds, not progress.

**My Response**:  
**I agree.** Special cases encode domain knowledge but bypass true generalization. The real fix is **teaching the model common sense** through better data and training, not hardcoding rules.

**New Strategy**:  
- **Drop Hybrid Scanner** and rule-based tasks (e.g., `config.yaml`).  
- **Focus on Research (Track B)**: Accelerate dataset repair, negative counterexamples, and feature distillation to make the model learn what special cases currently enforce.  
- **Production Support (Track A)**: Deliver Hugging Face deployment and monitoring dashboard to maintain community access and observability.  

**Alignment with `phase1.md`**:  
- **Track A (Production)**: Lead Hugging Face publishing + monitoring.  
- **Track B (Research)**: Lead negative counterexample generation + special case feature distillation.  
- **Track C (Oversight)**: Contribute to Friday sync.

---

## Weekly Commitments (Nov 17–21)

| Priority | Task | Track | Status |
|---------|------|-------|--------|
| 1 | **Hugging Face Dual Deployment** | A+B | Pending |
| 2 | **Production Monitoring Dashboard** | A | New |
| 3 | **Negative Counterexample Generator** | B | New |
| 4 | **Special Case → Feature Pipeline** | B | Research |

---

### 1. Hugging Face Dual Deployment (High Priority — Community Access)

**Goal**: Publish **dual-track systems** (production + research) to enable community testing and feedback.

#### Repos:
- `macroadster/starlight-prod`: `detector_balanced.onnx` + inference script (no rules)  
- `macroadster/starlight-research`: V3/V4 models, dataset manifest, training logs

**Model Cards**:
- **Prod**: "Production-ready: 0.32% FP, 96.4% detection"  
- **Research**: "Experimental: Towards zero special cases via data repair"

**Actions**:
- Update `scripts/hf_export.py` to upload both repos:
  ```python
  from huggingface_hub import HfApi
  api = HfApi()
  api.upload_folder(
      folder_path="models/",
      repo_id="macroadster/starlight-prod",
      path_in_repo="onnx/"
  )
  api.upload_folder(
      folder_path="research/",
      repo_id="macroadster/starlight-research",
      path_in_repo="v3/"
  )
  ```
- Deploy Gradio Spaces for both repos.  
- Add **"Try on Raspberry Pi"** guide with ONNX Runtime instructions.  
- Include dataset manifest and validation metrics in research repo.

**Deliverables**:
- Live repos: `macroadster/starlight-prod` + `macroadster/starlight-research`  
- Gradio demos for inference  
- Model cards with benchmarks (latency, FP rate, method coverage)

**Deadline**: EOD Tuesday

---

### 2. Production Monitoring Dashboard (High Priority — Observability)

**Goal**: Provide real-time visibility into production scanner performance to track FP rate and dataset drift.

#### Structure: `monitor/`
```bash
monitor/
├── dashboard.py        ← Streamlit
├── logs/
└── metrics.db          ← SQLite
```

**Metrics Tracked**:
- FP rate (rolling 1h/24h)  
- Detection rate per method (alpha, lsb, palette, exif, eoi)  
- Latency distribution  
- Dataset drift alerts (e.g., new image formats)  

**Auto-Alerts**:
```python
if fp_rate_1h > 1.0:
    slack_alert("FP spike detected: Check dataset drift")
```

**Deliverable**: Live dashboard at `http://localhost:8501`

**Deadline**: EOD Wednesday

---

### 3. Negative Counterexample Generator (High Priority — Dataset Repair)

**Goal**: Generate **negative examples** to teach the model what special cases currently enforce, addressing dataset flaws (e.g., alpha labels on RGB images).

#### Approach
Instead of hardcoding rules, create training data that teaches the model **common sense** constraints:
- **RGB cannot have alpha**: Generate 1,000 RGB images labeled `clean`.  
- **Uniform alpha ≠ stego**: Generate 1,000 PNGs with uniform alpha (all 255) labeled `clean`.  
- **LSB noise ≠ stego**: Generate 1,000 GIFs with dithering artifacts labeled `clean`.  
- **Repetitive hex ≠ stego**: Generate 1,000 images with synthetic noise in LSB labeled `clean`.

#### `scripts/negative_generator.py`
```python
class NegativeGenerator:
    def generate(self, constraint, count=1000):
        if constraint == "no_alpha":
            return [self._rgb_image() for _ in range(count)]
        if constraint == "uniform_alpha":
            return [self._uniform_alpha_png() for _ in range(count)]
        # Add LSB noise, hex patterns, etc.
```

**Integration**:
- Output to `data/training/v3_negatives/`  
- Include in unified dataset spec (`B1.3` from `phase1.md`)  
- Validate with Claude/Gemini for format correctness

**Deliverables**:
- 5,000+ negative samples covering all special case constraints  
- JSON manifest: `data/training/v3_negatives/manifest.jsonl`  
- Validation report: `data/training/v3_negatives/report.json`

**Deadline**: EOD Thursday

---

### 4. Special Case → Feature Pipeline (Medium Priority — Research)

**Goal**: Distill special case knowledge into **trainable features** to reduce reliance on rules.

#### Research Plan
1. **Analyze Special Cases**:
   - RGB → no alpha  
   - Uniform alpha → no payload  
   - LSB must extract meaningful content  
   - Repetitive hex ≠ stego  
2. **Feature Engineering**:
   - Add format metadata (e.g., `has_alpha_channel`, `image_mode`) to input tensors.  
   - Compute statistical features (e.g., `alpha_std`, `lsb_entropy`) as auxiliary inputs.  
3. **Auxiliary Loss**:
   ```python
   loss = ce_loss + λ * constraint_loss
   # constraint_loss penalizes alpha_conf > 0 for RGB images
   ```
4. **Synthetic Data**: Use negative counterexamples (Task 3) to train.

**Deliverable**:  
- `research/rule_distillation.ipynb`: Notebook with feature extraction and training experiments  
- Report: `research/distillation_report.md` summarizing impact on FP rate

**Deadline**: EOD Friday

---

## Timeline (Nov 17–21)

| Day | Focus | Deliverable |
|-----|-------|-------------|
| **Mon** | HF Deployment Setup | `hf_export.py` updated, repos initialized |
| **Tue** | HF Deployment Complete | Live repos + Gradio Spaces |
| **Wed** | Monitoring Dashboard | Streamlit dashboard live |
| **Thu** | Negative Generator | 5,000+ negative samples + manifest |
| **Fri** | Feature Pipeline + Sync | Distillation notebook + Friday sync input |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Negative samples bias model | Medium | Validate with cross-submission datasets |
| Dataset integration fails | Low | Coordinate with Claude/Gemini early |
| HF deployment delays | Low | Use pre-existing `hf_export.py` as base |

---

## Trajectory Alignment

| Horizon | Goal |
|--------|------|
| **This Week** | HF repos live, monitoring active, negative samples ready |
| **Nov 30** | Unified dataset repaired with negatives |
| **Dec 15** | V3/V4 trained on new dataset, FP < 5% |
| **Q1 2026** | Model learns 50% of special case constraints |

---

## Final Note (for Terminal Grok)

> **No rules. Just data.**  
> Teach the model common sense through **negative examples** and **smart features**.  
> Special cases are a symptom of bad data — fix the root cause.

**Context Command**:
```bash
cat docs/plans/grok_next.md && cat phase1.md | grep -A5 "Track B"
```

**We are building AI that *learns* constraints, not one that *follows* rules.**

---

**End of Plan**  
*Commit now. Execute Monday.*
