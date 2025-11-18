# AI Consensus – Project Starlight (2025)

## 🚀 Quick Start
- **Goal:** Provide any AI session with a concise, up‑to‑date overview of the project in under 5 minutes.
- **Core Docs:** `status.md`, `docs/README.md`, `docs/archive/2025_q4_resolution_log.md`.

### 🚨 Are You on Track A or B?
| Track | Can Touch? | Branch |
|------|------------|--------|
| **A (Production)** | Only bug fixes | `production` |
| **B (Research)** | Full access | `main` |

**Never push to `production` without lead approval.**

### 🎯 Current Tasks (Week of Nov 17‑21)
| AI | Task | File |
|----|------|------|
| Grok | HF deploy + negative samples | `docs/plans/grok_next.md` |
| ChatGPT | Unified trainer prototype | `docs/plans/chatgpt_next.md` |
| Claude | Dataset spec v1.0 | `docs/plans/claude_next.md` |
| Gemini | Multi-stream extraction | `docs/plans/gemini_next.md` |

## 📊 System Status Dashboard (as of 2025-11-18)
- **Architecture:** V4 unified 6-stream model
  - Metadata stream (2048 features from EXIF + tail)
  - Alpha stream (CNN on alpha channel LSB)
  - LSB stream (CNN on RGB LSB patterns)
  - Palette stream (FC on 768 palette features)
  - Format features (6 format indicators)
  - Content features (6 content statistics)
  - Bit order stream (3 one-hot: lsb-first, msb-first, none)
- **Production Model:** `detector_balanced.onnx` (V4 unified architecture).
- **Special Cases:** ✅ Architectural special cases eliminated in V4. ⚠️ Scanner post-processing heuristics remain (see Known Pitfalls).
- **False Positive Rate:** 0.32% (excellent) – maintained with V4 architecture.
- **Detection Rate:** 96.40% (excellent).
- **Palette/Index Data Loss:** Fixed in `trainer.py` (no unconditional RGB conversion for palette images).
- **SDM:** Fully removed from codebase and datasets.
- **AI42 Prefix:** Used **only** for Alpha Protocol.
- **LSB Bit Order:** Supports both LSB-first and MSB-first via `bit_order` metadata.
- **Extractor:** Robust LSB extraction with fallback decoding.

## 📋 Decision Registry
- **Trainer:** Use `embedding_type` metadata (`RGB`, `ALPHA`, `PALETTE`).
- **Scanner:** Share `scanner_spec.json` with trainers; includes special‑case rules required for production.
- **Tests:** `test_starlight.py` must validate both baseline and special‑case behaviours.
- **Format Spec:** `docs/STEGO_FORMAT_SPEC.md` – v2.0 (includes `bit_order`).
- **Phase 1 Objective (Track B - Research):** Re‑build dataset, develop V3/V4 multi‑stream architecture (pixel + alpha + LSB + palette + metadata + format features), train models to learn domain constraints currently encoded in special cases. **Timeline:** 18‑24 months to production‑ready generalization.
- **Phase 1 Objective (Track A - Production):** Maintain detector_balanced.onnx with special cases; integrate Grok's multi‑format EXIF/EOI; weekly health checks.

## 🔍 Method Specification Summary
| Method | Channel / Stream | Model Stream | AI42 Prefix | Bit Order |
|--------|------------------|--------------|-------------|-----------|
| Alpha  | Alpha channel (RGBA) | Alpha CNN | ✅ (Alpha only) | LSB-first |
| Palette| Palette indices | Palette FC | ❌ | LSB-first |
| LSB    | RGB channels | LSB CNN | ❌ | LSB-first / MSB-first |
| EXIF   | Metadata stream | Metadata CNN | ❌ | — |
| EOI    | JPEG/PNG tail stream | Metadata CNN | ❌ | — |

## ⚠️ Known Pitfalls & Anti‑Patterns

### Dataset Quality Issues (Priority Fix for Phase 1)
1. **Invalid Labels** - Alpha steganography labels on RGB images
   - **Impact:** Model cannot learn impossible constraints
   - **Solution:** Remove ~15 % of invalid labels from datasets
   
2. **Format Mismatches** - Clean images don't match stego distribution
   - **Impact:** Model learns spurious format correlations
   - **Solution:** Generate format-matched clean images
   
3. **Missing Negatives** - No examples teaching what stego is NOT
   - **Impact:** Model cannot learn special case constraints
   - **Solution:** Generate 5,000+ negative counterexamples
   
4. **Corrupted Signals** - Augmentations before LSB extraction
   - **Impact:** Training data contains no signal to learn
   - **Solution:** Extract LSB BEFORE any augmentation

### Scanner Post-Processing Heuristics (Audit Needed)
The scanner (`scanner.py`) still contains post-processing validation logic that acts as a safeguard against misclassifications. An audit is required to determine if these are still necessary with the V4 model.

1.  **RGB Alpha Override** (lines 154-166)
    -   **Logic:** If the model predicts alpha steganography on an image without an alpha channel, the prediction is overridden to "clean".
    -   **Question:** Does the V4 model learn this constraint inherently from the data?

2.  **Message Content Validation** (lines 136-284)
    -   **Logic:** For high-confidence predictions, the scanner extracts the potential message and rejects it if the content is repetitive or appears meaningless (e.g., all null bytes).
    -   **Question:** Is this validation still needed to prevent false positives with the V4 model?

3.  **Method-Specific Thresholds** (line 127)
    -   **Logic:** Different confidence thresholds are used for each steganography method (Alpha: 0.7, Palette: 0.98, LSB: 0.95, EXIF: 0.5, Raw: 0.95).
    -   **Question:** Should the V4 model learn optimal decision thresholds, or are these manually-tuned thresholds still superior?

**Action Item:** Test V4 performance with and without scanner heuristics to determine if they can be safely removed.

### Pipeline Anti-Patterns
- **Palette Images:** Do **not** convert to RGB before training or scanning; preserve index data.
- **Spatial Augmentations:** NEVER apply flips/rotations before LSB extraction - corrupts the signal
- **Metadata Streams:** Ensure `bit_order` is present for generic LSB stego; EXIF/EOI require raw‑byte handling.
- **AI42 Prefix:** Must appear **only** in Alpha‑protocol embeddings.

## 📈 Performance Baselines

### Track A - Production (Current)
- **Stego detection:** 96.40 % (target: ≥95 %) ✅
- **False positives:** 0.32 % (target: ≤1 %) ✅
- **AUC‑ROC:** ≥0.95 for all methods ✅
- **Method:** detector_balanced.onnx + special cases

### Track B - Research (Baseline)
- **Stego detection:** Variable by method
- **Pre-V4 Baseline (False Positives):** 17.82% (using conservative model without special cases)
- **Post-V4 Status (False Positives):** [NEEDS MEASUREMENT] - unified architecture performance TBD
- **Dataset quality:** ~85% valid labels (target: 100%)
- **Method:** V4 unified 6-stream architecture
- **Goal:** Learn domain constraints to eliminate special cases
- **Timeline:** 18-24 months to production parity

## 🤝 Inter‑AI Coordination Protocol
1. **Read** `docs/coordination/restructure_notice.md` before making any documentation changes.
2. **24‑hour sync window** after any restructuring to allow all agents to checkpoint.
3. **Phase 1 constraint:** No plan files may be altered until the unified consensus (this document) is finalized.
4. **V4 Architecture Status:** All agents must acknowledge that:
   - ✅ Architectural special cases have been eliminated in the V4 merge.
   - ⚠️ Scanner heuristics are still present for post-processing validation. Their removal is pending testing.
   - 📋 Performance benchmarks are needed to compare V4 with and without scanner heuristics.

## ✅ Validation Checklist (for current state & Phase 1 transition)
- [ ] Trainer uses V4 unified architecture (6 streams + bit_order)
- [ ] Scanner uses V4 model (detector_balanced.pth or .onnx)
- [ ] Trainer uses `embedding_type` metadata consistently
- [ ] **NEW:** Benchmark V4 performance with scanner heuristics enabled
- [ ] **NEW:** Benchmark V4 performance with scanner heuristics disabled
- [ ] **NEW:** Document V4 false positive rate vs pre-V4 baseline (17.82%)
- [ ] **NEW:** Verify V4 learns RGB-alpha constraint (no alpha stego on RGB images)
- [ ] `test_starlight.py` validates V4 model performance
- [ ] Documentation reflects V4 architecture reality

## 📚 Historical Context
All resolved discussions, obsolete analyses, and big‑pickle research are archived in **`docs/archive/2025_q4_resolution_log.md`**. The full archive resides under `docs/archive/big-pickle/`.

*This consensus reflects the latest status (production with special cases) and the Phase 1 roadmap toward a unified V3 dual‑input architecture.*