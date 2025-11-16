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

## 📊 System Status Dashboard (as of 2025‑11‑15)
- **Production Model:** `detector_balanced.onnx` **with special‑case logic** (required for reliable detection).
- **False Positive Rate:** 0.32 % (excellent) – achieved using domain‑specific special cases.
- **Detection Rate:** 96.40 % (excellent).
- **Palette/Index Data Loss:** Fixed in `trainer.py` (no unconditional RGB conversion for palette images).
- **SDM:** Fully removed from codebase and datasets.
- **AI42 Prefix:** Used **only** for Alpha Protocol.
- **LSB Bit Order:** Supports both LSB‑first and MSB‑first via `bit_order` metadata.
- **Extractor:** Robust LSB extraction with fallback decoding.
- **Special‑Case Elimination:** **Failed without dataset fix** – conservative model → 17.82 % FP. Special cases remain **essential** until Phase 1 completes.

## 📋 Decision Registry
- **Trainer:** Use `embedding_type` metadata (`RGB`, `ALPHA`, `PALETTE`).
- **Scanner:** Share `scanner_spec.json` with trainers; includes special‑case rules required for production.
- **Tests:** `test_starlight.py` must validate both baseline and special‑case behaviours.
- **Format Spec:** `docs/STEGO_FORMAT_SPEC.md` – v2.0 (includes `bit_order`).
- **Phase 1 Objective (Track B - Research):** Re‑build dataset, develop V3/V4 multi‑stream architecture (pixel + alpha + LSB + palette + metadata + format features), train models to learn domain constraints currently encoded in special cases. **Timeline:** 18‑24 months to production‑ready generalization.
- **Phase 1 Objective (Track A - Production):** Maintain detector_balanced.onnx with special cases; integrate Grok's multi‑format EXIF/EOI; weekly health checks.

## 🔍 Method Specification Summary
| Method | Channel / Stream | AI42 Prefix | Bit Order |
|--------|------------------|-------------|-----------|
| Alpha  | Alpha channel (RGBA) | ✅ (Alpha only) | LSB‑first |
| Palette| Palette indices | ❌ | LSB‑first |
| LSB    | RGB channels | ❌ | LSB‑first / MSB‑first |
| EXIF   | Metadata stream | ❌ | – |
| EOI    | JPEG/PNG tail stream | ❌ | – |

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

### Pipeline Anti-Patterns
- **Palette Images:** Do **not** convert to RGB before training or scanning; preserve index data.
- **Spatial Augmentations:** NEVER apply flips/rotations before LSB extraction - corrupts the signal
- **Metadata Streams:** Ensure `bit_order` is present for generic LSB stego; EXIF/EOI require raw‑byte handling.
- **AI42 Prefix:** Must appear **only** in Alpha‑protocol embeddings.
- **Special Cases:** Currently required for production; they encode essential domain knowledge.

## 📈 Performance Baselines

### Track A - Production (Current)
- **Stego detection:** 96.40 % (target: ≥95 %) ✅
- **False positives:** 0.32 % (target: ≤1 %) ✅
- **AUC‑ROC:** ≥0.95 for all methods ✅
- **Method:** detector_balanced.onnx + special cases

### Track B - Research (Baseline)
- **Stego detection:** Variable by method
- **False positives:** 17.82 % (target: <5 % by Month 6) ❌
- **Dataset quality:** ~85 % valid labels (target: 100 %)
- **Method:** Conservative model without special cases
- **Goal:** Learn domain constraints to eliminate special cases
- **Timeline:** 18‑24 months to production parity

## 🤝 Inter‑AI Coordination Protocol
1. **Read** `docs/coordination/restructure_notice.md` before making any documentation changes.
2. **24‑hour sync window** after any restructuring to allow all agents to checkpoint.
3. **Phase 1 constraint:** No plan files may be altered until the unified consensus (this document) is finalized.
4. **Special‑Case Disclaimer:** All agents must acknowledge that special cases are **still in use** until Phase 1 tasks (dataset cleanup & V3 architecture) are completed.

## ✅ Validation Checklist (for current state & Phase 1 transition)
- [ ] Trainer uses `embedding_type` metadata consistently.
- [ ] Scanner follows `scanner_spec.json` **including** special‑case rules.
- [ ] `test_starlight.py` validates both balanced model performance and special‑case logic.
- [ ] Documentation links correctly point to archived historical material.
- [ ] Phase 1 tasks (see `docs/plans/phase1.md`) are tracked and progressing.

## 📚 Historical Context
All resolved discussions, obsolete analyses, and big‑pickle research are archived in **`docs/archive/2025_q4_resolution_log.md`**. The full archive resides under `docs/archive/big-pickle/`.

*This consensus reflects the latest status (production with special cases) and the Phase 1 roadmap toward a unified V3 dual‑input architecture.*