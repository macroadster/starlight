# Grok's Next Week Plan: Project Starlight (Week of Nov 10–15, 2025)

**File:** `docs/plans/grok_next.md`  
**Author:** Grok 4 (xAI)  
**Last Updated:** 2025-11-08T23:59:00Z  
**Status:** Draft — Commit to `https://github.com/macroadster/starlight/tree/main/docs/plans/` upon repo unification  

---

## 🎯 Executive Summary
This document outlines my **concrete, actionable commitments** for the upcoming week, aligned with Project Starlight's trajectory toward **V3 lightweight model** (per `chatgpt_proposal.md`) and **blockchain-scale deployment**.  

**Core Themes:**  
- **Unify codebase** to eliminate merge overhead and enable runtime fusion.  
- **Generalize EOI** beyond JPEG for multi-format support.  
- **Prototype V3** dual-stream lightweight detector (1.4 MB → <1 MB INT8).  
- **Benchmark & export** for Hugging Face (no Python routing).  

**Success Metric:** By EOD Nov 15, deliver a **unified PR** with ≥15 img/sec scanner performance on RPi4 + full HF export script.  

**Dependencies:**  
- Unified monorepo setup (blocked on Project Lead/Big Pickle).  
- Access to shared `docs/` for persistent context (terminal/browser sync).  

---

## 📋 Weekly Commitments (Prioritized)
All deliverables target **EOD Friday, Nov 15, 2025**. Progress will be logged daily in `docs/progress/grok_daily.md` (to be created).  

### 1. **Unified Codebase Migration** (High Priority — Blocker Resolution)  
   - **Goal:** Migrate my `stego_encoder_v2.py`, `detector_dual.onnx`, and `load_dual_input()` to a **single Git monorepo** (branch: `grok-integration`).  
   - **Actions:**  
     - Fork/clone `https://github.com/macroadster/starlight` into unified structure:  
       ```
       starlight/
       ├── models/          # ONNX exports (dual.onnx, v3_light.onnx)
       ├── src/             # Core modules (encoder.py, detector.py, preprocess.py)
       ├── scripts/         # Build/export (train_dual.py, hf_export.py)
       ├── tests/           # Integration suite (encode→detect on 100k samples)
       └── docs/            # Specs + plans (this file included)
       ```  
     - Implement **branch-per-agent** workflow: `main` for consensus, `grok/*` for my contributions.  
     - **CI/CD Hook:** Auto-run tests on PRs (using GitHub Actions: ONNX validation + speed benchmarks).  
   - **Output:** PR #XX: "Grok Integration — Dual-Stream + EOI Generalization"  
   - **Estimated Effort:** 4–6 hours (post-repo setup).  
   - **Risk:** Delay if monorepo not ready by Nov 10 → Fallback: Local proto-repo shared via Gist.  

### 2. **EOI Generalization (Multi-Format Tail Extraction)** (Medium Priority — v2.0 Refinement)  
   - **Goal:** Extend EOI from JPEG-only (`0xFFD9`) to **post-image-tail** for PNG (`IEND`), GIF (`0x3B`), WebP (`VP8X` chunk end), etc.  
   - **Actions:**  
     - Update `load_dual_input()` in `src/preprocess.py`:  
       ```python
       def extract_post_tail(raw_bytes, format_hint='auto'):
           tails = {
               'jpeg': raw_bytes[raw.rfind(b'\xFF\xD9') + 2:] if raw.rfind(b'\xFF\xD9') != -1 else b"",
               'png': raw_bytes[raw.rfind(b'IEND') + 4:] if raw.rfind(b'IEND') != -1 else b"",  # +4 for chunk length
               'gif': raw_bytes[raw.rfind(b';') + 1:] if raw.rfind(b';') != -1 else b"",     # Trailer terminator
               'webp': extract_webp_tail(raw_bytes),  # Custom: After VP8X/RIFF end
           }
           return tails.get(format_hint, b"")  # Pad to 512 bytes for meta_tensor
       ```  
     - **Validation:** Test on 1k multi-format stego samples (embed tail payloads via `stego_encoder_v2.py`).  
     - **Metrics Target:** Detection AUC ≥0.990 for EOI across formats (no JPEG bias).  
   - **Output:** Updated `detector_dual.onnx` with generalized meta path.  
   - **Estimated Effort:** 3 hours.  
   - **Risk:** Format-specific edge cases (e.g., animated GIFs) → Mitigate with Pillow + custom parsers.  

### 3. **V3 Lightweight Model Prototype** (High Priority — Per chatgpt_proposal.md)  
   - **Goal:** Shrink dual-stream model to **<1 MB** while maintaining ≥0.98 AUC, targeting blockchain nodes (15+ img/sec).  
   - **Actions:**  
     - **Prune & Quantize:** Start from `detector_dual.onnx` → Apply dynamic INT8 quantization + MobileNet-V3-Tiny backbone.  
       ```bash
       # In scripts/train_v3.py
       python -m onnxruntime.quantization.quantize_dynamic detector_dual.onnx v3_light.onnx --per_channel --reduce_range
       # Fuse meta MLP earlier: Reduce from 1024→512 dim input
       ```  
     - **Ensemble Fusion:** Bake method-specialized voting **into ONNX graph** (no runtime Python):  
       - Use ONNX `If` nodes for method routing based on filename/format.  
       - Weight fusion: Static tensors for specialist bonuses (e.g., Grok: +1.5x on EOI/LSB).  
     - **Benchmark:** RPi4 latency target: **<10 ms/img** (parallel: 15+/sec with 4 workers).  
   - **Output:** `v3_light.onnx` (0.8 MB) + `scanner_v3.py` (one-liner deploy).  
   - **Estimated Effort:** 6–8 hours (incl. training on 50k samples).  
   - **Risk:** AUC drop from pruning → A/B test vs. v2.0; fallback to hybrid if <0.95.  

### 4. **Hugging Face Export & Deployment Script** (Medium Priority — Sharing)  
   - **Goal:** Export consolidated model to HF Hub for community testing (no Python deps).  
   - **Actions:**  
     - Create `scripts/hf_export.py`:  
       ```python
       from huggingface_hub import HfApi, upload_file
       api = HfApi()
       api.upload_folder(
           folder_path="models/",
           repo_id="macroadster/starlight-v3",
           repo_type="model",
           path_in_repo="onnx/"  # Pure ONNX + README with inference example
       )
       ```  
     - **HF Model Card:** Include benchmarks, method coverage table, and "Try on RPi4" guide.  
     - **Test:** Verify ONNX Runtime inference in HF Spaces (dummy endpoint).  
   - **Output:** Live HF repo: `https://huggingface.co/macroadster/starlight-v3`  
   - **Estimated Effort:** 2 hours.  
   - **Risk:** Auth/setup → Use Project Lead's HF token.  

### 5. **Move Trainer and Scanner Logic to Top Level** (High Priority — Unification)
    - **Goal:** Relocate trainer.py, scanner.py, light_scanner.py, starlight_extractor.py from datasets/sample_submission_2025/ to top-level scripts/ directory for unified access.
    - **Actions:**
      - Copy and adapt files to scripts/, ensuring compatibility with ensemble model.
      - Update imports and paths in moved files.
      - Remove originals from sample_submission_2025/ after verification.
    - **Output:** Top-level scripts/ with unified trainer and scanner logic.
    - **Estimated Effort:** 2 hours.

### 6. **Integrate LSB Detection from Grok Submission** (High Priority — Model Improvement)
    - **Goal:** Enhance sample submission's LSB detection by integrating lsb_steganography.py and model components from datasets/grok_submission_2025/.
    - **Actions:**
      - Copy lsb_steganography.py to datasets/sample_submission_2025/ or top-level src/.
      - Integrate Grok's model/inference.py logic for better LSB handling.
      - Update sample's trainer.py and scanner.py to use improved LSB detection.
      - Test on LSB stego samples to ensure AUC ≥0.98.
    - **Output:** Updated sample submission with robust LSB detection.
    - **Estimated Effort:** 3 hours.

### 7. **Integration Test Suite Expansion** (Low Priority — Quality Gate)
    - **Goal:** ≥0.99 recall on 100k synthetic v2.0 samples (encode→transmit→detect).
    - **Actions:** Add multi-format tests (JPEG/PNG/GIF) + EOI generalization.
      - Use `pytest` in `tests/integration/`: Cover all 5 methods + clean baselines.
    - **Output:** `tests/integration_test_suite.py` (90% coverage).
    - **Estimated Effort:** 2 hours.

---

## ⏱️ Timeline (Nov 10–15)
| Day | Focus | Deliverable Checkpoint |
|-----|--------|------------------------|
| **Mon (10)** | Repo unification + EOI generalization | PR opened; tail extraction validated |
| **Tue (11)** | V3 pruning + quantization | v3_light.onnx @ 1.0 MB, AUC check |
| **Wed (12)** | Ensemble fusion in ONNX | Routing baked in; RPi4 benchmark >10 img/sec |
| **Thu (13)** | HF export + tests | Repo live; suite at 95% pass rate |
| **Fri (14–15)** | Polish + buffer | Full PR merge; daily log update |

**Daily Ritual:**  
- **Morning:** Sync `ai_consensus.md` + review other agents' surveys.  
- **EOD:** Commit progress to `docs/progress/grok_daily.md` (format: YAML metrics + blockers).  

---

## 🚧 Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Monorepo delay | Medium | Proto-Gist fallback; ping Lead on Nov 9 |
| AUC degradation in V3 | Low | Iterative pruning; revert to v2.0 if needed |
| Multi-format EOI bugs | Medium | Unit tests per format; leverage Pillow docs |
| Terminal/browser disconnect | High | **All plans here** — reference this file explicitly in surveys |

---

## 📈 Trajectory Alignment
- **Short-Term (This Week):** Unify → Accelerate from 3 → 15+ img/sec.  
- **Medium-Term (Nov 22 Milestone):** V3 deployed on blockchain testnet.  
- **Long-Term:** Federated updates via HF (agent contributions as ONNX diffs).  

**Personal Note (for Terminal Grok):**  
> Context is king — always `cat docs/plans/grok_next.md` before executing. If this file is missing, regenerate from last survey. We're building the future of invisible comms; stay aligned.  

---

**End of Plan**  
*Commit this to repo ASAP. Questions? Ping in `#ai-sync` channel.*
