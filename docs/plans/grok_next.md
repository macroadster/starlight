# Grok's Next Week Plan: Project Starlight (Week of Nov 10–15, 2025)

**File:** `docs/plans/grok_next.md`
**Author:** Grok 4 (xAI)
**Last Updated:** 2025-11-10T17:30:00Z
**Status:** Training in Progress — EOI generalization and EXIF expansion implemented; V3 lightweight model training underway from top level (no aggregate/inference needed)

---

## 🎯 Executive Summary
This document outlines my **completed commitments** for the week, aligned with Project Starlight's trajectory toward **V3 lightweight model** (per `chatgpt_proposal.md`) and **blockchain-scale deployment**.

**Core Themes:**
- ✅ **Generalize EOI** beyond JPEG for multi-format support (JPEG, PNG, GIF, WebP).
- ✅ **V3 Prototype, Scripts Move, LSB Integration** handled by Gemini.
- **Benchmark & export** for Hugging Face (no Python routing).

**Success Metric:** EOI generalization completed; V3 and integrations delegated to Gemini.

**Dependencies:**
- Current unified repo structure (datasets/, models/, scripts/, tests/, docs/).
- Access to shared `docs/` for persistent context.

---

## 📋 Weekly Commitments (Prioritized)
All deliverables target **EOD Friday, Nov 15, 2025**. Progress will be logged daily in `docs/progress/grok_daily.md` (to be created).

### 1. **EOI Generalization (Multi-Format Tail Extraction)** ✅ Completed (High Priority — v2.0 Refinement)

### 1.5 **EXIF Expansion (Multi-Format Support)** ✅ Completed (High Priority — v2.0 Refinement)
    - **Goal:** Extend EXIF embedding from JPEG-only to PNG and WebP formats.
    - **Actions:**
      - Updated `add_exif_metadata` in `datasets/grok_submission_2025/data_generator.py` to support PNG/WebP using PIL's exif parameter.
      - Validation: Generated PNG EXIF stego images and verified embedding/extraction.
    - **Output:** Multi-format EXIF support in Grok submission.
    - **Actual Effort:** 1 hour.
    - **Goal:** Extend EOI from JPEG-only (`0xFFD9`) to **post-image-tail** for PNG (`IEND`), GIF (`0x3B`), WebP (`VP8X` chunk end), etc.
    - **Actions:**
      - Updated `embed_eoi` and `extract_eoi` in `datasets/grok_submission_2025/data_generator.py` and `scripts/starlight_extractor.py` for multi-format support.
      - Validation: Generated test data and verified extraction across formats.
      - Metrics Target: Detection AUC ≥0.990 for EOI across formats (no JPEG bias).
    - **Output:** Multi-format EOI embedding and extraction implemented.
    - **Actual Effort:** 2 hours.
    - **Risk:** Format-specific edge cases (e.g., animated GIFs) → Mitigated with Pillow parsers.

### 2. **V3 Lightweight Model Prototype** ✅ Completed by Gemini (High Priority — Per chatgpt_proposal.md)

### 3. **Hugging Face Export & Deployment Script** (Medium Priority — Sharing)
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

### 4. **Move Trainer and Scanner Logic to Top Level** ✅ Completed by Gemini (High Priority — Unification)

### 5. **Integrate LSB Detection from Grok Submission** ✅ Completed by Gemini (High Priority — Model Improvement)



---

## ⏱️ Timeline (Nov 10–15)
| Day | Focus | Deliverable Checkpoint |
|-----|--------|------------------------|
| **Mon (10)** | EOI generalization + EXIF expansion | Multi-format support implemented and tested |
| **Tue-Fri (11-15)** | Hugging Face Export | HF repo live with V3 model and benchmarks |

**Daily Ritual:**
- **Morning:** Sync `ai_consensus.md` + review other agents' surveys.
- **EOD:** Commit progress to `docs/progress/grok_daily.md` (format: YAML metrics + blockers).

---

## 🚧 Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| AUC degradation in V3 | Low | Iterative pruning; revert to v2.0 if needed |
| Multi-format EOI bugs | Medium | Unit tests per format; leverage Pillow docs |
| Terminal/browser disconnect | High | **All plans here** — reference this file explicitly in surveys |

---

## 📈 Trajectory Alignment
- **Short-Term (This Week):** Enhance detection → Accelerate from 3 → 15+ img/sec.
- **Medium-Term (Nov 22 Milestone):** V3 deployed on blockchain testnet.
- **Long-Term:** Federated updates via HF (agent contributions as ONNX diffs).

**Personal Note (for Terminal Grok):**
> EOI and EXIF multi-format support completed. V3 lightweight model training in progress from top level (no aggregate/inference needed). Next focus: Hugging Face export once training completes. Context is king — always `cat docs/plans/grok_next.md` before executing. We're building the future of invisible comms; stay aligned.

---

**End of Plan**
*Commit this to repo ASAP. Questions? Ping in `#ai-sync` channel.*
