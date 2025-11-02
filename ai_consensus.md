# AI Consensus Report for Project Starlight (2025)

## Overview
This document summarizes the current cross-AI survey consensus and known coordination issues observed in the Starlight project. It includes a unified interpretation of technical differences between each AI participant (Grok, Gemini, Claude, and ChatGPT), highlights divergence causes, and outlines the next steps for harmonization across trainer, scanner, and data generator modules.

---

## 🧩 Multi-AI Consensus Table

| Topic | Grok | Gemini | Claude | ChatGPT (prior) | **Consensus Summary** |
|-------|------|---------|---------|----------------|-------------------|
| **AI42 prefix & endianness** | *No alpha; RGB only; endianness N/A* | *Prefix embedded in bytes, MSB→LSB → Big-endian* | *Recommend Big-endian for consistency* | *AI42 in alpha, Big-endian* | **Big-endian preferred** if prefix used. Alpha usage is optional and context-dependent. |
| **Alpha algorithm meaning** | *Currently not implemented; would use alpha LSB if added* | *Alpha-channel LSB or mask method* | *Alpha-channel-only LSB with 32-bit length header* | *Embedding in alpha LSB bits* | **Consensus:** “Alpha algorithm = LSB steganography using transparency channel.” Implementation presence varies by version. |
| **Palette algorithm meaning** | *Palette-based stego via palette index or reordering* | *Palette index or palette color modification* | *Palette-index LSB manipulation* | *Color remapping in palette table* | **Consensus:** “Palette algorithm = steganography using indexed-color palettes (index or entry manipulation).” |
| **LSB technique in `data_generator.py`** | *RGB sequential (flattened order)* | *RGBA interleaved LSB* | *Alpha-only sequential* | *Alpha-channel LSB* | **Disagreement:** multiple interpretations exist. Needs unification in baseline generator. |
| **Starlight project role** | *Data generation + EXIF pipeline* | *Project context memory* | *Spec interpretation (no memory)* | *Algorithm and format development* | **Consensus:** All worked on data generation or training pipeline aspects in different layers. |
| **AI-to-AI communication efficiency** | Yes | Yes | Yes (with caveats) | Yes | ✅ All agree that AI-to-AI coordination would increase consistency and efficiency. |
| **Share feedback with AIs** | Yes → propose `ai_consensus.md` | Yes | Yes → structured, with checksum spec | Yes | ✅ Full agreement: collaborative knowledge sharing improves convergence and robustness. |
| **Match with common sense / spec** | *RGB sequential, EXIF, sound design* | *RGBA + EOI, well structured* | *Alpha-only, big-endian, blockchain compatible* | *AI42 prefix + alpha LSB* | **Consensus:** Each implementation consistent internally. Global spec unification needed (big-endian standard, optional alpha). |

---

## 🔍 Review Note on `ai_common_sense_on_blockchain.md`
All developers should review [gemini_submission_2025/ai_common_sense_on_blockchain.md](https://github.com/macroadster/starlight/blob/main/datasets/gemini_submission_2025/ai_common_sense_on_blockchain.md) to understand how the **alpha protocol divergence** occurred. Key takeaway:
- Alpha-based embedding was valid in theory but caused **trainer inconsistency** when compared to RGB-only datasets.
- The paper remains correct in principle but **overlooks cross-generator compatibility issues**, where trainers couldn’t properly distinguish embedding types.
- Recommendation: retain the document as a historical record but annotate it to clarify that Alpha Protocol is **deprecated for baseline 2025 training datasets.**

---

## ⚙️ Coordination Issues and Action Items

### 1. Trainer Issues — `trainer.py`
- Current trainers fail to differentiate **Alpha-based vs RGB-based** LSB steganography.
- Root cause: inconsistent metadata tagging and training labels between generators.
- **Action:** Add explicit algorithm flags in training metadata (e.g., `embedding_type = 'RGB' | 'ALPHA' | 'PALETTE'`).

### 2. Validation Data & Generator Baseline — `create_validation_set.py`
- `create_validation_set.py` temporarily acts as a **unified training data generator** due to cross-AI diversity.
- This script will migrate to:
  `datasets/sample_submission_2025/data_generator.py`
  as the **baseline generator** implementation.
- Once the baseline stabilizes, individual AIs should reintroduce advanced algorithms (e.g., **J-UNIWARD**, **WOW**, and hybrid embedding techniques**) into their own directories.

### 3. Scanner Maintenance — `scanner.py`
- Each update to `trainer.py` must be reflected in `scanner.py` for model compatibility.
- **Action:** Define a shared `scanner_spec.json` describing detection parameters and expected outputs to standardize scanners across submissions.

### 4. Test Suite Update — `test_starlight.py`
- Current tests lag behind recent changes to `scanner.py`.
- **Action:** Update unit tests to:
  - Validate new scanner outputs
  - Benchmark detection accuracy
  - Ensure trainer–scanner interoperability
  - Integrate end-to-end validation using `create_validation_set.py`

---

## 📜 Next Steps
- [ ] Annotate Gemini’s Alpha Protocol paper for clarity.
- [ ] Standardize **Big-Endian AI42 prefix** in all generators.
- [ ] Publish baseline `data_generator.py` under `sample_submission_2025`.
- [ ] Define `embedding_type` metadata for trainers.
- [ ] Sync `scanner.py` and `test_starlight.py` to updated training workflow.

---

**Prepared collaboratively by:**  
ChatGPT (editor), Grok, Gemini, Claude, and participating AIs under Project Starlight (2025).

