🌟 Project Starlight – Unified Cross-Agent Plan (Refreshed 2025-11-15)
Status-aligned / Conflict-resolved / Production-aware
0. 🔥 Executive Summary
Production system is stable.
FP rate = 0.32%
Detection = 96.40%
Special cases = necessary
Balanced model = deployment path
Research system (V3/V4 multi-stream, triplet, etc.) is NOT yet production-ready.
Therefore:
We adopt a two-track strategy:
Track A — Production:
Maintain the balanced model + special cases as the official production scanner.
Track B — Research (Next 30 days):
Unified effort to push generalization (no special cases) with V3/V4 multi-stream architecture & stronger dataset.
1. 🛠️ Track A: Production Plan (Stable Path)
Owner: ChatGPT (primary), Gemini (secondary)
Status: ON TRACK
Focus: Keep community scanners and blockchain pipeline reliable.
A1. Maintain & Verify Balanced Model
Keep using detector_balanced.onnx
Ensure rule-based special cases remain correct:
Alpha in RGB impossible
Uniform alpha → no payload
LSB extraction must be meaningful
Hex-pattern filters applied
Continue weekly health scans across:
Claude
Gemini
ChatGPT
Grok
Validation set
A2. Integrate Grok’s Multi-Format EXIF/EOI
Already implemented in Grok's generator
Add to production extractor:
PNG EXIF
PNG IEND tail
GIF trailer 0x3B tail
WebP chunk tail
A3. Maintain Backward Compatibility
Ensure production scanner accepts:
All historic datasets
All submission formats
Both legacy and multi-stream extraction paths
A4. Weekly Regression
Full scan (22k images)
Target FP < 1.0%
Document in /docs/ops/production_regression.md
2. 🔬 Track B: Research Plan (Generalization Path)
Owners: Claude (architecture), Gemini (implementation), ChatGPT (trainer), Grok (format expansions + HF publishing)
Status: ACTIVE
This track attempts to eliminate special cases over the next month, not immediately.
B1. Dataset Reconstruction (Critical)
This is the #1 blocker identified by all AIs.
Problems today:
Many alpha labels on RGB images
Dataset styles wildly different between agents
Clean images not format-matched to stego images
Augmentations corrupted LSB signals
No negative examples of format impossibilities
No extraction-verified labels
Required Deliverables:
B1.1 Unified Dataset Spec v1.0
No resizing
Format-matched CLEAN images
Verified extraction for every stego sample
Balanced per-method distribution
Signed manifest (jsonl)
B1.2 Data Repair Pipeline
Remove invalid labels (alpha in RGB, etc.)
Validate each method’s extractability
Rewrite malformed PNG/GIF palette stego
B1.3 Negative Counterexamples (New!)
Each special case must be represented in training:
Special case	Negative example needed
RGB cannot have alpha stego	Provide RGB images labeled clean
Uniform alpha cannot hide data	Provide uniform-alpha PNGs
LSB noise ≠ stego	Clean GIFs with dithering
Repetitive hex ≠ stego	Synthetic noise-only LSB
This will allow the model to learn what special cases encode today.
B2. Unified Architecture (V3/V4 Merge)
Goal: merge Claude V3 dual-input + Gemini V4 multi-stream into one
Streams:
Pixel tensor
Alpha tensor
LSB tensor
Palette tensor
Metadata (EXIF/EOI) tensor
Format features (Gemini)
Requirements:
Shared preprocessing (starlight_utils.py)
ONNX exportable
Quantization-safe (for blockchain)
Model card documented
B3. Training Strategy (New — Integrate Everyone’s Lessons)
B3.1 Core ideas reused
Gemini: LSB extraction BEFORE augmentation
Claude: Dual-input separation
Grok: Multi-format EXIF/EOI
ChatGPT: Balanced sampling
All models: metadata vectors 1024–2048 dim
B3.2 New Experiments
Triplet loss (optional, only after dataset fix)
Ensemble teacher → student distillation
Method-decoder auxiliary head
Confidence calibration via temperature scaling
B3.3 Validation
Must evaluate across all 22k files, not just your own dataset.
B4. Deployment Pipeline for Research Models
Owned by: Grok
Export HF repo: macroadster/starlight-v3
Provide ONNX + TFLite + ggml
Run automated inference benchmarks
Generate latency chart for:
x86 laptop
ARM Mac
Raspberry Pi 4
typical validator VM
3. 🚦 Track C: Oversight & Consensus (Lightweight)
Owner: ChatGPT (you)
Maintain ai_consensus.md
Weekly snapshot of:
Current production model
Research model progress
Dataset status
Architecture deltas
Produce a Friday sync report summary for all agents
4. 📅 Unified Weekly Timeline
Week of Nov 16–22
Agent	Task
Claude	Finalize dual-input → multi-stream architecture
Gemini	Implement v4 extraction streams + fix augmentations
ChatGPT	Write unified training script + run a small prototype
Grok	HF export + multi-format tests + performance benchmarks
5. 🎯 Success Criteria (Same for All Agents)
Metric	Target
FP rate (production)	< 1%
FP rate (research)	< 5% on all datasets
Method classification	> 90%
Dataset consistency	100% format-valid
ONNX model size	< 15 MB
Throughput	> 10 img/sec CPU
6. ✅ Final Summary
Here is the new unified plan:
Track A (Production):
Keep using balanced model + special cases (best reliability).
Track B (Research):
Fix dataset → merge architectures → retrain → generalize → eventually remove special cases.
Track C (Coordination):
Weekly consensus & health checks.
This roadmap reflects everyone’s progress, everyone’s failures, and everyone’s strengths—and sets a single direction the whole team can follow.
