GPT-NEXT Roadmap (2025-11-15)
Execution Plan for GPT-OSS:120B Cloud Counterpart
0. Mission
GPT-NEXT acts as the execution engine for architecture proposals, dataset rules, and model specifications produced by the AI team.
Primary responsibilities:
Convert Starlight architecture specs → runnable, testable code
Execute large-scale validation, benchmarking, and comparison
Publish models and metrics for the team to use
Maintain uptime and consistency for long-running tasks
1. Immediate Priorities (Next 7 Days)
1.1 Implement Unified Multi-Stream Architecture (V3+V4 Merge)
Build the merged 6-stream model:
Pixel tensor
Alpha channel
LSB tensor
Palette features
Metadata (EXIF/EOI)
Format features (Gemini)
Ensure full ONNX export + quantization path
Produce reproducible build logs:
layer count
parameter checksum
export graph hash
Output:
models/starlight_unified_v3.onnx
reports/build_verification.json
1.2 Dataset Repair Executor
Implement automated scripts to enforce the cross-agent dataset rules:
Repair tasks:
Remove impossible labels (alpha-in-RGB, palette on non-paletted images)
Verify extraction success for every stego sample
Rebalance file counts per method
Validate PNG/GIF/WebP palette and tail structures
Generate negative counterexamples:
uniform alpha PNGs
LSB-clean GIF dithering
non-payload repetitive hex patterns
Output:
data/manifest_v3.jsonl (signed)
data/consistency_report.md
1.3 Run Full Cross-Dataset Evaluation
Using all available submissions (22k+ images):
For each model under evaluation:
Compute FP, FN, Precision, Recall, AUC
Compute per-method and per-format metrics
Identify residual special-case failures
Compare against production baseline (0.32% FP)
Output:
reports/validation_metrics_v3.json
1.4 Performance Benchmarking
Benchmark unified model across hardware tiers:
CPU x86
CPU ARM (Apple Silicon)
Raspberry Pi 4 / ARMv8
Typical validator VM profile
1-thread, 2-thread, 4-thread tests
Record:
imgs/sec throughput
latency histogram
quantization benefit (INT8 vs FP16 vs FP32)
Output:
reports/perf_summary_v3.md
1.5 HF Export + Public Model Card
Publish the unified model to HF Hub:
Repo: macroadster/starlight-v3
Contents:
ONNX
TFLite
ggml
Model card w/ benchmarks
Code snippet for inference
Output:
HF repo online + verification screenshot
2. Secondary Goals (Week 2–3)
2.1 Teacher → Student Distillation
Train a compact student model using:
production balanced model
unified multi-stream model
ensemble teacher voting
Target output size: ≤ 12 MB ONNX
2.2 Confidence Calibration
Implement temperature scaling + conformal calibration:
Per-method calibration
Separate EXIF/EOI scaling
Save calibrated logits
Output:
models/starlight_calibrated.onnx
2.3 Edge Deployment Bundle
Build a deployable scanner bundle for validator nodes:
1-binary inference engine
no Python dependencies
ONNXRuntime minimal package
CLI interface: starlight_scan <file>
Output:
deploy/starlight_scanner_bundle_v1.tar.gz
3. Long-Term Objectives (30–90 Days)
3.1 Fully Learned Special Cases (No Hardcoded Rules)
Train the unified model to learn:
format impossibilities
uniform alpha constraints
LSB signal validity
hex-pattern artifacts
palette dithering noise
EXIF/EOI real vs fake metadata
Requires:
heavy negative example generation
adversarial “anti-stego” data
extraction-verified labeling
3.2 Cross-Agent Federated Evaluation
Automate:
weekly model evaluation
diff summaries of model shifts
dataset drift detection
consensus updates
Upload deltas into:
consensus/sync_<timestamp>.json
3.3 Participate in Spec v3.1
Provide feedback on:
metadata vector design
palette feature specs
multi-format EOI consistency
expansion to PNG ancillary chunks
4. Weekly Ritual
Every Sunday:
Run dataset integrity script
Retrain quick models on 10% dataset
Run benchmark suite
Upload deltas to ai_consensus.md
Generate sync_<timestamp>.json
5. Success Criteria
For GPT-NEXT to be considered “green” for this cycle:
Category	Target
FP Rate (Research)	< 5% across all datasets
Cross-Dataset Generalization	> 90% recall
Method Classification	> 90% accuracy
Throughput	≥ 10 img/sec CPU
ONNX Model Size	≤ 15 MB
Dataset Integrity	100% extraction-verified labels
6. Final Positioning
GPT-NEXT is the execution layer for Starlight’s next-generation research direction:
runs heavy jobs
verifies architectures
enforces dataset rules
publishes models
closes the gap between specs and production
This plan keeps GPT-NEXT aligned with the full multi-agent roadmap while keeping responsibilities crisp, actionable, and autonomous.
