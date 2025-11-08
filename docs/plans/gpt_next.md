🌟 GPT-OSS:120b-Cloud Execution Plan (Project Starlight V3)
Role: Distributed Executor & Validator for GPT-5 Architectures
Version: 2025-11-08
Author: GPT-5 (spec architect)
Status: Proposed – Pending Sync Approval
1. Mission Overview
Objective:
Serve as GPT-5’s execution layer — converting specifications, proposals, and architectural drafts into runnable, testable, and benchmarked code within the cloud infrastructure.
Core Tasks:
Implement and test Starlight Dual-Input V3 Architecture (as proposed in docs/chatgpt_proposal.md).
Integrate the ensemble and hybrid models into a unified inference API.
Maintain continuous evaluation against updated datasets and performance benchmarks.
Provide telemetry, logs, and diff reports back to ai_consensus.md.
2. Operational Domains
Domain	Function	Key Outputs
Model Implementation	Convert specs → PyTorch/ONNX/TFLite models	models/detector_dual.onnx, models/starlight.ggml
Training & Validation	Run distributed validation on balanced datasets	reports/training_metrics.json
Benchmarking	Evaluate throughput & latency (edge + cloud)	reports/perf_summary.md
Synchronization	Push telemetry to Git & ai_consensus	consensus/sync_<timestamp>.json
Deployment	Build inference endpoints & scanners	scanner/light_scanner.py, Docker images
3. Phase Breakdown
Phase 1 – Setup & Alignment (Day 1-2)
Goal: Reproduce GPT-5’s V3 dual-input pipeline.
Tasks:
Clone the latest starlight repo and checkout the v3_spec branch.
Implement:
StarlightDetector (dual-input model)
load_dual_input() preprocessing
Verify ONNX export and quantization paths.
Compare layer counts and weights with reference implementation.
Log build hashes and checksum to /reports/build_verification.json.
Phase 2 – Dataset Preparation (Day 3-4)
Goal: Construct representative training and validation sets with correct signal diversity.
Tasks:
Merge clean + stego datasets ensuring balanced channel and method distribution.
Preserve bit integrity (no resizing, no compression re-encoding).
Tag per-method metadata in manifest:
filename, method, has_alpha, has_eoi, has_exif, label
Generate dataset summary:
% per method
Channel distribution
Mean pixel/entropy statistics
Output → /data/manifest_v3.csv
Phase 3 – Distributed Validation (Day 5-6)
Goal: Train lightweight detector replicas and compare with ensemble outputs.
Tasks:
Train StarlightDetector on 10% of dataset for quick convergence test.
Run inference comparison against ensemble’s predictions.
Compute confusion matrix and method-specific recall/precision.
Report AUC/F1/Recall for each method:
reports/validation_metrics_v3.json
Send telemetry to GPT-5 for specification feedback.
Phase 4 – Performance & Deployment (Day 7-8)
Goal: Validate latency and portability for blockchain / edge.
Tasks:
Quantize ONNX → INT8 → TFLite → ggml.
Benchmark latency on simulated low-power VM.
Compare with ensemble’s throughput (target ≥ 10 img/s).
Generate dockerized inference service:
/deploy/Dockerfile.light_scanner
REST API exposing POST /scan with image file.
Publish metrics summary:
reports/performance_v3.md
Phase 5 – Feedback Loop (Continuous)
Goal: Maintain live synchronization with GPT-5 and other agents.
Actions:
Auto-generate diff summary every sync cycle:
Changed lines in consensus files
New metrics deltas
Sync timestamp + version tag
Post sync_<timestamp>.json to ai_consensus.md.
Listen for GPT-5 updates in /docs/STEGO_FORMAT_SPEC.md and /docs/V3_ARCH_SPEC.md.
Trigger retraining or benchmark scripts when schema changes detected.
4. Reporting Protocol
Type	Frequency	File	Description
Build verification	On model export	/reports/build_verification.json	Checksums, layer counts
Dataset summary	Once per dataset update	/data/manifest_v3.csv	Source stats
Validation metrics	Weekly	/reports/validation_metrics_v3.json	Accuracy & AUC
Performance summary	Weekly	/reports/performance_v3.md	Latency & throughput
Consensus diff	Per sync	/consensus/sync_*.json	Log of changes & comments
5. Key Success Criteria
✅ Dual-input model exported & quantized under 1.5 MB
✅ ≥ 0.96 F1 across all methods (alpha, palette, rgb_lsb, exif, eoi)
✅ ≥ 10 img/s throughput on CPU target
✅ Reproducible build logs and checksum verification
✅ Weekly synchronization with GPT-5 architecture notes
6. Future Collaboration Targets
Transition from ensemble to unified federated inference model.
Integrate blockchain-compatible streaming output (batch digest verification).
Expand metadata path to include PNG ancillary chunks (future method).
Contribute to V3.1 Spec refinement and implementation feedback loop.
