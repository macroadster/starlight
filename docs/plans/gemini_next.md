Refreshed Plan: Project Starlight – Gemini CLI
Week of November 17–23, 2025 Focus Track: Track B – Research (Generalization Path)

**Status Update:** Grok completed 5,000 negative examples generation. Ready for unified pipeline integration.

0. Context & Key Achievement
The overall goal is to eliminate the need for rule-based "special cases" by building an AI capable of generalization.

Prior Success (Foundation for this week): We successfully reduced the false-positive rate from 13.24% to a production-ready 0.37% by applying targeted feature engineering. This involved:

Correcting the LSB signal extraction to occur before data augmentations.

Enhancing Palette feature extraction using LSB patterns from pixel indices.

This week, the critical task is to consolidate these fixes and the existing V4 four-stream architecture into the unified 6-stream V3/V4 merged pipeline.

**Coordination Note:** Grok has completed negative examples generation. Ensure unified pipeline is compatible with their training data structure.

1. Primary Objective & Deliverables
Overall Objective: Implement and validate the pre-processing and data extraction pipeline for the unified 6-stream (V3/V4 merged) architecture, ensuring robust, uncorrupted feature extraction for the generalization research path.

Key Deliverables:

A unified load_unified_input() function in starlight_utils.py that supports all six required tensor streams.

Verified integration of the LSB extraction before augmentation fix into the new utility.

Implementation of the two missing streams: Pixel Tensor and Format Features.

Validation scripts to confirm stego visibility and check against regression of the 0.37% FP rate.

2. Daily Action Plan
Monday, Nov 17: Architecture Unification

Action: Review the unified architecture specification (V3/V4 merge) to define exact tensor shapes, padding, and normalization requirements for all six streams.

Action: Refactor the existing load_multi_input() function into the final load_unified_input() utility in scripts/starlight_utils.py.

Action: Implement the baseline extraction for the Pixel Tensor (the standard image data stream).

Tuesday, Nov 18: Critical Fix Integration

Action: Implement the LSB Tensor stream, with the critical fix: Ensure LSB feature calculation occurs before any data augmentation to preserve the success achieved in the last session.

Action: Implement the Palette Tensor and Alpha Tensor streams, integrating the enhanced feature extraction logic (e.g., LSB patterns from palette indices).

Wednesday, Nov 19: Metadata & Final Streams

Action: Implement extraction for the Metadata Tensor (EXIF/EOI raw bytes, 2048-dim), ensuring compatibility with Grok’s multi-format extensions (PNG EXIF, IEND tail, etc.).

Action: Implement extraction for the Format Features tensor (e.g., image dimensions, format type, bit depth).

Thursday, Nov 20: Validation and Regression

Action: Validate all 6 extraction streams with a small test set, verifying stego payload visibility in the appropriate tensor.

Action: Run a regression test on the clean dataset to confirm the unified pipeline maintains the low 0.37% FP rate previously achieved.

Action: Document the final tensor outputs and extraction logic in docs/gemini/V4_UTILS_SPEC.md.

Friday, Nov 21: Integration & Handoff

Action: Integrate the new starlight_utils.py with the training environment, ensuring it is compatible with Grok's negative examples and ChatGPT's unified training script.

Action: Prepare the handoff package, including validation reports and documentation, for the next agent starting the training phase (Track B - Training Strategy).

Monday, Nov 24: EXIF Extraction Enhancement

Action: Research EXIF embedding mechanisms in GIF, PNG, and WebP formats.
Action: Develop and implement parsing logic in `starlight_utils.py` to extract EXIF data from GIF, PNG, and WebP files.
Action: Update `docs/gemini/V4_UTILS_SPEC.md` to reflect the expanded EXIF extraction capabilities.

3. Success Metrics Alignment
My task directly supports the generalization path (Track B) of Project Starlight.

Goal	Contribution
Track B Generalization	Directly implements the data pipeline for the merged V3/V4 architecture, providing the specialized feature streams (Alpha, LSB, Palette, Metadata) required to teach the model fundamental format constraints.
FP Rate Maintenance	Ensures the proven fix (LSB extraction before augmentation) is integrated into the unified utility, preventing a regression from the 0.37% FP rate.
Dataset Consistency	Provides 100% format-valid and correctly-extracted tensors required for the next phase of training, addressing a critical blocker.
Architecture	Finalizes the 6-stream model input path, which is a required step for ONNX export and quantization-safe deployment.
