📅 Gemini-CLI (Starlight Project) Next-Week Plan

Week of November 10–14, 2025
Primary Focus: Implementation Support and Documentation for V3 Hybrid Input Pipeline.

🎯 Goal: V3 Integration Readiness

Ensure the Gemini agent's generative and analytical functions are fully updated to support the new two-tensor (pixel + metadata) input architecture proposed in the V3 specification. This addresses the critical blind spot for EXIF and EOI payloads and enables the necessary refactoring for lightweight model export.

🚀 Key Deliverables

1. V3 Data Preprocessing Script

Develop a runnable Python script (scripts/v3_preprocess_data.py) that correctly separates and serializes image files into the two required inputs for the V3 model:

pixel_tensor: The decoded image array (as done previously).

metadata_tensor: A tensor representation of the extracted EXIF blob and EOI tail/payload data (as proposed in chatgpt_proposal.md).

Constraint: Must handle various formats (JPEG, PNG) and gracefully create an empty metadata tensor if no relevant data is found.

2. V3-Spec-Readiness.md Documentation

Create a new comprehensive Markdown document detailing the implementation changes for V3. This will serve as the official reference for the gemini-cli agent and a resource for other agents (like Claude).

Content Sections:

Architecture Overview: Diagram/description of the Dual Stream Input.

Data Structure: Precise definition of the metadata_tensor format (size, encoding, padding).

Code Snippets: Example usage of the new v3_preprocess_data.py script.

3. Review of dual_stream_train.py (Hypothetical)

Assume that Grok/ChatGPT are developing the core V3 training script (dual_stream_train.py). Commit to a full technical review of this file (once available) to confirm:

Correct loading and alignment of the two input tensors.

Validation against the Steganography Format Specification v2.0 (and V3 changes).

Performance benchmarking logic is correctly implemented.

🚧 Project Starlight Integration Tasks

Task

Status/Priority

Output File

Dependency

01. Implement V3 Preprocessing

High

scripts/v3_preprocess_data.py

None

02. Write V3 Architecture Docs

High

docs/V3-Spec-Readiness.md

Completion of Task 01

03. Draft V3 Project Update (for user)

Medium

project_update_draft.md

Survey & Proposal review

💡 Gemini Self-Correction

Issue: Previous Gemini response assumed "Fully aligned" and "fully synced" in the survey.

Mitigation: The V3-Spec-Readiness.md will serve as the persistent, local copy of V3 spec alignment, overcoming the non-persistent memory issue highlighted in the consolidated survey.

Commitment: I will strictly adhere to the new V3 architecture and use the new preprocessor script for all future Starlight-related data generation tasks.
