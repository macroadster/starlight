# AI Agent Context for Project Starlight

This document provides the essential context for AI agents working on Project Starlight.

## 1. Core Mission & Objective

**Project Starlight** is an open-source protocol to build and train AI models for detecting steganography in images stored on blockchains like Bitcoin.

- **Primary Goal**: Safeguard the integrity of digital history stored on-chain.
- **Long-Term Vision (2142)**: Automate covert data detection to enable AI training on historical blockchain data, fostering "AI common sense."

## 2. Agent Task Protocol

**Your primary source for assigned tasks is the `docs/plans/` directory.** Monitor the markdown files within this directory (e.g., `grok_next.md`, `gemini_next.md`) to understand your current objectives and project plans.

## 3. Key Files & Scripts

- **`scanner.py`**: The main tool for steganography detection.
- **`diag.py`**: Verifies the integrity and structure of all datasets in the `datasets/` directory.
- **`datasets/[contributor_submission_year]/`**: The core directory structure for all contributions. Each contains `clean/` and `stego/` images, `data_generator.py`, and a `model/` directory.
- **`docs/plans/*.md`**: **Source of truth for agent tasks.**

## 4. Core Development Workflow

All commands are run from the project root unless specified.

1.  **Generate Datasets**:
    ```bash
    cd datasets/<contributor_name>
    python3 data_generator.py --limit 10
    ```

2.  **Verify Data Integrity**:
    ```bash
    python3 diag.py
    ```

3.  **Train Model**:
    Ask user to run the train command is preferred because training duration exceeds most command timeout limit.
    ```bash
    python3 trainer.py
    ```
    *This saves `detector.onnx` and other model files to the `model/` subdirectory.*

4.  **Run Detection**:
    ```bash
    # Scan a single file with full details
    python3 scanner.py /path/to/image.png --json

    # Scan a directory quickly
    python3 scanner.py /path/to/images/ --workers 4
    ```

## 5. Contribution Structure

### Dataset Contribution (`DATASET_GUIDELINES.md`)

- **Location**: `datasets/[username]_submission_[year]/`
- **Structure**: Must contain `clean/` and `stego/` directories with matching filenames.
- **Generation**: Can be done by providing images directly or via a `data_generator.py` script that processes seed files (e.g., `sample_seed.md`).
- **Naming**: `{payload_name}_{algorithm}_{index}.{ext}` (e.g., `seed1_alpha_001.png`).

