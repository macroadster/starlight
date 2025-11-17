# USAGE: Building and Running Project Starlight

These instructions detail the complete workflow for generating training data, training models, creating ensembles, and running steganography detection. All commands should be executed from the project's **root directory** unless otherwise specified.

---

## 1. Generating Training Datasets

The protocol relies on community-contributed, diverse datasets. Generate data from the project root directory.

* **Action:** Run the data generation script from the project root. The `--limit` flag controls the number of files generated (e.g., 1000).
* **Command:**
    ```bash
    python3 data_generator.py --limit 10
    ```

---

## 2. Data Integrity Verification

Run the diagnostic script from the project root to ensure that all contributed datasets are properly structured, labeled, and ready to be used for model training.

* **Action:** Execute the validation script from the project's root directory.
* **Command:**
    ```bash
    python3 diag.py
    ```
* **Expected Output:** The script will confirm that file counts, labeling, and data format across all contributor subdirectories meet the protocol standards.

---

## 3. Training Individual Models

Train individual steganalysis models for each contributor submission.

* **Action:** Execute the training script from the project root directory.
* **Command:**
    ```bash
    python3 trainer.py
    ```
* **Output:** Trained model files (e.g., `detector.onnx`) will be saved to the contributor's `model/` directories.

---

## 4. Running Steganography Detection

Use scanner.py to scan for steganography.

* **Action:** Run scanner with default ensemble mode.
* **Command:**
    ```bash
    # Scan single file (with extraction by default)
    python3 scanner.py /path/to/image.png
    
    # Scan directory (quick mode by default)
    python3 scanner.py /path/to/images/ --json
    ```
* **Features:**
  - **Method-specialized voting**: Only models supporting detected method vote
  - **Smart defaults**: Single files extract messages, directories use quick scan
  - **Parallel processing**: Multi-threaded scanning with `--workers` option
  - **Cached ensemble**: One-time model loading for performance
  - **Specialist bonuses**: Method specialists get higher voting weight
  - **All steganography methods**: Alpha, LSB, EXIF, EOI, Palette support

---

## 5. Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure ONNX files exist in each submission's `model/` directory
2. **Method compatibility**: Check `method_config.json` for supported steganography methods
3. **Import errors**: Check `requirements.txt` in each submission directory
4. **Low accuracy**: Verify training data quality and model card metrics
