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

## 3. Training the Generalized Model

Train the high-performance V3 generalized model with Attention mechanisms and automatic bit-order prediction.

* **Action:** Execute the training script with recommended parameters.
* **Command:**
    ```bash
    python3 trainer.py --epochs 20 --batch_size 16 --out models/detector_balanced.pth
    ```
* **V3 Features:**
  - **Bit-Order Prediction**: Learns to distinguish between LSB-first and MSB-first bitstreams.
  - **Attention Mechanisms**: Specialized layers for subtle EXIF and EOI detection.
  - **Balanced Sampling**: Automatically balances clean and stego classes across all sub-datasets.

---

## 4. Running Steganography Detection & Auto-Extraction

Use `scanner.py` to detect steganography and automatically recover hidden messages.

* **Action:** Run the scanner on a file or directory.
* **Command:**
    ```bash
    # Scan single file (Detects AND Extracts automatically)
    python3 scanner.py /path/to/image.png
    
    # Scan directory (Optimized for speed, detections only)
    python3 scanner.py /path/to/images/ --json
    ```
* **Advanced Extraction Features:**
  - **Model-Guided Recovery**: Uses the model's predicted bit-order (LSB/MSB) to extract messages instantly.
  - **Auto-Technique Selection**: Automatically routes to Alpha, LSB, EXIF, EOI, or Palette extractors.
  - **High Performance**: Achieving 98.9%+ detection rate with <0.5% false positives.


---

## 5. Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure ONNX files exist in each submission's `model/` directory
2. **Method compatibility**: Check `method_config.json` for supported steganography methods
3. **Import errors**: Check `requirements.txt` in each submission directory
4. **Low accuracy**: Verify training data quality and model card metrics
