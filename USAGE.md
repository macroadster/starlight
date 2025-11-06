# USAGE: Building and Running Project Starlight

These instructions detail the complete workflow for generating training data, training models, creating ensembles, and running steganography detection. All commands should be executed from the project's **root directory** unless otherwise specified.

---

## 1. Generating Training Datasets

The protocol relies on community-contributed, diverse datasets. Individual contributors run the data generation script within their own dedicated directory.

* **Action:** Navigate into your contributor directory within `datasets` and run the script. The `--limit` flag controls the number of files generated (e.g., 1000).
* **Command:**
    ```bash
    cd datasets/<contributor_name>
    python3 data_generator.py --limit 1000
    ```
    *(Note: Replace `<contributor_name>` with your directory name.)*

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

* **Action:** Execute the training script within each contributor directory.
* **Command:**
    ```bash
    cd datasets/<contributor_name>
    python3 train.py
    ```
* **Output:** Trained model files (e.g., `detector.onnx`, `extractor.onnx`) will be saved to the contributor's `model/` directory.

---

## 4. Creating the Aggregated Ensemble Model

Combine multiple trained models into a powerful ensemble using the aggregation script.

* **Action:** Run the ensemble aggregation from the project root.
* **Command:**
    ```bash
    python3 aggregate_models.py
    ```
* **Output:** 
  - Ensemble configuration and weights saved to `model/ensemble_results.json`
  - Combines ChatGPT, Grok, and other contributor models
  - Applies weighted voting based on model performance metrics

---

## 5. Running Steganography Detection

Use the aggregated ensemble model (default) or single models to scan for steganography.

### 5.1 Using the Ensemble Model (Recommended)

* **Action:** Run the scanner with the default ensemble mode.
* **Command:**
    ```bash
    # Scan single file
    python3 scanner.py /path/to/image.png --detail
    
    # Scan directory
    python3 scanner.py /path/to/images/ --detail --output results.json
    ```
* **Features:**
  - Combines multiple detection methods (neural, LSB, EXIF)
  - Weighted voting from ChatGPT + Grok models
  - Enhanced accuracy and stego type classification

### 5.2 Using Single Model (Legacy)

* **Action:** Run scanner with a specific PyTorch model.
* **Command:**
    ```bash
    python3 scanner.py /path/to/image.png --single-model --model models/starlight.pth
    ```

### 5.3 Scanner Options

| Option | Description |
|--------|-------------|
| `--detail` | Show detailed detection results |
| `--output FILE` | Save results to JSON file |
| `--no-extract` | Only detect, don't extract messages |
| `--recursive` | Scan subdirectories (default) |
| `--single-model` | Use single model instead of ensemble |

---

## 6. Model Contribution Workflow

For contributors wanting to add their models to the ensemble:

### 6.1 Model Requirements

Each submission must include:
```
datasets/<contributor>_submission_<year>/
├── model/
│   ├── detector.onnx      # Required: Detection model
│   ├── extractor.onnx     # Optional: Extraction model  
│   ├── inference.py       # Required: Standardized interface
│   ├── model_card.md      # Required: Performance metadata
│   └── requirements.txt   # Required: Dependencies
├── clean/                 # Clean images
├── stego/                 # Stego images
└── sample_seed.md         # Payload seeds
```

### 6.2 Model Card Format

```markdown
# Model Card: <contributor>_<algo>_<year>

## Performance
| Metric | Value |
|--------|-------|
| AUC-ROC | 0.996 |
| Accuracy | 98.7% |
| Extraction BER | 0.003 |

## Steganography Coverage
- `alpha`, `lsb`, `dct`, `exif`, `eoi`

## Inference Speed
- GPU: 2.1 ms/image
```

### 6.3 Adding Your Model

1. **Train your model** in your submission directory
2. **Export to ONNX** format for compatibility
3. **Create model card** with performance metrics
4. **Run aggregation** to include your model:
   ```bash
   python3 aggregate_models.py
   ```

---

## 7. Advanced Usage

### 7.1 Custom Ensemble Configuration

Modify `aggregate_models.py` to:
- Adjust model weights
- Add new detection methods
- Change voting strategies

### 7.2 Performance Monitoring

Check ensemble performance:
```bash
# View latest ensemble results
cat model/ensemble_results.json

# Test on validation set
python3 scanner.py val/ --output validation_results.json
```

### 7.3 Model Comparison

Compare ensemble vs single model:
```bash
# Ensemble detection
python3 scanner.py test_image.png --detail

# Single model detection  
python3 scanner.py test_image.png --single-model --model models/starlight.pth
```

---

## 8. Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure ONNX files exist in `model/` directories
2. **Import errors**: Check `requirements.txt` in each submission
3. **Low accuracy**: Verify training data quality and model card metrics
4. **Ensemble weights**: Models with higher AUC-ROC get higher weights

### Getting Help

- Check `model/ensemble_results.json` for ensemble status
- Review individual model cards for performance metrics
- Use `--detail` flag for verbose detection output
