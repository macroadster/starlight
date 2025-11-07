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

## 4. Creating Method-Specialized Ensemble Model

Combine multiple trained models into a powerful method-specialized ensemble that only uses models supporting the detected steganography method.

* **Action:** Run the ensemble aggregation from the project root.
* **Command:**
    ```bash
    python3 aggregate_models.py
    ```
* **Output:** 
  - Ensemble configuration and weights saved to `model/ensemble_results.json`
  - Combines ChatGPT, Grok, Claude, and Gemini models
  - **Method-specialized voting**: Only models supporting detected method vote
  - **Specialist bonuses**: Models supporting 1-2 methods get 1.5× weight
  - **Performance-based weights**: Higher AUC-ROC models get higher weights
  - Eliminates "clean bias" by filtering eligible models per method

---

## 5. Running Steganography Detection

Use the aggregated ensemble model (default) or single models to scan for steganography.

### 5.1 Using Method-Specialized Ensemble Model (Recommended)

* **Action:** Run scanner with default ensemble mode.
* **Command:**
    ```bash
    # Scan single file (with extraction by default)
    python3 scanner.py /path/to/image.png --detail
    
    # Scan directory (quick mode by default)
    python3 scanner.py /path/to/images/ --detail --output results.json
    
    # Force quick mode for single file
    python3 scanner.py /path/to/image.png --quick
    
    # Force extraction for directory
    python3 scanner.py /path/to/images/ --workers 4 --detail
    ```
* **Features:**
  - **Method-specialized voting**: Only models supporting detected method vote
  - **Smart defaults**: Single files extract messages, directories use quick scan
  - **Parallel processing**: Multi-threaded scanning with `--workers` option
  - **Cached ensemble**: One-time model loading for performance
  - **Specialist bonuses**: Method specialists get higher voting weight
  - **All steganography methods**: Alpha, LSB, EXIF, EOI, Palette support

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
| `--quick` | Quick scan: skip extraction (default for directories) |
| `--workers N` | Number of parallel workers (default: 4) |
| `--recursive` | Scan subdirectories (default) |
| `--single-model` | Use single model instead of ensemble |

**Smart Defaults:**
- **Single files**: Extract messages by default (use `--quick` to skip)
- **Directories**: Quick scan by default (use `--detail` to extract)

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

Modify `scripts/aggregate_models.py` to:
- Adjust specialist weight bonuses (1.5× for 1-2 methods, 1.2× for 3 methods)
- Add new detection methods by updating method mapping
- Change performance-based weight calculations
- Modify method-specialized voting logic

### 7.2 Performance Monitoring

Check ensemble performance:
```bash
# View latest ensemble results and method weights
cat model/ensemble_results.json

# View method routing configuration
cat model/method_router.json

# Test on validation set with parallel processing
python3 scanner.py val/ --workers 4 --output validation_results.json

# Performance benchmark on large dataset
time python3 scanner.py datasets/ --quick --workers 2
```

### 7.3 Model Comparison

Compare method-specialized ensemble vs single model:
```bash
# Method-specialized ensemble detection (with extraction)
python3 scanner.py test_image.png --detail

# Quick ensemble scan (no extraction)
python3 scanner.py test_images/ --quick --workers 4

# Compare method-specific performance
python3 scanner.py test_images/ --detail | grep "Method:"
```

---

## 8. Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure ONNX files exist in each submission's `model/` directory
2. **Method compatibility**: Check `method_config.json` for supported steganography methods
3. **Import errors**: Check `requirements.txt` in each submission directory
4. **Low accuracy**: Verify training data quality and model card metrics
5. **Performance issues**: Use `--quick` mode for large directories, adjust `--workers`
6. **Ensemble weights**: Models with higher AUC-ROC get higher weights, specialists get bonuses

### Getting Help

- Check `model/ensemble_results.json` for ensemble status and weights
- Review `model/method_router.json` for method-to-model mapping
- Check individual model cards for performance metrics and method support
- Use `--detail` flag for verbose detection output with voter counts
- For method-specific errors, verify `method_config.json` in each model directory
