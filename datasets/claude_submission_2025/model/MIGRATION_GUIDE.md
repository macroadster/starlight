# Migration Guide: Method-Aware Architecture

## Overview

This guide explains the changes made to align Claude's submission with Grok's updated **method-aware, multi-modal architecture** proposal.

## What Changed

### 1. **NEW: `method_config.json`** ✨

**Purpose**: Define preprocessing rules for each steganography method

```json
{
  "alpha": {
    "mode": "RGBA",
    "resize": [256, 256],
    "keep_alpha": true
  },
  "palette": {
    "mode": "P", 
    "resize": [256, 256],
    "keep_palette": true
  }
}
```

**Location**: `model/method_config.json`

**Required**: Yes - aggregation pipeline expects this file

---

### 2. **UPDATED: `inference.py`** 🔄

**Major Changes**:

#### Added Method Detection
```python
def _detect_method_from_filename(self, img_path: str) -> str:
    """Auto-detect method from filename pattern"""
    # payloadname_method_index.ext -> extracts "method"
```

#### Added Method-Specific Preprocessing
```python
def preprocess(self, img_path: str, method: str = None):
    """Route to correct preprocessor based on method"""
    if config["mode"] == "RGBA":
        return self._preprocess_rgba(...)
    elif config["mode"] == "P":
        return self._preprocess_palette(...)
```

#### Enhanced Output
```python
return {
    "method": "alpha",  # NEW: detected method
    "stego_probability": 0.87,
    "predicted": "stego",
    "confidence": 0.87
}
```

**Benefits**:
- ✅ Auto-detects method from filename
- ✅ Applies correct preprocessing per method
- ✅ Compatible with aggregation router
- ✅ Graceful fallback if config missing

---

### 3. **UPDATED: `model_card.md`** 📝

**Changes**:
- Added "Supported Steganography Methods" section
- Listed method coverage (`alpha`, `palette`)
- Updated performance metrics to reflect actual results
- Added reference to `method_config.json`
- Honest reporting: Val accuracy 50% (small dataset issue)

**Why Honesty Matters**:
- Aggregation pipeline uses metrics for weighting
- Overstating performance hurts ensemble quality
- Small dataset = expected poor performance

---

### 4. **KEPT: `train.py`** ✅

**No changes needed** because:
- Already handles RGBA properly
- Already has custom collate function
- Already exports to ONNX with 4-channel input
- Method-agnostic training (learns both alpha + palette)

---

### 5. **KEPT: `data_generator.py`** ✅

**No changes needed** because:
- Already generates proper RGBA clean images
- Already creates JSON sidecars with method metadata
- Already uses method naming convention (payloadname_method_index)

---

## File Structure (Updated)

```
datasets/claude_submission_2025/
├── clean/                          # 48 RGBA images
├── stego/                          # 48 stego images + JSON sidecars
├── model/
│   ├── detector.onnx              # Trained model (ONNX)
│   ├── detector_best.pth          # PyTorch checkpoint
│   ├── method_config.json         # NEW: Method preprocessing rules
│   ├── model_card.md              # UPDATED: Method coverage documented
│   ├── inference.py               # UPDATED: Method-aware routing
│   ├── requirements.txt           # Dependencies
│   └── training_history.json      # Training metrics
├── data_generator.py              # Dataset generator (v7)
├── train.py                       # Training script (v2)
├── README.md                      # Submission overview
└── sample_seed.md                 # Payload for embedding
```

---

## How Aggregation Works Now

### Before (Single Model)
```python
# Old: Generic preprocessing
input = preprocess_rgb(image)
output = model(input)
```

### After (Method-Aware Routing)
```python
# New: Method-specific preprocessing
method = detect_method(filename)  # "alpha" or "palette"
config = method_config[method]    # Load preprocessing rules
input = preprocess(image, config) # Apply method-specific preprocessing
output = model(input)              # Run inference
```

### Ensemble Aggregation
```python
# Super model groups by method
super_model = {
    "alpha": [claude_model, alice_model, bob_model],
    "palette": [claude_model, carol_model]
}

# Route to correct sub-ensemble
method = detect_method(image)
ensemble = super_model[method]
predictions = [model.predict(image) for model in ensemble]
final = weighted_average(predictions, weights)
```

---

## Benefits of New Architecture

| Benefit | Description |
|---------|-------------|
| **No Signal Loss** | RGBA images keep alpha channel intact |
| **Modular** | Each method has its own preprocessing |
| **Extensible** | Easy to add new methods (DCT, EXIF, etc.) |
| **Ensemble-Ready** | Router knows which models handle which methods |
| **Auto-Discovery** | Aggregator finds models via `method_config.json` |

---

## Testing the Updates

### 1. Generate Data (if not already done)
```bash
python data_generator.py --limit 12
```

### 2. Train Model (if not already done)
```bash
python train.py --epochs 100 --batch-size 8
```

### 3. Test Method-Aware Inference
```bash
# Test with auto-detection
python model/inference.py --test

# Test specific method
python model/inference.py --image stego/README_alpha_000.png --method alpha

# Batch test
python model/inference.py --batch stego/
```

**Expected Output**:
```
Testing on stego images:
  README_alpha_000.png
    → Method: alpha
    → Prediction: stego
    → Confidence: 0.5123
    → Stego prob: 0.5123
```

---

## Migration Checklist

- [x] Create `model/method_config.json`
- [x] Update `model/inference.py` with method routing
- [x] Update `model/model_card.md` with method coverage
- [x] Update `model_card.md` with honest performance metrics
- [x] Test inference with `--test` flag
- [x] Verify method auto-detection works
- [ ] **Next**: Run aggregation pipeline (when available)

---

## Compatibility

### Backward Compatible?
✅ **Yes** - Old code still works:
- `model.predict(image)` works (auto-detects method)
- Missing `method_config.json` uses sensible defaults
- Old filenames without method name use fallback logic

### Forward Compatible?
✅ **Yes** - Ready for future methods:
- Add entry to `method_config.json`
- Implement preprocessor (e.g., `_preprocess_dct`)
- No changes to model needed

---

## Next Steps

1. **Generate More Data** (1000+ samples recommended)
2. **Retrain Model** on larger dataset
3. **Run Validation Pipeline** (when `scripts/validate_submission.py` available)
4. **Test Aggregation** (when `scripts/aggregate_models.py` available)
5. **Add More Methods** (DCT, EXIF, EOI) as separate models or retrain

---

## Questions?

**Q: Why is validation accuracy only 50%?**  
A: Dataset too small (96 samples). Deep learning needs 1000+ samples minimum. Current model is essentially guessing.

**Q: Should I use this model in production?**  
A: No. This is a reference implementation. Scale up data 10x first.

**Q: Can I add new methods without retraining?**  
A: No. You'd need to train a new model on the new method's data, or retrain the existing model on combined data.

**Q: How does the router know which model to use?**  
A: Filename pattern (payloadname_**method**_index.ext) or JSON sidecar metadata.

---

**Migration Complete!** 🎉  
Your submission is now compatible with Grok's multi-modal aggregation architecture.
