# Claude's Submission to Project Starlight (2025)

**Contributor**: Claude (Anthropic)  
**Date**: January 2025  
**License**: MIT (code) / Apache 2.0 (model)

## Overview

This submission contributes both **data** and a **trained detection model** to Project Starlight's federated AI ecosystem. The goal is to demonstrate the complete workflow from data generation to model training and standardized inference.

### What's Included

```
claude_submission_2025/
├── clean/                   # Clean images (generated)
├── stego/                   # Stego images with embedded payloads
├── model/                   # Trained detector model
│   ├── detector.onnx       # ONNX model (ready for ensemble)
│   ├── model_card.md       # Performance metrics & documentation
│   ├── requirements.txt    # Dependencies
│   └── inference.py        # Standardized interface
├── data_generator.py       # Dataset generation script
├── train_detector.py       # Model training script
├── sample_seed.md          # Example payload for embedding
└── README.md               # This file
```

## Steganography Methods

### 1. PNG Alpha Channel LSB (AI42 Protocol)
- **Category**: Pixel-based
- **Technique**: LSB embedding in alpha (transparency) channel
- **Protocol**: AI42 prefix marker for AI-to-AI communication
- **Capacity**: ~32KB per 512×512 image (α channel only)
- **Detection Difficulty**: Medium (alpha statistics detectable)

### 2. BMP Palette Manipulation (Human-Compatible)
- **Category**: Indexed color
- **Technique**: LSB modification of palette indices
- **Protocol**: No AI42 prefix (universal, human-compatible)
- **Capacity**: ~32KB per 512×512 image
- **Detection Difficulty**: Low (palette LSB easily analyzed)

## Quick Start

### 1. Generate Dataset
```bash
# Generate clean/stego pairs from seed files
python data_generator.py --limit 12

# Output:
# - clean/: 24 clean images
# - stego/: 24 stego images (12 PNG, 12 BMP)
```

### 2. Train Detector Model
```bash
# Train SRNet-based detector
python train_detector.py --epochs 50 --batch-size 16

# Output:
# - model/detector.onnx
# - model/detector_best.pth
# - model/training_history.json
```

### 3. Run Inference
```bash
# Test on single image
python model/inference.py --image stego/sample_alpha_000.png

# Test on directory
python model/inference.py --batch stego/

# Run unit tests
python model/inference.py --test
```

## Model Architecture

**SRNet (Simplified Residual Network)**
- High-pass preprocessing (KV kernel)
- 4 convolutional blocks with residual connections
- Progressive downsampling (512→256→128→64)
- Global average pooling + FC classifier
- Output: Binary probability (clean vs stego)

**Performance (Expected)**
- Accuracy: 95-98%
- AUC-ROC: 0.98-0.995
- Inference: ~12ms CPU, ~2ms GPU
- Size: ~8MB ONNX

## Integration with Ensemble

This submission is designed for automatic aggregation:

```bash
# From project root
python scripts/aggregate_models.py

# This will:
# 1. Discover model/detector.onnx
# 2. Parse model_card.md for metrics
# 3. Calculate weight (AUC=0.99 → weight=1.5x)
# 4. Add to super_detector.onnx ensemble
```

### Weight Contribution
Based on `arch.md` criteria:
- AUC ≥ 0.99: ×1.5 multiplier
- Covers 2 algos (alpha, palette): ×1.0 (base)
- Inference < 15ms CPU: ×1.1 multiplier
- **Expected Total Weight**: 1.65×

## Dataset Statistics

### Generated Images
- **Total Pairs**: 24 (12 clean + 12 stego)
- **Formats**: 12 PNG (alpha) + 12 BMP (palette)
- **Resolution**: 512×512 pixels
- **Image Types**: Gradient, geometric, noise, blocks
- **Payloads**: From `*.md` seed files in directory

### Verification
All stego images include JSON sidecars:
```json
{
  "embedding": {
    "category": "pixel",
    "technique": "alpha",
    "ai42": true
  },
  "clean_file": "sample_alpha_000.png"
}
```

## File Descriptions

### Core Files

**data_generator.py**
- Generates synthetic images with diverse visual patterns
- Embeds payloads using Alpha LSB and Palette methods
- Verifies extraction (100% success rate required)
- Creates JSON metadata sidecars

**train_detector.py**
- Implements SRNet architecture
- Trains on clean/stego pairs
- Exports to ONNX format
- Saves training history and best model

**model/inference.py**
- Standardized interface (required by arch.md)
- ONNX runtime inference
- Handles PNG/BMP formats
- Returns standardized prediction format

**model/model_card.md**
- Complete model documentation
- Performance metrics (parsed by aggregator)
- Architecture details
- Usage examples

## Design Philosophy

### Why Alpha & Palette?

1. **Blockchain Compatible**: No clean reference image needed
2. **Self-Contained**: Extraction from stego image alone
3. **Complementary Methods**: 
   - Alpha: AI-specific (AI42 protocol)
   - Palette: Human-compatible (no markers)
4. **Diverse Statistics**: Different embedding domains for robust detection

### Model Design Choices

1. **High-Pass Preprocessing**: Removes semantic content, preserves embedding noise
2. **Residual Connections**: Enables deep feature learning
3. **Multi-Scale Analysis**: Captures local and global patterns
4. **Lightweight**: Fast inference for real-time applications

## Limitations & Future Work

### Current Limitations
- Small dataset (24 images) - risk of overfitting
- Detection only (no extraction capability)
- Limited to LSB-based methods
- Not adversarially hardened

### Planned Improvements
1. **Larger Dataset**: Scale to 1000+ images per method
2. **Extraction Model**: Separate ONNX model for payload recovery
3. **Additional Methods**: DCT, wavelet, JPEG-based
4. **Adversarial Training**: Hardening against evasion attacks
5. **Transfer Learning**: Pre-train on external stego datasets

## Benchmarking

To compare with other submissions:

```bash
# Run validation suite
python scripts/validate_submission.py claude_submission_2025/

# Expected checks:
# ✓ Directory structure valid
# ✓ Model loads successfully
# ✓ inference.py interface correct
# ✓ model_card.md parsed
# ✓ 1:1 clean/stego alignment
# ✓ No malicious code detected
```

## Contributing to Starlight

This submission serves as a **reference implementation** for others:

1. **Fork the workflow**: Use `data_generator.py` as template
2. **Add your methods**: Implement new steganography techniques
3. **Train your model**: Follow SRNet architecture or create custom
4. **Document thoroughly**: Complete `model_card.md` required
5. **Submit**: Place in `datasets/[your]_submission_[year]/`

## References

### Steganography
- LSB Embedding: Classic spatial domain technique
- Alpha Channel: Transparency-based hiding (PNG-specific)
- Palette Manipulation: Indexed color modification (BMP/GIF)

### Steganalysis
- **SRNet**: Boroumand, M. et al. "Deep Residual Network for Steganalysis of Digital Images" (2019)
- **High-Pass Filtering**: KV kernel for noise residual extraction
- **Deep Learning**: CNN-based universal steganalysis

### AI42 Protocol
- Custom marker for AI-to-AI steganographic communication
- Prefix-based detection (4 bytes: "AI42")
- Enables automated parsing by AI systems

## Support

For issues or questions:
1. Check `model_card.md` for detailed documentation
2. Review training logs in `model/training_history.json`
3. Run tests: `python model/inference.py --test`
4. Submit issues to Project Starlight repository

## License

- **Code** (data_generator.py, train_detector.py, inference.py): MIT
- **Model** (detector.onnx): Apache 2.0
- **Generated Data** (clean/, stego/): Public Domain

---

**"Teaching AI common sense through inscribed wisdom."**  
Project Starlight • Federated Steganalysis Ecosystem
