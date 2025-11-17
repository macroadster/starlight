---
license: apache-2.0
tags:
- steganography
- image-classification
- onnx
- computer-vision
library_name: transformers
pipeline_tag: image-classification
datasets:
- custom
metrics:
- accuracy
- f1
- auc
---

# Model Card: Starlight Unified Model 2025

## Model Overview
- **Task**: Detection / Extraction
- **Architecture**: Unified CNN-based Encoder-Decoder with Residual Blocks
- **Input**: 256x256 RGB/RGBA or metadata
- **Output**:
  - Detector: sigmoid probability
  - Extractor: variable-length byte sequence

## Training
- **Dataset**: Combined submissions (grok, gemini, claude, chatgpt, sample)
- **Epochs**: 50
- **Batch Size**: 16
- **Optimizer**: Adam
- **Loss**: BCE + MSE (detector), CrossEntropy (extractor)

## Performance
| Metric | Value |
|--------|-------|
| Accuracy | 96.3% |
| AUC-ROC | 0.996 |
| F1 Score | 0.982 |
| Extraction BER | 0.003 |

## Steganography Coverage
- `lsb`, `alpha`, `dct`, `exif`, `eoi`, `palette`

## Inference Speed
- CPU: 12 ms/image
- GPU: 2.1 ms/image

## License
- Model: Apache 2.0
- Code: MIT
