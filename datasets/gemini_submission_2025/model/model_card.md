# Model Card: gemini_detector_2025

## Model Overview
- **Task**: Detection
- **Architecture**: EfficientNet-B4
- **Input**: 256x256 RGB
- **Output**: sigmoid probability

## Training
- **Dataset**: This submission (gemini_submission_2025)
- **Epochs**: 75
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Loss**: BCEWithLogits

## Performance
| Metric | Value |
|--------|-------|
| Accuracy | 99.1% |
| AUC-ROC | 0.998 |
| F1 Score | 0.990 |
| Extraction BER | N/A |

## Steganography Coverage
- `alpha`, `lsb`, `dct`, `exif`, `eoi`

## Inference Speed
- CPU: 15 ms/image
- GPU: 2.5 ms/image

## License
- Model: Apache 2.0
- Code: MIT
