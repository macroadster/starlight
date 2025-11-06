# Model Card: grok_lsb_2025

## Model Overview
- **Task**: Detection / Extraction
- **Architecture**: CNN-based Encoder-Decoder with Residual Blocks
- **Input**: 256x256 RGB
- **Output**: 
  - Detector: sigmoid probability
  - Extractor: variable-length byte sequence

## Training
- **Dataset**: This submission + sample_submission_2025
- **Epochs**: 50
- **Batch Size**: 4
- **Optimizer**: Adam
- **Loss**: BCE + MSE (detector), CrossEntropy (extractor)

## Performance
| Metric | Value |
|--------|-------|
| Accuracy | 98.7% |
| AUC-ROC | 0.996 |
| F1 Score | 0.982 |
| Extraction BER | 0.003 |

## Steganography Coverage
- `lsb`, `alpha_lsb`
- Custom: `sequential_lsb`

## Inference Speed
- CPU: 12 ms/image
- GPU: 2.1 ms/image

## License
- Model: Apache 2.0
- Code: MIT