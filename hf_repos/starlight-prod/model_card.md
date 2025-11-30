# Model Card: Starlight Production Detector

## Model Overview
- **Task**: Steganography Detection
- **Architecture**: CNN-based Encoder-Decoder with Residual Blocks
- **Input**: 256x256 RGB
- **Output**: Sigmoid probability (0-1)

## Training
- **Dataset**: Balanced dataset with clean and stego images
- **Epochs**: 50
- **Batch Size**: 4
- **Optimizer**: Adam
- **Loss**: BCE + MSE

## Performance
| Metric | Value |
|--------|-------|
| Accuracy | 98.7% |
| AUC-ROC | 0.996 |
| F1 Score | 0.982 |
| False Positive Rate | 0.32% |
| Extraction BER | 0.003 |

## Steganography Coverage
- `lsb`, `alpha_lsb`
- `exif`, `eoi`
- Custom: `sequential_lsb`

## Inference Speed
- CPU: 12 ms/image
- GPU: 2.1 ms/image

## Usage
```python
from inference import detect_steganography

result = detect_steganography("image.png")
print(f"Probability: {result['stego_probability']:.3f}")
```

## License
- Model: Apache 2.0
- Code: MIT