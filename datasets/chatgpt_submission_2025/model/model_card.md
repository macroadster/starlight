# Model Card: chatgpt_cnn_2025

## Model Overview
- **Task**: Steganography Detection (Binary Classification)
- **Architecture**: Simple CNN with metadata fusion
- **Input**: 224x224 RGB image + 3-dimensional metadata vector
- **Output**: Binary classification logits [clean, stego]

## Training
- **Dataset**: ChatGPT Submission 2025 (self-generated)
  - Clean images: 450
  - Stego images: 590 (across 5 algorithms)
  - Algorithms: alpha, eoi, exif, lsb, palette
- **Training Samples**: 280 paired samples (140 stego/clean pairs)
- **Validation Samples**: 280 paired samples
- **Epochs**: 15 (with early stopping, patience=5)
- **Batch Size**: 32 (CUDA accelerated)
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Loss**: MSELoss (Siamese training)
- **Hardware**: NVIDIA GeForce RTX 3080 Ti (CUDA)
- **Best Validation Accuracy**: 74.64% (threshold: 0.35)

## Performance
| Metric | Value |
|--------|-------|
| Validation Accuracy | 74.64% (achieved) |
| Input Size | 224x224 RGB |
| Model Size | ~7MB |
| Parameters | ~2M |
| Training Time | ~3 minutes (CUDA) |

## Steganography Coverage
- `alpha` - Alpha channel LSB embedding
- `lsb` - Standard LSB embedding in RGB channels  
- `palette` - Palette index manipulation (GIF)
- `exif` - EXIF metadata embedding (JPEG)
- `eoi` - End-of-image payload appending

## Inference Speed
- CPU: ~15ms/image (estimated)
- GPU: ~3ms/image (estimated)

## Technical Details
### Architecture
- **CNN Backbone**: 3-layer CNN (32→64→128 channels)
- **Metadata Processing**: 2-layer MLP for EXIF/EOI features
- **Feature Fusion**: Concatenation of CNN and metadata features
- **Classifier**: 2-layer MLP with dropout

### Input Format
- **RGB**: Normalized tensor [0,1] with shape (batch, 3, 224, 224)
- **Metadata**: 3-dimensional vector [exif_present, exif_length, eoi_length]

### Output Format
- **Logits**: Raw scores for [clean, stego] classes
- **Probability**: Apply softmax for confidence scores

## Usage
```python
# Load ONNX model
import onnxruntime as ort
session = ort.InferenceSession("model/detector.onnx")

# Preprocess image
rgb_tensor = preprocess_image("path/to/image.jpg")  # (1, 3, 224, 224)
metadata = extract_metadata("path/to/image.jpg")  # (1, 3)

# Run inference
outputs = session.run(None, {"rgb": rgb_tensor, "metadata": metadata})
probabilities = softmax(outputs[0])
stego_probability = probabilities[0, 1]  # Probability of stego
```

## Limitations
- Trained on synthetic dataset; performance may vary on real-world images
- Binary classification only (does not identify specific steganography algorithm)
- Assumes images are preprocessed to 224x224 RGB format

## License
- Model: MIT License
- Code: MIT License

## Starlight Compatibility
- ✅ ONNX export (detector.onnx)
- ✅ Standardized inference interface
- ✅ Model card documentation
- ✅ Requirements specification
- ✅ Multi-algorithm coverage