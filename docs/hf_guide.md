# Hugging Face Deployment Guide

## Starlight Model Repositories

### Production Repository: [macroadster/starlight-prod](https://huggingface.co/macroadster/starlight-prod)
- **Purpose**: Production-ready steganography detection model
- **Model**: `detector_balanced.onnx` (0.32% false positive rate)
- **Features**: 
  - No special cases or rules
  - Optimized for CPU inference
  - Supports multiple steganography methods (LSB, EXIF, EOI)

### Research Repository: [macroadster/starlight-research](https://huggingface.co/macroadster/starlight-research)
- **Purpose**: Experimental V3 model development
- **Status**: Placeholder - under active development
- **Warning**: Experimental results, not for production use

## Try on Raspberry Pi

The production model is optimized for CPU inference and can run on resource-constrained devices like Raspberry Pi:

```bash
# Install dependencies
pip install onnxruntime pillow numpy

# Download model
wget https://huggingface.co/macroadster/starlight-prod/resolve/main/detector_balanced.onnx

# Run inference
python3 -c "
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession('detector_balanced.onnx')

# Load and preprocess image
img = Image.open('your_image.png').convert('RGB').resize((256, 256))
arr = np.array(img).astype(np.float32) / 255.0
arr = np.transpose(arr, (2, 0, 1))
arr = np.expand_dims(arr, 0)

# Run detection
outputs = session.run(None, {'input': arr})
prob = outputs[0][0]
print(f'Steganography probability: {prob:.3f}')
"
```

## Usage

```python
from inference import detect_steganography

result = detect_steganography("image.png")
print(f"Probability: {result['stego_probability']:.3f}")
```