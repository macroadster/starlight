# **Project Starlight: Model Contribution & Aggregation Proposal**  
**Goal:** Evolve Starlight from a *data-centric* steganalysis dataset into a **multi-modal, federated AI ecosystem** where contributors submit not only data, but **trained detection/extraction models** — enabling a **super model** via ensemble aggregation.

---

## **1. New Directory Structure (Model-Enabled)**

```text
datasets/
└── [username]_submission_[year]/
    ├── clean/                      # Clean images
    ├── stego/                      # Stego images
    ├── sample_seed.md              # Payload seeds
    ├── data_generator.py           # (Optional) Data generation
    ├── model/                      # NEW: Model contribution
    │   ├── detector.onnx           # ONNX model (detection)
    │   ├── extractor.onnx          # ONNX model (extraction, optional)
    │   ├── model_card.md           # Model metadata & performance
    │   ├── requirements.txt        # Dependencies
    │   └── inference.py            # Inference wrapper (standardized)
    └── README.md                   # Submission overview
```

---

## **2. Model Contribution Guidelines**

### **2.1 Model Types**
| Type | Purpose | Output |
|------|-------|--------|
| `detector` | Binary classification: **clean vs stego** | Probability score `[0,1]` |
| `extractor` | Payload recovery: **extract hidden message** | Bytes or string |

> Both must be exported in **ONNX 1.12+** format for interoperability.

---

### **2.2 `model_card.md` (Required)**

```markdown
# Model Card: [username]_[algo]_[year]

## Model Overview
- **Task**: Detection / Extraction
- **Architecture**: e.g., EfficientNet-B3, ResNet-50, CNN-LSTM, ViT
- **Input**: 256x256 RGB (or specify)
- **Output**: 
  - Detector: sigmoid probability
  - Extractor: variable-length byte sequence

## Training
- **Dataset**: This submission + [external sources]
- **Epochs**: 50
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Loss**: BCEWithLogits (detector), CrossEntropy (extractor)

## Performance
| Metric | Value |
|--------|-------|
| Accuracy | 98.7% |
| AUC-ROC | 0.996 |
| F1 Score | 0.982 |
| Extraction BER | 0.003 |

## Steganography Coverage
- `alpha`, `lsb`, `dct`, `exif`, `eoi`
- Custom: `wavelet_lsb`

## Inference Speed
- CPU: 12 ms/image
- GPU: 2.1 ms/image

## License
- Model: Apache 2.0
- Code: MIT
```

---

### **2.3 `inference.py` (Standardized Interface)**

```python
# inference.py
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

class StarlightModel:
    def __init__(self, model_path="model/detector.onnx", task="detect"):
        self.session = ort.InferenceSession(model_path)
        self.task = task  # 'detect' or 'extract'
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img_path):
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)

    def predict(self, img_path):
        input_data = self.preprocess(img_path)
        outputs = self.session.run(None, {self.input_name: input_data})
        
        if self.task == "detect":
            prob = float(outputs[0][0])  # sigmoid
            return {"stego_probability": prob, "predicted": prob > 0.5}
        else:
            payload_bytes = ''.join([chr(int(b)) for b in outputs[0]])
            return {"extracted_payload": payload_bytes}
```

> All models **must** include this file with identical interface.

---

### **2.4 `requirements.txt`**

```txt
onnxruntime>=1.16.0
Pillow>=9.0.0
numpy>=1.21.0
```

---

## **3. Top-Level Aggregation Script: `aggregate_models.py`**

Located at: `scripts/aggregate_models.py`

### **3.1 Features**
- Auto-discovers all `model/detector.onnx` in `datasets/*`
- Builds **weighted ensemble** based on `model_card.md` AUC
- Supports **detection-only**, **extraction-only**, or **joint**
- Outputs final **super model** in ONNX + PyTorch script

---

### **3.2 Aggregation Logic**

```python
# scripts/aggregate_models.py
import os
import json
import onnx
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

class SuperStarlightDetector(nn.Module):
    def __init__(self, model_paths, weights):
        super().__init__()
        self.models = [self.load_onnx(p) for p in model_paths]
        self.weights = torch.tensor(weights)

    def load_onnx(self, path):
        # Convert ONNX → TorchScript via tracing (dummy input)
        dummy = torch.randn(1, 3, 256, 256)
        return torch.onnx.export_to_script_module(path, dummy)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        stacked = torch.stack(outputs, dim=0)
        return torch.sum(stacked * self.weights.unsqueeze(1), dim=0)
```

---

### **3.3 Weight Assignment Strategy**

| Criteria | Weight Multiplier |
|--------|-------------------|
| AUC ≥ 0.99 | ×1.5 |
| AUC ≥ 0.95 | ×1.2 |
| Covers 3+ algos | ×1.1 |
| Extraction BER < 0.01 | ×1.3 |
| Inference < 5ms (GPU) | ×1.1 |

> Final weight = base(1.0) × product of multipliers

---

### **3.4 Output**

```text
models/
├── super_detector.onnx
├── super_extractor.onnx
├── ensemble_weights.json
└── leaderboard.md
```

#### `leaderboard.md` (Auto-generated)
```markdown
# Starlight Leaderboard (Nov 2025)

| Rank | Contributor | AUC | Coverage | Speed | Weight |
|------|-------------|-----|----------|-------|--------|
| 1 | grok_submission_2025 | 0.998 | 5 algos | 2.1ms | 2.31 |
| 2 | alice_submission_2025 | 0.992 | 4 algos | 3.4ms | 1.98 |
```

---

## **4. Validation Pipeline**

```bash
python scripts/validate_submission.py datasets/grok_submission_2025/
# Checks:
# - model/inference.py runs
# - ONNX loads
# - model_card.md parsed
# - 1:1 clean/stego alignment
# - No malicious code
```

---

## **5. Benefits of This System**

| Benefit | Impact |
|-------|--------|
| **Federated Learning** | No data sharing needed |
| **Diversity** | Models trained on unique data/algos |
| **Continuous Improvement** | New submissions auto-boost super model |
| **Explainability** | Leaderboard shows contribution impact |
| **Extraction Capability** | Beyond detection → full recovery |

---

## **6. Future Extensions**

- **Model Distillation**: Distill ensemble into single efficient model
- **Adversarial Hardening**: Auto-generate attacks from extractors
- **On-Chain Verification**: Run super model via blockchain oracle
- **API Endpoint**: `POST /detect` → returns ensemble result

---

## **7. Call to Action**

> **Contribute your model today!**  
> Even a single well-trained detector on `alpha` + `dct` can boost the super model.

```bash
# Example contribution
mkdir -p datasets/grok_submission_2025/model
cp my_detector.onnx datasets/grok_submission_2025/model/detector.onnx
cp inference.py model_card.md requirements.txt datasets/grok_submission_2025/model/
```

---

**Starlight is now a living, evolving AI — powered by the community.**  
Let’s build the ultimate blockchain steganalysis superintelligence.
