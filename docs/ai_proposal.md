# **Project Starlight: Model Contribution & Aggregation Proposal**  
**Updated: November 05, 2025**  
**Goal:** Evolve Starlight into a **multi-modal, federated AI ecosystem** where contributors submit trained **detection and extraction models** alongside data. Models are **method-aware**, support **non-RGB steganography**, and are aggregated into a **super model** via ensemble fusion.

---

## **1. Directory Structure (Model-Enabled)**

```text
datasets/
└── [username]_submission_[year]/
    ├── clean/                      # Clean images
    ├── stego/                      # Stego images
    ├── sample_seed.md              # Optional payload seeds
    ├── data_generator.py           # (Optional) Data generation
    ├── model/                      # NEW: Model contribution
    │   ├── detector.onnx           # ONNX detection model
    │   ├── extractor.onnx          # ONNX extraction model (optional)
    │   ├── model_card.md           # Model metadata & performance
    │   ├── method_config.json      # REQUIRED: Per-method preprocessing rules
    │   ├── inference.py            # Standardized, method-aware inference
    │   └── requirements.txt        # Python dependencies
    └── README.md                   # Submission overview
```

---

## **2. Model Contribution Guidelines**

### **2.1 Model Types**

| Type | Purpose | Output |
|------|---------|--------|
| `detector` | Binary classification: **clean vs stego** | `stego_probability ∈ [0,1]` |
| `extractor` | Payload recovery | Extracted bytes or string |

> **All models must be exported in ONNX 1.12+ format**.

---

### **2.2 `model_card.md` (Required)**

```markdown
# Model Card: [username]_[algo]_[year]

## Model Overview
- **Task**: Detection / Extraction
- **Architecture**: e.g., ResNet-50, EfficientNet-B3, ViT, CNN-LSTM
- **Input**: Varies by method (see `method_config.json`)
- **Output**: 
  - Detector: sigmoid probability
  - Extractor: variable-length byte sequence

## Supported Steganography Methods
- `alpha`, `lsb`, `dct`, `exif`, `eoi`, `palette`
- Custom: `wavelet_lsb`

## Training
- **Dataset**: This submission + [external]
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

## Inference Speed
- CPU: 12 ms/image (RGB), 15 ms (RGBA)
- GPU: 2.1 ms/image

## License
- Model: Apache 2.0
- Code: MIT
```

---

### **2.3 `method_config.json` (REQUIRED)**

```json
{
  "lsb": {
    "mode": "RGB",
    "resize": [256, 256]
  },
  "alpha": {
    "mode": "RGBA",
    "resize": [256, 256],
    "keep_alpha": true
  },
  "dct": {
    "mode": "DCT",
    "input_shape": [1, 4096]
  },
  "exif": {
    "mode": "EXIF",
    "max_bytes": 1024
  },
  "eoi": {
    "mode": "EOI",
    "tail_bytes": 1024
  },
  "palette": {
    "mode": "P",
    "resize": [256, 256],
    "keep_palette": true
  }
}
```

> **All supported methods in `model_card.md` must have an entry in `method_config.json`**.

---

### **2.4 `inference.py` — **Multi-Modal & Method-Aware**

```python
# datasets/[username]_submission_[year]/model/inference.py
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import json
from typing import Dict, Any
import jpegio as jio  # pip install jpegio

class StarlightModel:
    def __init__(
        self,
        detector_path: str = "model/detector.onnx",
        extractor_path: str = None,
        task: str = "detect"
    ):
        self.detector = ort.InferenceSession(detector_path)
        self.extractor = ort.InferenceSession(extractor_path) if extractor_path else None
        self.task = task
        self.input_name = self.detector.get_inputs()[0].name
        self.method_config = self._load_method_config()

    def _load_method_config(self) -> Dict:
        config_path = "model/method_config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError("method_config.json is required")
        with open(config_path) as f:
            return json.load(f)

    def _detect_method_from_filename(self, img_path: str) -> str:
        basename = os.path.basename(img_path)
        parts = basename.split("_")
        if len(parts) >= 3:
            return parts[-2]  # e.g., alpha, eoi, dct
        return "lsb"

    def preprocess(self, img_path: str, method: str = None) -> np.ndarray:
        method = method or self._detect_method_from_filename(img_path)
        config = self.method_config.get(method, self.method_config["lsb"])

        if config["mode"] == "RGB":
            return self._preprocess_rgb(img_path, config)
        elif config["mode"] == "RGBA":
            return self._preprocess_rgba(img_path, config)
        elif config["mode"] == "DCT":
            return self._preprocess_dct(img_path)
        elif config["mode"] == "EXIF":
            return self._preprocess_exif(img_path)
        elif config["mode"] == "EOI":
            return self._preprocess_eoi(img_path)
        elif config["mode"] == "P":
            return self._preprocess_palette(img_path, config)
        else:
            raise NotImplementedError(f"Mode {config['mode']} not supported")

    def _preprocess_rgb(self, img_path, config):
        img = Image.open(img_path).convert("RGB").resize(config["resize"])
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)

    def _preprocess_rgba(self, img_path, config):
        img = Image.open(img_path).convert("RGBA").resize(config["resize"])
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # (4, H, W)
        return np.expand_dims(arr, 0)

    def _preprocess_dct(self, img_path):
        jpeg = jio.read(img_path)
        coeffs = jpeg.coef_blocks[0]
        coeffs = (coeffs - coeffs.mean()) / (coeffs.std() + 1e-8)
        padded = np.pad(coeffs.flatten(), (0, 4096), 'constant')[:4096]
        return padded.astype(np.float32).reshape(1, -1)

    def _preprocess_exif(self, img_path):
        img = Image.open(img_path)
        exif = img.info.get("exif")
        data = np.frombuffer(exif or b'', dtype=np.uint8)[:1024]
        padded = np.zeros(1024, dtype=np.float32)
        padded[:len(data)] = data.astype(np.float32) / 255.0
        return padded.reshape(1, -1)

    def _preprocess_eoi(self, img_path):
        with open(img_path, 'rb') as f:
            f.seek(-1024, 2)
            tail = f.read()
        eoi_pos = tail.find(b'\xFF\xD9')
        appended = tail[eoi_pos + 2:] if eoi_pos != -1 else tail
        data = np.frombuffer(appended, dtype=np.uint8)[:1024]
        padded = np.zeros(1024, dtype=np.float32)
        padded[:len(data)] = data.astype(np.float32) / 255.0
        return padded.reshape(1, -1)

    def _preprocess_palette(self, img_path, config):
        img = Image.open(img_path).convert("P").resize(config["resize"])
        indices = np.array(img)
        palette = np.array(img.getpalette()[:768]).reshape(256, 3) / 255.0
        hist = np.histogram(indices, bins=256, range=(0, 255), density=True)[0]
        combined = np.concatenate([hist, palette.flatten()])
        return combined.astype(np.float32).reshape(1, -1)

    def predict(self, img_path: str, method: str = None) -> Dict[str, Any]:
        method = method or self._detect_method_from_filename(img_path)
        input_data = self.preprocess(img_path, method)

        if self.task == "detect":
            outputs = self.detector.run(None, {self.input_name: input_data})
            prob = float(outputs[0][0])
            return {
                "method": method,
                "stego_probability": prob,
                "predicted": prob > 0.5
            }
        elif self.task == "extract" and self.extractor:
            outputs = self.extractor.run(None, {self.extractor.get_inputs()[0].name: input_data})
            payload = ''.join(chr(int(b)) for b in np.argmax(outputs[0], axis=-1))
            return {
                "method": method,
                "extracted_payload": payload
            }
```

---

### **2.5 `requirements.txt`**

```txt
onnxruntime>=1.16.0
Pillow>=9.0.0
numpy>=1.21.0
jpegio>=0.2.0
```

---

## **3. Top-Level Aggregation: `scripts/aggregate_models.py`**

### **3.1 Features**
- Auto-discovers all `model/detector.onnx` and `method_config.json`
- **Groups models by steganography method**
- Builds **per-method weighted ensembles**
- Routes input to correct sub-ensemble at inference
- Outputs **super model** with **method router**

---

### **3.2 Aggregation Logic**

```python
class SuperStarlightModel:
    def __init__(self):
        self.method_ensembles = {}  # "alpha": [model1, model2], ...
        self.weights = {}           # "alpha": [0.6, 0.4]
        self.load_all_models()

    def load_all_models(self):
        for subdir in Path("datasets").glob("*_submission_*"):
            config_path = subdir / "model" / "method_config.json"
            if not config_path.exists():
                continue
            with open(config_path) as f:
                config = json.load(f)
            for method, cfg in config.items():
                model_path = subdir / "model" / "detector.onnx"
                if model_path.exists():
                    self.method_ensembles.setdefault(method, []).append(model_path)
                    # Weight by AUC from model_card.md
```

---

### **3.3 Output**

```text
models/
├── super_detector.onnx
├── super_extractor.onnx
├── method_router.json
├── ensemble_weights.json
└── leaderboard.md
```

---

## **4. Validation Pipeline**

```bash
python scripts/validate_submission.py datasets/grok_submission_2025/
```
**Checks:**
- `method_config.json` exists and is valid
- All methods in `model_card.md` are in `method_config.json`
- `inference.py` runs on sample images:
  - `alpha` → `(1,4,256,256)`
  - `eoi` → `(1,1024)`
  - `dct` → `(1,4096)`
- ONNX models load
- No malicious code

---

## **5. Multi-Modal Input Support**

| Method | Input Shape | Preserved Signal |
|-------|-------------|------------------|
| `lsb` | `(1,3,256,256)` | RGB LSBs |
| `alpha` | `(1,4,256,256)` | Alpha channel |
| `dct` | `(1,4096)` | DCT coefficients |
| `exif` | `(1,1024)` | EXIF blob |
| `eoi` | `(1,1024)` | Post-EOI bytes |
| `palette` | `(1,1024)` | Palette + histogram |

---

## **6. Benefits**

| Benefit | Impact |
|-------|--------|
| **Full Method Coverage** | No signal loss from RGB conversion |
| **Federated & Modular** | Add new method → new sub-model |
| **Auto-Routing** | Super model knows *how* to process |
| **Continuous Evolution** | New submissions improve per-method accuracy |

---

## **7. Future Extensions**

- **Dynamic Method Detection** via metadata
- **Adversarial Training Loop** using extractors
- **On-Chain Inference** via WASM/ONNX Runtime
- **API**: `POST /detect` → returns method + probability + payload

---

## **8. Call to Action**

> **Contribute your method-aware model today!**  
> Even a single `alpha` or `eoi` detector boosts the super model.

```bash
mkdir -p datasets/grok_submission_2025/model
cp detector.onnx method_config.json inference.py model_card.md requirements.txt datasets/grok_submission_2025/model/
```

---

**Starlight is now a living, multi-modal steganalysis superintelligence — powered by the community.**  
Let’s detect and extract *every* hidden message in the blockchain.
