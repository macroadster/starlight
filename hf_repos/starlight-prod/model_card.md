---
license: apache-2.0
tags:
- steganography
- steganalysis
- gguf
- balanced-detector
- image-classification
---

# Model Card: Starlight Production Detector

## Model Overview

- **Task**: Steganography detection
- **Architecture**: `BalancedStarlightDetector`
- **Primary artifact**: `starlight.gguf` (GGUF v3 / F32)
- **Sidecar**: `starlight_gguf_map.json`

## Inference

Load via **Stargate / Trin** (Go) GGUF path. GGUF is the source of truth for production.

```
https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight.gguf
https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight_gguf_map.json
```

ONNX/Python paths are optional/legacy only — not the primary product inference path.

## Training source

Train and export from the starlight training repo:

```bash
python3 trainer.py --out models/detector_balanced.pth
python3 scripts/export_starlight_gguf.py \
  --input models/detector_balanced.pth \
  --output models/starlight.gguf \
  --name-map models/starlight_gguf_map.json
HF_TOKEN=... ./scripts/publish_to_hf.sh
```

## License

- Model: Apache 2.0
