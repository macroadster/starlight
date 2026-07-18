---
license: apache-2.0
tags:
- steganography
- steganalysis
- gguf
- balanced-detector
- image-classification
- computer-vision
- bitcoin
library_name: gguf
pipeline_tag: image-classification
datasets:
- custom
metrics:
- accuracy
- f1
- auc
---

# Model Card: Starlight Balanced Detector (GGUF)

## Model Overview

- **Task**: Steganography detection (and method / bit-order heads)
- **Architecture**: `BalancedStarlightDetector` (multi-stream CNN + fusion gate)
- **Input streams**: pixel, meta, alpha, LSB, palette, format/content features (see GGUF export docs)
- **Primary artifact**: **GGUF** for Stargate / Trin (Go)

## Artifacts (this repo on HF)

| File | Description |
|------|-------------|
| `starlight.gguf` | **Primary** production weights (GGUF v3 / F32) |
| `starlight_gguf_map.json` | Tensor name map and export metadata |
| `README.md` | This model card |

Optional secondary (may appear if uploaded): `detector_balanced.pth`, `detector_balanced.onnx`.

## Inference

**Production**: load `starlight.gguf` via **Stargate / Trin** GGUF path — not Python ONNX as the primary path.

Download:

```
https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight.gguf
https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight_gguf_map.json
```

## Training

- **Dataset**: Combined submissions (grok, gemini, claude, chatgpt, sample, val)
- **Checkpoint**: `models/detector_balanced.pth` (training repo)
- **Export**: `scripts/export_starlight_gguf.py` → `models/starlight.gguf`
- **Typical recipe**: Adam, balanced clean/stego sampling (see training repo `trainer.py`)

## Steganography coverage

- `lsb`, `alpha`, `exif`, `eoi` / raw, `palette` (and related variants in datasets)

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | see training run / leaderboard |
| False positive focus | balanced detector design |

Update metrics after each publish-worthy training run.

## License

- Model: Apache 2.0
- Code: see training repository LICENSE
