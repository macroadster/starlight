# Hugging Face Deployment Guide

Starlight production weights for **Stargate / Trin** are published as **GGUF**. GGUF is the source of truth; ONNX (if present) is optional/legacy only.

## Production Repository

**[macroadster/starlight-prod](https://huggingface.co/macroadster/starlight-prod)**

| Artifact | Role |
|----------|------|
| `starlight.gguf` | **Primary** — F32 GGUF v3 weights (`BalancedStarlightDetector`) |
| `starlight_gguf_map.json` | Sidecar — tensor names, shapes, skipped buffers |
| `README.md` | Model card (from `models/model_card.md`) |

Optional secondary (uploaded only if present locally): `detector_balanced.onnx`, `detector_balanced.pth`.

### Download URLs (for Stargate)

```
https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight.gguf
https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight_gguf_map.json
```

Example:

```bash
wget https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight.gguf
wget https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight_gguf_map.json
```

## Inference path

- **Primary**: load GGUF in **Stargate / Trin** (Go). This is the product inference path.
- **Not primary**: Python ONNX Runtime on Raspberry Pi or a local FastAPI server in this repo.
- **Local eval only**: `python3 scanner.py ...` during training development (not a product API).

Tensor contract and export details: [GGUF_EXPORT.md](GGUF_EXPORT.md).

## Research Repository

**[macroadster/starlight-research](https://huggingface.co/macroadster/starlight-research)**

- Experimental / placeholder work
- Not for production Stargate deployments

## How to publish

From the starlight repo root, after train + export:

```bash
# Ensure artifacts exist
ls -lh models/starlight.gguf models/starlight_gguf_map.json models/model_card.md

# Publish (requires HF_TOKEN; never commit the token)
export HF_TOKEN=hf_...
./scripts/publish_to_hf.sh
# or: make publish-hf
```

### Environment overrides

| Variable | Default |
|----------|---------|
| `HF_TOKEN` | *(required)* |
| `HF_REPO` or `REPO_NAME` | `macroadster/starlight-prod` |
| `GGUF_PATH` | `models/starlight.gguf` |
| `MAP_PATH` | `models/starlight_gguf_map.json` |
| `MODEL_CARD_PATH` | `models/model_card.md` |

Requires: `pip install huggingface_hub` (and a write-capable HF token for the target repo).

## Related docs

- [USAGE.md](../USAGE.md) — full train → export → publish workflow
- [GGUF_EXPORT.md](GGUF_EXPORT.md) — export + parity
- [models/model_card.md](../models/model_card.md) — HF README source
