# USAGE: Building and Publishing Project Starlight

These instructions cover the **training-only** workflow: generate data, train the detector, export **GGUF** for Stargate/Trin, and publish to Hugging Face. All commands run from the project **root** unless noted.

Production inference is **not** a local Python API. Serve via **Stargate / Trin** (Go) using the published GGUF.

Canonical flow: **dataset → train → export GGUF → publish HF**.

---

## 1. Datasets

Generate or refresh community-style training data.

```bash
# Root generator (if present)
python3 data_generator.py --limit 10

# Or per-submission generators
cd datasets/<contributor_submission_year>
python3 data_generator.py --limit 10
cd ../..
```

See [DATASET_GUIDELINES.md](DATASET_GUIDELINES.md) for structure (`clean/`, `stego/`, seeds, naming).

---

## 2. Integrity check

Verify dataset layout before training.

```bash
python3 diag.py
```

Expected: file counts, labels, and structure across contributor subdirectories meet protocol standards.

---

## 3. Train `BalancedStarlightDetector`

```bash
python3 trainer.py --epochs 20 --batch_size 16 --out models/detector_balanced.pth
# or (when available): make train
```

**Output (primary checkpoint):** `models/detector_balanced.pth`

Training notes:

- Balanced sampling across clean/stego and sub-datasets
- Multi-head outputs: stego, method, bit-order, embedding
- Ask a human to run long train jobs when agent timeouts apply

---

## 4. Export GGUF (source of truth for Stargate)

```bash
python3 scripts/export_starlight_gguf.py \
  --input models/detector_balanced.pth \
  --output models/starlight.gguf \
  --name-map models/starlight_gguf_map.json
# or: make export-gguf
```

**Artifacts:**

| Local path | Role |
|------------|------|
| `models/starlight.gguf` | Production weights (F32 GGUF v3) |
| `models/starlight_gguf_map.json` | Tensor name map / export metadata |

Full layout and tensor contract: [docs/GGUF_EXPORT.md](docs/GGUF_EXPORT.md).

---

## 5. Parity check

Confirm GGUF tensors match the PyTorch `state_dict` (max abs error &lt; 1e-6).

```bash
python3 scripts/parity_starlight_gguf.py \
  --gguf models/starlight.gguf \
  --input models/detector_balanced.pth
```

For a shape-only CI smoke without trained weights:

```bash
python3 scripts/export_starlight_gguf.py --init-random \
  --output models/starlight.gguf \
  --name-map models/starlight_gguf_map.json
python3 scripts/parity_starlight_gguf.py --gguf models/starlight.gguf --init-random
```

---

## 6. Publish to Hugging Face

Primary upload target: **[macroadster/starlight-prod](https://huggingface.co/macroadster/starlight-prod)**

Requires `HF_TOKEN` and `huggingface_hub` (`pip install huggingface_hub`).

```bash
export HF_TOKEN=hf_...   # never commit tokens
./scripts/publish_to_hf.sh
# or: make publish-hf
```

**Env overrides:**

| Variable | Default |
|----------|---------|
| `HF_TOKEN` | *(required)* |
| `HF_REPO` / `REPO_NAME` | `macroadster/starlight-prod` |
| `GGUF_PATH` | `models/starlight.gguf` |
| `MAP_PATH` | `models/starlight_gguf_map.json` |
| `MODEL_CARD_PATH` | `models/model_card.md` |

**Repo filenames (Stargate download convention):**

- `starlight.gguf`
- `starlight_gguf_map.json`
- `README.md` (from model card)

Download URLs:

```
https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight.gguf
https://huggingface.co/macroadster/starlight-prod/resolve/main/starlight_gguf_map.json
```

Details: [docs/hf_guide.md](docs/hf_guide.md).

---

## 7. Optional local scan (eval only)

```bash
# Single file
python3 scanner.py /path/to/image.png --json

# Directory
python3 scanner.py /path/to/images/ --workers 4
```

This is for offline evaluation during development. It is **not** the product inference API. Production serving is Stargate/Trin loading GGUF from HF.

---

## Troubleshooting

1. **Missing GGUF**: train then run `scripts/export_starlight_gguf.py` (or `make export-gguf`).
2. **Parity failures**: re-export from the same `.pth`; ensure export/parity scripts match `BalancedStarlightDetector`.
3. **HF publish errors**: set `HF_TOKEN`, install `huggingface_hub`, confirm `models/starlight.gguf` and map exist.
4. **Import / deps**: `pip install -r requirements.txt`.
5. **ONNX**: optional/legacy secondary artifact only. Primary path for Stargate is **GGUF**, not ONNX Runtime on device.
