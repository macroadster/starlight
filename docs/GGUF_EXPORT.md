# Starlight GGUF Export (BalancedStarlightDetector)

Export the production balanced detector to a **GGUF v3 / F32** container that Trin
(`internal/weights.WriteGGUFF32` / `OpenGGUF`) can load.

## Quick start

```bash
# With a trained checkpoint
python3 scripts/export_starlight_gguf.py \
  --input models/detector_balanced.pth \
  --output models/starlight.gguf \
  --name-map models/starlight_gguf_map.json

# CI / no weights available (random init; shapes are deterministic)
python3 scripts/export_starlight_gguf.py --init-random \
  --output models/starlight.gguf \
  --name-map models/starlight_gguf_map.json

# Parity: GGUF bytes vs live state_dict (max |Δ| < 1e-6)
python3 scripts/parity_starlight_gguf.py \
  --gguf models/starlight.gguf --init-random
```

## Architecture

| Field | Value |
|-------|-------|
| Class | `BalancedStarlightDetector` (`trainer.py`) |
| `meta_weight` | `0.3` |
| `proj_dim` | `64` |
| `fusion_dim` | `6*64 + 2*32 = 448` |
| Params (approx) | ~0.81M (~3.3 MB F32 payload) |

### Meta keys (GGUF string KVs)

| Key | Example |
|-----|---------|
| `general.name` | `starlight-detector-balanced-v2` |
| `starlight.meta_weight` | `0.3` |
| `starlight.proj_dim` | `64` |
| `starlight.fusion_dim` | `448` |
| `starlight.arch` | `BalancedStarlightDetector` |

Also written: `general.alignment` = `32` (UINT32), matching Trin.

## Tensor naming contract

**Keys are exact PyTorch `state_dict` names.** Examples:

- `pixel_conv.0.weight` — Conv2d (4-D)
- `pixel_conv.1.running_mean` — BatchNorm buffer
- `alpha_proj.weight` — Linear (2-D)
- `stream_gate.0.weight`
- `stego_head.bias`

Layout in GGUF:

- Type: `GGML_F32` (type code `0`)
- Values: little-endian float32, **row-major**, identical to
  `tensor.detach().cpu().contiguous().float().reshape(-1)`
- Shape: `list(tensor.shape)` stored as `n_dims × u64` in torch order
- Tensor infos sorted **alphabetically by name** (stable)
- Per-tensor data offsets padded to 32-byte alignment

### Skipped buffers

Integer buffers such as `*.num_batches_tracked` (dtype `int64`) are **not**
exported. Eval uses `running_mean` / `running_var` only. Skipped keys are listed
in `models/starlight_gguf_map.json` under `"skipped"`.

## Name map JSON

`models/starlight_gguf_map.json` documents every exported tensor:

```json
{
  "arch": "BalancedStarlightDetector",
  "n_tensors_exported": 128,
  "n_skipped": 15,
  "skipped": [ { "name": "...", "dtype": "int64", "reason": "..." } ],
  "tensors": {
    "pixel_conv.0.weight": {
      "shape": [16, 3, 3, 3],
      "dtype": "F32",
      "nbytes": 1728,
      "role": "conv",
      "numel": 432
    }
  }
}
```

Role heuristic: `conv` / `conv1d` / `linear` / `bias` / `bn` / `param`.

## Input tensors (inference)

From `scripts/starlight_utils.py` → `load_unified_input` (add batch dim `1`):

| Stream | Shape | Notes |
|--------|-------|-------|
| `pixel_tensor` | `(1, 3, 256, 256)` | RGB ToTensor |
| `meta` | `(1, 2048)` | EXIF+tail bytes / 255; model does `unsqueeze(1)` → `(1,1,2048)` for Conv1d |
| `alpha` | `(1, 2, 256, 256)` | full alpha + alpha LSB (zeros if no alpha) |
| `lsb` | `(1, 3, 256, 256)` | RGB LSBs |
| `palette` | `(1, 768)` | GIF palette or zeros |
| `palette_lsb` | `(1, 1, 256, 256)` | palette-index LSBs or zeros |
| `format_features` | `(1, 6)` | format / alpha stats |
| `content_features` | `(1, 6)` | LSB/alpha content stats |

## Forward outline (Trin im2col path)

1. **Streams** (each → `proj_dim=64` or 32 for small FCs):
   - `pixel_conv` → flatten → `pixel_proj`
   - `meta_conv` (on unsqueezed meta) → flatten → `meta_proj` × `meta_weight`
   - `alpha_conv` → flatten → `alpha_proj`
   - `lsb_conv` → flatten → `lsb_proj`
   - `palette_fc`
   - `palette_lsb_conv` → flatten → `palette_lsb_proj`
   - `format_features_fc` → 32
   - `content_features_fc` → 32
2. **Concat** all 8 streams → `(B, 448)`
3. **Gate**: `stream_gate` → sigmoid weights per stream; multiply each chunk
4. **Fusion**: Linear stack → 192 dims; split embedding (128) + cls (64)
5. **Heads**:
   - `stego_head` → 1 logit
   - `method_head` → 5 (alpha, palette, lsb.rgb, exif, raw)
   - `bit_order_head` → 3 (lsb-first, msb-first, none)
   - `embedding_head` → 64-D embedding from cls features

Outputs: `stego_logits, method_logits, bit_order_logits, method_id, method_probs, bit_order_id, embedding`.

## GGUF file layout (Trin-compatible)

Matches `WriteGGUFF32` in Trin `internal/weights/gguf.go`:

1. magic `u32 LE` `0x46554747` (`"GGUF"`)
2. version `u32 LE` = `3`
3. `n_tensors` `u64 LE`
4. `n_kv` `u64 LE` = `1 + len(meta)`
5. KV `general.alignment` (STRING key, type UINT32=4, value 32)
6. User meta KVs (type STRING=8)
7. Tensor infos in sorted name order: name, n_dims, dims, type=0 (F32), offset
8. Pad header to 32 bytes; write F32 payloads with per-tensor 32-byte padding

## Parity verification

```bash
python3 scripts/parity_starlight_gguf.py --gguf models/starlight.gguf --init-random
```

Exit 0 only if every exported tensor matches within `1e-6` abs error and shapes
agree. Optionally dumps `fixtures/parity_logits.json` from a sample image.

Trin (Go) smoke:

```bash
# from trin repo
go run /tmp/verify_starlight_gguf.go /path/to/starlight/models/starlight.gguf
```

## Notes

- Do **not** retrain here — export only.
- `*.pth` / `*.onnx` are gitignored; commit the name map + scripts + docs.
- A random-init GGUF is ~3–4 MB and may be committed for CI fixtures if useful;
  regenerate with `--init-random --seed 0` for a reproducible shape artifact.
