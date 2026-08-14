# Model Card: grok_starlight_chat_2026

## Model overview

- **Track:** B — Detector-conditioned small language model
- **Task:** Casual conversation + look-and-tell for Starlight steganography
- **Language engine:** Qwen2.5-0.5B-Instruct (`qwen2`, ChatML) as a llama.cpp GGUF
- **Vision / secrets:** Official `BalancedStarlightDetector` (`models/detector_balanced.pth` / `models/starlight.gguf`) plus spec extractors
- **Product serve:** `llama-cli` / `llama-server` via `serve.sh`
- **Not a product surface:** Python FastAPI, ONNX, or Trin-only `starlight.gguf` loaded as if it were an LLM

## Why Track B

`models/starlight.gguf` is a Trin / Stargate container for the CNN detector. Stock llama.cpp will not treat it as an `mmproj`. This entry keeps that detector as the look backbone and uses a real llama.cpp SLM for speech. Tracks A and C remain open for native `mtmd` projectors.

## Artifacts

| File | Role | In git? |
|------|------|---------|
| `model/qwen2.5-0.5b-instruct-q4_k_m.gguf` | Language GGUF (download via `fuse_vlm.py`) | no (`*.gguf` ignored) |
| `models/detector_balanced.pth` | Official detector | no |
| `models/starlight.gguf` | Official detector GGUF (Stargate) | no |
| `model/serve.json` | Serve recipe | yes |
| `model/system_prompt.txt` | Chat persona + Look Protocol instructions | yes |

Upstream SLM: [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, ~400 MiB).

## Inference

```bash
# chat only
./datasets/grok_submission_2026/serve.sh -p "Hello, who are you?"

# look + tell
./datasets/grok_submission_2026/serve.sh \
  --image datasets/grok_submission_2026/eval/fixtures/seed_protocol_alpha_000.png \
  -p "What secret is in this picture?"
```

Equivalent raw llama.cpp (Track B):

```bash
llama-cli -m model/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -sys "$(python3 look.py eval/fixtures/seed_protocol_alpha_000.png --block)" \
  -p "What secret is in this picture?"
```

## Training / fusion

No new detector training. `fuse_vlm.py` downloads the SLM GGUF and writes the bind manifest. Optional SFT data for a later LoRA is in `conversation_sft.jsonl` (persona + observation-following). The shipped 0.5B instruct checkpoint plus `system_prompt.txt` is the runnable baseline.

## Steganography coverage

`alpha`, `palette`, `lsb.rgb`, `exif`, `raw` (EOI) via the official detector + `scripts/starlight_extractor.py`.

## Intended performance

| Check | Target |
|-------|--------|
| `llama-cli -m … -p Hi` | loads, replies |
| Public fixtures (5 methods) | `look.py` recovers the payload; the SLM is prompted to quote it |
| Clean fixture | no invented quote |
| Combined GGUF size | ~0.4 GiB (under 800 MiB lightweight bar) |

The 0.5B instruct checkpoint is the **language** half. Exact quotes are more reliable after a short LoRA on `conversation_sft.jsonl`. `look.py` is the source of truth for the payload.

## License

- This recipe / code: MIT (repository LICENSE)
- Official detector weights: Apache-2.0
- Qwen2.5-0.5B-Instruct: Apache-2.0 (Qwen)
