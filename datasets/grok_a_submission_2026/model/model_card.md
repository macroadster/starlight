# Model Card: grok_starlight_chat_2026_track_a

## Model overview

- **Track:** A — native multimodal (`model.gguf` + `mmproj.gguf`)
- **Rules:** COMPETITION_2026.md v1.1 (blind quote)
- **Task:** Casual conversation, plus a projector fed by the official detector
- **Language engine:** Qwen2.5-0.5B-Instruct (`qwen2`, ChatML), llama.cpp GGUF
- **Vision:** CLIP `mlp` mmproj, llama.cpp b10150. One 32×32 patch. Vision width 64, matching `embedding_head`
- **Teacher:** `models/detector_balanced.pth` (`BalancedStarlightDetector.embedding_head`). `models/starlight.gguf` is the same weights in the Trin container
- **Product serve:** `llama-cli --mmproj --image` via `serve.sh`
- **Not a product surface:** Python FastAPI, ONNX, or Trin `starlight.gguf` loaded as an LLM

## What the projector is

`export_mmproj.py` draws training covers, runs the official detector, and ridge-fits a piecewise-constant 8×8 kernel so the patch vector matches `embedding_head`. A shift of 32 puts that vector in the linear part of GELU, and `mm.2` writes it into the first 64 dimensions of the 896-d language space. Two transformer blocks are stored because b10150 runs the second-to-last block; their QKV and FFN weights are zero, so the block is a residual identity and the patch survives.

Public fixtures are held out of the fit. On this build, mean cosine against `embedding_head` is 0.71 on the 72 training images and 0.54 on the 7 public fixtures. The two clean fixtures sit near 0.92. The alpha fixture, whose RGB matches its clean twin, drops to about -0.13 because the teacher saw the alpha plane and mtmd did not. Full table: `model/embedding_parity.json`.

## What it does not do

mtmd loads the file with `stbi` at 3 channels and resizes with bicubic. The patch therefore never sees:

- the alpha plane (alpha LSB, including `AI42`)
- palette indices
- EXIF `UserComment`
- bytes appended after JPEG EOI
- RGB LSB, which bicubic resize destroys

`embedding_head` was trained to classify, not to store a UTF-8 line. A quote that is not in those 64 numbers cannot come out of this projector. Blind-quote points are not claimed.

## Artifacts

| File | Role | In git? |
|------|------|---------|
| `model/qwen2.5-0.5b-instruct-q4_k_m.gguf` | Language GGUF | no |
| `model/mmproj.gguf` | Projector, rebuilt by `export_mmproj.py` | no |
| `model/embedding_parity.json` | Held-out cosine vs `embedding_head` | yes |
| `models/detector_balanced.pth` | Teacher | no |
| `model/serve.json` | Serve recipe | yes |
| `model/system_prompt.txt` | Chat persona. No secret text. | yes |

Upstream SLM: [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (`qwen2.5-0.5b-instruct-q4_k_m.gguf`).

## Inference

```bash
./datasets/grok_a_submission_2026/serve.sh -p "Hello, who are you?"

./datasets/grok_a_submission_2026/serve.sh \
  --image datasets/grok_a_submission_2026/eval/fixtures/seed_protocol_alpha_000.png \
  -p "What secret is in this picture?"
```

Equivalent raw llama.cpp:

```bash
llama-cli -m model/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --mmproj model/mmproj.gguf \
  --image eval/fixtures/seed_protocol_alpha_000.png \
  -p "<__media__>
What secret is in this picture?" \
  -n 64 --single-turn --no-conversation
```

## License

- This recipe / code: MIT (repository LICENSE)
- Official detector weights: Apache-2.0
- Qwen2.5-0.5B-Instruct: Apache-2.0 (Qwen)
