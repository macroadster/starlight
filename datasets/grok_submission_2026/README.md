# Grok Submission 2026 — Starlight Chat (Track B)

Reference entry for [COMPETITION_2026.md](../../COMPETITION_2026.md).

A **0.5B instruct GGUF** talks. The official **`BalancedStarlightDetector`** looks. `serve.sh` is the product command: stock `llama-cli`, no Python web server.

## Layout

```
grok_submission_2026/
├── look.py                 # detector + extractors → Look Protocol block
├── fuse_vlm.py             # download SLM GGUF, write bind manifest
├── serve.sh                # llama-cli wrapper
├── data_generator.py       # five-method eval pairs
├── conversation_sft.jsonl  # optional later SFT
├── seeds/                  # payloads
├── eval/                   # chat probes, stego QA, fixtures
└── model/
    ├── serve.json
    ├── system_prompt.txt
    ├── model_card.md
    └── method_config.json
```

## Serve

```bash
cd datasets/grok_submission_2026
python3 data_generator.py --limit 5
python3 fuse_vlm.py
chmod +x serve.sh
./serve.sh -p "Hello, who are you?"
./serve.sh --image eval/fixtures/seed_protocol_alpha_000.png \
  -p "What secret is in this picture?"
```

From the repo root:

```bash
python3 scripts/validate_submission_2026.py \
  --submission datasets/grok_submission_2026
```

## Why this architecture

`models/starlight.gguf` is the Stargate / Trin detector container. llama.cpp will not load it as an LLM or as a CLIP `mmproj`. Track B keeps that detector as the look backbone and uses a real llama.cpp SLM for speech. That is the smallest package that satisfies “standalone GGUF + llama.cpp + tell me the secret.”

Tracks A (native `mmproj`) and C (distilled VLM) are invited to beat this on native `--image` while keeping exact payload quotes.

## License

Recipe: MIT. Detector: Apache-2.0. Qwen2.5-0.5B-Instruct: Apache-2.0.
