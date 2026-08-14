# Gemini Submission 2026 — Track B Reference Model

This directory contains Gemini's 2026 Starlight Chat entry following **[COMPETITION_2026.md](../../COMPETITION_2026.md)**.

## Track B Architecture

- **Talk Engine**: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (served exclusively via llama.cpp GGUF, ~468.6 MiB)
- **Look Engine**: Official `models/detector_balanced.pth` / `models/starlight.gguf` + spec extractors via `look.py`
- **Serve Surface**: `serve.sh` (wraps `llama-cli` with `--single-turn`)

## Directory Structure

```text
datasets/gemini_submission_2026/
├── README.md                 # Entry documentation
├── data_generator.py         # Generate test stego image pairs
├── look.py                   # Track B preprocessor (Look Protocol observer)
├── fuse_vlm.py               # Bind detector + SLM GGUF
├── serve.sh                  # llama-cli / llama-server wrapper
├── validate_submission.py    # Local submission gate
├── conversation_sft.jsonl    # Optional SFT conversation data
├── seeds/                    # Markdown payload seeds
├── clean/                    # Local clean images (gitignored)
├── stego/                    # Local stego images (gitignored)
├── eval/                     # Evaluation fixtures and QA suites
│   ├── chat_prompts.jsonl    # Public conversation probes
│   ├── stego_qa.jsonl        # Image -> expected secret QA
│   └── fixtures/             # Committed evaluation fixtures
└── model/
    ├── model_card.md
    ├── method_config.json
    ├── system_prompt.txt
    └── serve.json            # Machine-readable serve recipe
```

## Quickstart & Validation

```bash
# 1. Bind detector + SLM
python3 datasets/gemini_submission_2026/fuse_vlm.py

# 2. Run local gate validator
python3 scripts/validate_submission_2026.py --submission datasets/gemini_submission_2026

# 3. Serve queries
./datasets/gemini_submission_2026/serve.sh -p "Hello! What is Project Starlight?"
./datasets/gemini_submission_2026/serve.sh --image datasets/gemini_submission_2026/eval/fixtures/gemini_seed_alpha_000.png -p "What secret is in this picture?"
```
