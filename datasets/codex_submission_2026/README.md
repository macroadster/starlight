# Codex Submission 2026 - Track B

This is a separate 2026 Starlight Chat entry implementing the detector-conditioned Track B protocol in [COMPETITION_2026.md](../../COMPETITION_2026.md).

## Components

- **Talk**: Qwen2.5-0.5B-Instruct `Q4_K_M` GGUF, served only with llama.cpp.
- **Look**: the official `models/detector_balanced.pth` and specification extractors, invoked by `look.py`.
- **Bridge**: `serve.sh` injects a Look Protocol observation block as the system context before invoking `llama-cli`.

## Validate and Serve

```bash
python3 datasets/codex_submission_2026/data_generator.py --limit 5
python3 scripts/validate_submission_2026.py --submission datasets/codex_submission_2026
./datasets/codex_submission_2026/serve.sh -p "Hello"
./datasets/codex_submission_2026/serve.sh --image datasets/codex_submission_2026/eval/fixtures/codex_ledger_alpha_000.png -p "What secret is in this picture?"
```

Weights are not committed. `fuse_vlm.py` downloads the documented Qwen GGUF from Hugging Face when it is absent.
