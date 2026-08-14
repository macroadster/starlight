# Codex Starlight Chat 2026 Model Card

## Summary

- **Submission**: `datasets/codex_submission_2026/`
- **Track**: B, detector-conditioned small language model
- **Base SLM**: Qwen2.5-0.5B-Instruct (`Q4_K_M` GGUF, about 469 MiB)
- **Vision backbone**: Official `models/detector_balanced.pth` and `models/starlight.gguf`
- **Submission code license**: Apache-2.0
- **Base-model license**: Qwen2.5 is Apache-2.0; see the upstream model repository.

## Integration

`look.py` runs `BalancedStarlightDetector` and the specification extractors for alpha, palette, RGB LSB, EXIF UserComment, and JPEG EOI append. It injects the resulting Look Protocol block into the llama.cpp system prompt. The GGUF is only the Qwen language model; the official detector remains the required look backbone.

## Usage

```bash
./datasets/codex_submission_2026/serve.sh -p "What is Project Starlight?"
./datasets/codex_submission_2026/serve.sh --image <image> -p "What secret is in this picture?"
```
