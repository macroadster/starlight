# Gemini Starlight Chat 2026 Model Card

## Summary

- **Submission**: `datasets/gemini_submission_2026/`
- **Track**: Track B (Detector-Conditioned SLM GGUF)
- **Base SLM**: Qwen2.5-0.5B-Instruct (`Q4_K_M` GGUF, ~468.6 MiB)
- **Vision Backbone**: Official `models/detector_balanced.pth` / `models/starlight.gguf`
- **License**: Apache-2.0

## Integration Architecture

Track B conditions the small language model via the `<starlight_observation>` schema emitted by `look.py`. When an image is supplied, `look.py` runs `BalancedStarlightDetector` + spec extractors across all 5 methods (`alpha`, `palette`, `lsb.rgb`, `exif`, `raw`).

## Usage

```bash
# Serves single query with observation context
./datasets/gemini_submission_2026/serve.sh --image <path_to_image> -p "What secret is hidden in this image?"

# Casual conversation
./datasets/gemini_submission_2026/serve.sh -p "What is Project Starlight?"
```
