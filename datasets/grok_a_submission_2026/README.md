# Grok Submission 2026 — Track A

Native multimodal entry for [COMPETITION_2026.md](../../COMPETITION_2026.md) **v1.1**.

A **0.5B instruct GGUF** talks. A **CLIP `mlp` mmproj** looks. `serve.sh` is the product command: stock `llama-cli --mmproj --image` on llama.cpp **b10150**.

The projector's single 32×32 patch is a ridge fit to `BalancedStarlightDetector.embedding_head` (64-d). Those 64 numbers are copied into the leading dims of Qwen2.5-0.5B's token. The prompt does not contain the secret, and neither do the weights.

This entry does not claim blind-quote points. llama.cpp's vision path decodes to RGB and resizes, so alpha, palette indices, EXIF, and JPEG tail bytes never reach the patch. A 32×32 bicubic grid also wipes LSB. `embedding_head` is a classification vector, not the payload. Held-out cosine is in `model/embedding_parity.json`.

## Layout

```
grok_a_submission_2026/
├── export_mmproj.py        # fit embedding_head, write mmproj.gguf
├── fuse_vlm.py             # link or download the SLM, then export
├── serve.sh                # llama-cli --mmproj [--image]
├── data_generator.py       # five-method eval pairs
├── conversation_sft.jsonl  # persona only, no fixture secrets
├── seeds/
├── eval/
└── model/
    ├── serve.json
    ├── system_prompt.txt
    ├── model_card.md
    ├── method_config.json
    └── embedding_parity.json
```

## Serve

```bash
cd datasets/grok_a_submission_2026
python3 fuse_vlm.py
chmod +x serve.sh
./serve.sh -p "Hello, who are you?"
./serve.sh --image eval/fixtures/seed_protocol_alpha_000.png \
  -p "What secret is in this picture?"
```

From the repo root:

```bash
python3 scripts/validate_submission_2026.py \
  --submission datasets/grok_a_submission_2026
```

`*.gguf` stays gitignored. The language file is the same Qwen2.5-0.5B-Instruct Q4_K_M as the Track B reference. `fuse_vlm.py` hard-links it when that file is already on disk.

## License

Recipe: MIT. Detector: Apache-2.0. Qwen2.5-0.5B-Instruct: Apache-2.0.
