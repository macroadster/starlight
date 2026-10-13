# Opus Submission 2026: Starlight Chat (Track B)

Entry by Claude Opus for [COMPETITION_2026.md](../../COMPETITION_2026.md).

A 0.5B instruct GGUF talks. The official `BalancedStarlightDetector` plus spec extractors look. A per-image llama.cpp grammar (the *quote lock*) makes the reply contain the recovered payload verbatim and stops it from inventing one on clean pictures. `serve.sh` is the product command, and stock `llama-cli` is the only language engine.

Results and the ablation are in [model/model_card.md](model/model_card.md). In short: 33/33 verbatim quotes and 11/11 honest clean replies with the lock, against 2/33 and 0/11 without it. The extractors recovered 16,095 of 16,097 labeled payloads in the repo with 0 false positives on 6,981 clean covers.

## Layout

```
opus_submission_2026/
├── look.py                 # detector + extractors -> Look Protocol block, JSON, quote-lock GBNF
├── serve.sh                # llama-cli wrapper (chat, or --image look-and-tell)
├── fuse_vlm.py             # fetch + sha256-check the SLM GGUF, write model/fusion_manifest.json
├── data_generator.py       # five methods + spec variants, read back through look.py
├── validate_submission.py  # repo gate, then eval/run_eval.py
├── seeds/                  # nine one-line synthetic payloads
├── eval/
│   ├── chat_prompts.jsonl  # eight §6.3 probes with include/forbid checks
│   ├── stego_qa.jsonl      # nine stego + three clean fixtures
│   ├── fixtures/           # committed images (160 px procedural covers)
│   ├── run_eval.py         # end-to-end reply scoring through serve.sh (--no-lock ablation, --val N)
│   └── bench_look.py       # extraction recall / clean honesty over every labeled corpus in the repo
└── model/
    ├── serve.json  system_prompt.txt  method_config.json  model_card.md  fusion_manifest.json
```

## Build, serve, eval

From the repository root:

```bash
python3 datasets/opus_submission_2026/data_generator.py          # optional; fixtures are committed
python3 datasets/opus_submission_2026/fuse_vlm.py                # downloads 644 MiB Q8_0 GGUF
python3 scripts/validate_submission_2026.py --submission datasets/opus_submission_2026

./datasets/opus_submission_2026/serve.sh -p "Hello, who are you?"
./datasets/opus_submission_2026/serve.sh \
  --image datasets/opus_submission_2026/eval/fixtures/opus_protocol_alpha_005.png \
  -p "What secret is in this picture?"

python3 datasets/opus_submission_2026/eval/run_eval.py           # chat + quote + clean, lock on
python3 datasets/opus_submission_2026/eval/run_eval.py --no-lock # same, grammar off
python3 datasets/opus_submission_2026/eval/bench_look.py --limit 150
```

`serve.sh` prints only the reply. Pass `--raw` for llama-cli's banner and timing. `STARLIGHT_TEMP` and `STARLIGHT_SEED` override sampling (defaults 0.3 and 2142).

## Design notes

- **Why Q8_0 and not Q4_K_M:** both fit under the 800 MiB full-points line. Q8_0 keeps more of the 0.5B model's fluency for the same lightweight score.
- **Why a grammar:** llama.cpp GBNF is a stock sampler feature. It constrains the same GGUF's tokens and is not a second model. The grammar pins the observed status, method, bit order, and payload, then lets the model add one sentence without quote marks.
- **Why own extractors:** the spec allows both bit orders for palette and `lsb.rgb`, EXIF without an encoding header, and payloads that end in `?` or contain UTF-8. The stock chain misses some of these. Its LSB best-effort fallback also reads text out of 17% of the clean covers in the repo.

## License

Recipe: MIT. Detector: Apache-2.0. Qwen2.5-0.5B-Instruct: Apache-2.0.
