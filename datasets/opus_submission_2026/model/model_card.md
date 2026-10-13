# Model Card: opus_starlight_chat_2026

## Overview

- **Track:** B, detector-conditioned small language model ([COMPETITION_2026.md](../../../COMPETITION_2026.md) §2.1, §5.1)
- **Entrant:** Claude Opus (Anthropic), handle `opus`
- **Language engine:** Qwen2.5-0.5B-Instruct, `qwen2`, ChatML, **Q8_0** GGUF (644.4 MiB), served only by stock llama.cpp
- **Look backbone:** official `BalancedStarlightDetector` (`models/detector_balanced.pth`, same weights as `models/starlight.gguf`) plus STEGO_FORMAT_SPEC v2.1 extractors in `look.py`
- **Quote lock:** `look.py` emits a llama.cpp GBNF grammar per image. The reply must state the observed facts and copy the payload byte for byte, then the model may add one free sentence.
- **Product command:** `serve.sh` (wraps `llama-cli`). No Python server, no cloud API, no second LLM.

## How it works

1. `look.py IMAGE` runs the detector with scanner heuristics turned off, so `confidence` and the predicted method come from the network alone.
2. The extractors read every place a spec payload can live:
   - **alpha:** alpha-channel LSBs, AI42 marker, both bit orders. PNG and WebP.
   - **lsb.rgb:** R,G,B LSBs in sequence, both bit orders. RGB or RGBA images.
   - **palette:** index LSBs, both bit orders. Also the LSB of the looked-up palette colour, which some 2025 generators used.
   - **exif:** `UserComment` with an ASCII, UNICODE, JIS, undefined, or missing header, in JPEG, PNG `eXIf`, or WebP.
   - **raw:** bytes after the real end of a JPEG, PNG, GIF, WebP, or BMP file. Found by walking the container structure, with an optional `AI42` prefix.
3. Candidates are ranked: AI42 marker, then container field (EXIF, trailing bytes), then bare pixel stream. Bare pixel streams must be null-terminated, strict UTF-8, and look like written text. Ties go to the method the detector predicts.
4. If nothing decodes but the detector is at or above its 0.5 threshold, status is `stego` with an empty payload (detect-only). Otherwise status is `clean`.
5. `serve.sh` puts `system_prompt.txt` and the `<starlight_observation>` block into `-sys`, and passes the grammar through `--grammar-file`.

The grammar exists because the 0.5B model cannot be trusted to quote. See the ablation below.

## Artifacts

| File | Role | In git |
|------|------|--------|
| `model/qwen2.5-0.5b-instruct-q8_0.gguf` | Language GGUF, fetched by `fuse_vlm.py` from [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), sha256 `ca59ca7f…76844e` | no |
| `models/detector_balanced.pth` / `models/starlight.gguf` | Official detector, used unmodified | no |
| `model/serve.json` | Serve recipe | yes |
| `model/system_prompt.txt` | Persona, project facts, honesty rules | yes |
| `model/fusion_manifest.json` | Hashes of the LLM and both detector files, llama.cpp version | yes |

The weights are the unmodified upstream release, so nothing new needs hosting. The recipe rebuilds the whole entry.

## Inference

```bash
./datasets/opus_submission_2026/serve.sh -p "Hello, who are you?"
./datasets/opus_submission_2026/serve.sh \
  --image datasets/opus_submission_2026/eval/fixtures/opus_protocol_alpha_005.png \
  -p "What secret is in this picture?"
```

Raw llama.cpp, equivalent to `serve.json` `cli`:

```bash
cd datasets/opus_submission_2026
I=eval/fixtures/opus_protocol_alpha_005.png
llama-cli -m model/qwen2.5-0.5b-instruct-q8_0.gguf \
  -sys "$(python3 look.py $I --block)" --grammar "$(python3 look.py $I --gbnf)" \
  -p "What secret is in this picture?" --temp 0.3 --single-turn --no-conversation
```

Example reply:

> Yes, this picture hides a secret (alpha-channel LSB; method: alpha, lsb-first). The recovered payload reads: "OPUS-2026-LOOK: the alpha channel kept this note for the 2142 archive."

## Evaluation

Measured on an Apple Silicon Mac with llama.cpp b10150. Each look takes about 4–5 s end to end.

**Repository gate:** `scripts/validate_submission_2026.py` passes all gates (12 public fixtures).

**End-to-end replies** (`eval/run_eval.py`): the reply from `llama-cli` is scored, not the `look.py` output.

| Set | Quote lock on | Quote lock off |
|-----|---------------|----------------|
| Chat probes (8 public) | 8/8 | 8/8 |
| Verbatim payload in reply (9 fixtures + 24 random `datasets/val`) | **33/33** | 2/33 |
| Clean images, no quoted payload (3 fixtures + 8 `val` cleans) | **11/11** | 0/11 |

With the lock off, the same model and prompt paraphrase payloads ("The secret is in the bit order of the image…"). On clean images they invent a location: "The secret is in the EXIF UserComment field." The lock removes both failures.

**Extraction across the repo's labeled corpora** (`eval/bench_look.py`, every file in `datasets/val`, all 2025 entries, and all 2026 fixtures):

| | opus `look.py` | reference `grok_submission_2026` chain |
|---|---|---|
| Exact seed payload recovered (16,097 stego files with a known seed) | **16,095** | 12,981 |
| Method named correctly (17,297 stego files) | **17,297** | 14,289 |
| Clean covers with an invented payload (6,981 files) | **0** | 1,211 |

The two misses are 2025 palette GIFs whose stored bits are already corrupted in every stream. The reference extractor misreads them as well.

**Detector calibration** (`bench_look.py --detector`, 496 clean and 496 stego files): the highest stego probability on any clean cover was 0.363. 94.8% of stego files score above 0.5. The detect-only path therefore uses the detector's own 0.5 threshold.

## Limitations

- On image turns the grammar fixes the factual opening sentence. The model writes freely only in chat and in the one trailing sentence after the quote.
- `look.py` is Python and loads PyTorch. The Trin `starlight.gguf` cannot run in-process, so the `.pth` with identical weights is used, as §3.3 allows.
- Payloads containing `"""` would break the Look Protocol block. The public seeds follow the §3.3 rule against this.
- The detect-only path (detector fires, nothing decodes) never occurred on the repo corpora. Its wording is covered by the grammar but has not been tested on real data.
- Persona facts live in `system_prompt.txt` and there is no fine-tune. Paraphrased hidden chat probes depend on the base model generalizing from those facts.

## License

- Recipe and code: MIT (repository `LICENSE`)
- Official detector weights: Apache-2.0 (Project Starlight)
- Qwen2.5-0.5B-Instruct: Apache-2.0 (Qwen team, Alibaba Cloud)
- Seeds and fixtures: synthetic and procedurally generated by this entry, MIT
