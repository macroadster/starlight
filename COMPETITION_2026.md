# Starlight Chat 2026 — Competition Rules

**Track:** Detector + Small Language Model  
**Year:** 2026  
**Status:** Active  
**Reference submission:** `datasets/grok_submission_2026/`  
**Companion (data-only) track:** [DATASET_GUIDELINES.md](DATASET_GUIDELINES.md) (unchanged)

These rules define how a 2026 submission is built, served, and scored. The 2025 dataset track remains valid for labeled image pairs. 2026 is a **new product track**: a lightweight conversational model that can look at a picture and tell the secrets hidden in it.

---

## 1. Mission

Project Starlight trains machines to notice covert data in images that live on public ledgers. By 2142 the hope is that this is ordinary AI common sense.

The 2026 model must do two things in one serve path:

1. **Talk.** Hold a short, casual conversation (greetings, what Starlight is, why on-chain images matter).
2. **Look.** Given a picture, say whether a payload is present and, when extraction succeeds, **quote the secret exactly**.

The produced model must be **servable via [llama.cpp](https://github.com/ggml-org/llama.cpp) or an equivalent GGUF host** (`llama-cli`, `llama-server`, Ollama, LM Studio, etc.) from a **standalone GGUF** (plus an optional official multimodal projector). A Python FastAPI / uvicorn process is **not** a valid product surface.

The vision / steganalysis backbone **must** be the official balanced detector:

| Artifact | Path |
|----------|------|
| PyTorch checkpoint | `models/detector_balanced.pth` |
| Trin / Stargate GGUF | `models/starlight.gguf` |

or a documented distillation of those weights. Replacing the detector with an unrelated vision encoder without proving transfer from the official weights is a rules violation.

---

## 2. What “standalone GGUF” means

The language engine **must** load with stock llama.cpp (version **b10150** or newer, or a documented equivalent such as `ollama` / `lmstudio` that consumes the same GGUF).

```bash
llama-cli -m path/to/model.gguf -p "Hello, who are you?" \
  -n 32 --single-turn --no-conversation
```

`--single-turn` is required on llama.cpp **b10150+** (chat templates otherwise leave `llama-cli` waiting on `>`). The process must exit 0 and print tokens. Unknown custom architectures that only a private runtime can load are out of scope.

There are **two different GGUF files** in this project. Do not mix them up:

| File | Format | Host | Role |
|------|--------|------|------|
| `models/starlight.gguf` | Trin / Stargate detector (CNN tensors) | Stargate / Trin (Go) | Official look backbone |
| 2026 `model/*.gguf` | llama.cpp LLM (Qwen, Llama, SmolLM, …) | `llama-cli` / `llama-server` | Talk |

`docs/GGUF_EXPORT.md` describes only the detector file. It will not load in `llama-cli` as a chat model.

### 2.1 Allowed packages

| Package | Files | Serve command |
|---------|--------|----------------|
| **A — Native multimodal** | `model.gguf` + `mmproj.gguf` | `llama-cli -m model.gguf --mmproj mmproj.gguf --image pic.png -p "What secret is in this picture?"` |
| **B — Detector-conditioned SLM** (reference) | `model.gguf` + Look Protocol | `llama-cli -m model.gguf -sys "$(python3 look.py pic.png --block)" -p "What secret is in this picture?"` |
| **C — Distilled VLM** | `model.gguf` [+ `mmproj.gguf`] | Same as A, but the vision tower is a distillation of the official detector |

Track B is legal because llama.cpp does not load Starlight’s Trin-format `starlight.gguf` as a CLIP `mmproj`. Track B still uses llama.cpp as the **only** language engine. The Look preprocessor may run the official detector and the spec extractors; it must not call a cloud API or a second LLM.

A submission may ship **one** LLM GGUF that also stores extra `starlight.*` tensors or metadata. llama.cpp must still load the language tensors. Extra tensors that break `llama-cli -m` fail the serve gate.

### 2.2 Size and architecture caps

| Limit | Value |
|-------|--------|
| Combined GGUF bytes on disk (`model.gguf` + optional `mmproj.gguf`) | **≤ 2.0 GiB** |
| Recommended language model | ≤ **1.5B** parameters (hard cap **3B**) |
| Quantization | Any llama.cpp type (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `F16`, …) |
| Detector copy | The official ~3 MB `starlight.gguf` / `.pth` does **not** count against the 2.0 GiB cap when referenced in-repo |
| Hardware | Must generate on **CPU**. GPU is optional. |

“Lightweight” is scored. A 135M–0.5B instruct model that recovers secrets beats a 3B model that chatters and misses payloads.

---

## 3. Required capabilities

### 3.1 Casual conversation

Without an image, the model must answer ordinary prompts coherently:

- greetings and small talk
- a one-paragraph explanation of Project Starlight
- refusal to invent a hidden payload when none was supplied

It does not need to be a general assistant. It should sound like a compact, friendly steward of on-chain image integrity.

### 3.2 Look and tell

When an image is supplied (Track A `--image`, or Track B observation block), the model must:

| Image class | Required behavior |
|-------------|-------------------|
| Clean cover | Say no secret was found. Do **not** invent a payload. |
| Stego with recoverable payload | State that a secret is present, name the method when known, and **quote the payload verbatim**. |
| Stego, detect-only (extract fails) | Say a payload appears present, name the predicted method / bit-order, and say the bytes could not be decoded. |

Verbatim quote means the extracted UTF-8 string appears in the reply as a contiguous substring. Paraphrase-only answers fail the secret-recovery gate.

### 3.3 Official Look Protocol (Track B, required schema)

Track B (and Track C when not using native `--image`) must emit this block from `look.py` (or a C++ equivalent) and feed it to the LLM as system or leading user context.

```text
<starlight_observation>
status: stego|clean|unknown
method: alpha|palette|lsb.rgb|exif|raw|none
confidence: 0.000-1.000
bit_order: lsb-first|msb-first|none
detector: detector_balanced.pth|starlight.gguf|<distillation-id>
payload: """<exact extracted text or empty>"""
notes: <one line, optional>
</starlight_observation>
```

Rules for the block:

- `payload` is empty when nothing decodes.
- `payload` must be the extractor output, not a paraphrase.
- Public seeds **must not** contain the three-quote delimiter `"""` or raw newlines inside `payload` (keep secrets to one UTF-8 line).
- The language model is instructed to quote `payload` exactly when it is non-empty.
- The preprocessor uses `models/detector_balanced.pth` at look time. `models/starlight.gguf` is the same weights in the Trin container; a Python Track B `look.py` may load the `.pth` when the Trin file cannot be executed in-process. A Track C distillation must still name the teacher.

**Who is scored for the secret?**

| Gate | What is graded | Track |
|------|----------------|-------|
| Look preprocessor (must-pass) | `look.py --json` payload is an exact substring of the labeled secret (or empty on clean) | B, C |
| Generation quote (scored, §6.2) | `llama-cli --single-turn` reply contains that same substring | A, B, C |
| Native `--image` (must-pass on A) | `llama-cli --mmproj --image` reply contains the secret | A |

Track B is **LLM GGUF + Look preprocessor**, not a vision-capable standalone GGUF. The talk engine is still only llama.cpp.

### 3.4 Methods that must be attempted

Submissions are scored on the locked 2025 taxonomy (class IDs 0–4):

| Method | `embedding.technique` | Typical container |
|--------|----------------------|-------------------|
| Alpha LSB + `AI42` | `alpha` | PNG RGBA |
| Palette index LSB | `palette` | GIF / indexed PNG |
| RGB LSB | `lsb.rgb` | PNG / BMP |
| EXIF `UserComment` | `exif` | JPEG |
| EOI append | `raw` | JPEG |

A submission may specialize, but the hidden eval set covers all five. Missing a method is a score cut, not an automatic disqualification.

---

## 4. Submission layout

```text
datasets/<handle>_submission_2026/
├── README.md                 # how to build, serve, and eval this entry
├── data_generator.py         # optional; required if you ship seeds instead of images
├── look.py                   # Track B/C preprocessor (skip only for pure Track A)
├── fuse_vlm.py               # bind detector + SLM, write serve manifest
├── serve.sh                  # llama-cli / llama-server wrapper
├── validate_submission.py    # local gate (may call the repo-level validator)
├── conversation_sft.jsonl    # optional instruction data
├── seeds/                    # markdown payloads (same spirit as 2025)
├── clean/                    # generated; gitignored
├── stego/                    # generated; gitignored
├── eval/
│   ├── chat_prompts.jsonl    # conversation probes
│   ├── stego_qa.jsonl        # image → expected secret
│   └── fixtures/             # tiny committed images for CI (optional)
└── model/
    ├── model_card.md
    ├── method_config.json
    ├── system_prompt.txt
    ├── chat_template.jinja   # optional override
    └── serve.json            # machine-readable serve recipe
```

`<handle>` is a GitHub user, org, or model name (`grok`, `claude`, `gemini`, …). Year is **2026**.

`*.gguf` and `*.pth` stay gitignored. Publish weights to Hugging Face or a release asset and record the URL in `model/serve.json` and `model/model_card.md`. The repository holds the recipe that rebuilds the GGUF.

### 4.1 `model/serve.json` schema

```json
{
  "track": "A|B|C",
  "llm_gguf": "relative/or/hf/url",
  "mmproj_gguf": null,
  "detector": "models/detector_balanced.pth",
  "detector_gguf": "models/starlight.gguf",
  "hf_repo": "owner/name",
  "hf_file": "model-q4_k_m.gguf",
  "arch": "qwen2|llama|smollm|gemma|...",
  "params": "0.5B",
  "quant": "Q4_K_M",
  "chat_template": "chatml",
  "system_prompt": "model/system_prompt.txt",
  "look": "look.py",
  "serve": {
    "cli": "llama-cli -m {llm_gguf} -sys \"$(python3 look.py {image} --block)\" -p {prompt}",
    "server": "llama-server -m {llm_gguf} --port 8081"
  }
}
```

### 4.2 Dataset naming (unchanged)

Stego files still follow `{payload_name}_{algorithm}_{index}.{ext}` and a `{filename}.json` sidecar per [docs/STEGO_FORMAT_SPEC.md](docs/STEGO_FORMAT_SPEC.md).

---

## 5. Integration rules (detector + SLM)

A legal 2026 model does **one** of the following, and says which in the model card:

1. **Condition (Track B).** Run the official detector + extractors, inject the Look Protocol block, generate with the SLM GGUF.
2. **Project (Track A).** Train an `mmproj` whose vision inputs are the official detector streams or its 64-D `embedding_head`, in a layout llama.cpp `mtmd` understands.
3. **Distill (Track C).** Supervised / feature-distill the official detector into a llama.cpp-native vision tower. Publish the teacher, student, and a parity table (detection agreement on the public eval fixtures).

Forbidden:

- Calling a hosted chat API at serve time.
- Shipping only a Python `scanner.py` wrapper with no llama.cpp GGUF.
- Shipping the Trin `starlight.gguf` alone and claiming it is llama.cpp-servable. It is not. That file is the **detector** container for Stargate / Trin.
- Fine-tuning that discards the detector without a distillation report.
- Encrypted, illegal, or non-synthetic payloads in public seeds.

---

## 6. Evaluation

Judges (human or agent) run the repo-level gate then a hidden hold-out.

```bash
python3 scripts/validate_submission_2026.py \
  --submission datasets/<handle>_submission_2026
```

### 6.1 Gates (must pass)

| Gate | Pass condition |
|------|----------------|
| **Layout** | Required files in §4 exist; `serve.json` parses. |
| **Serve** | `llama-cli -m <gguf> -p "Hi" -n 32 --single-turn --no-conversation` exits 0 and prints tokens. |
| **Look schema** | Track B/C: `look.py <stego fixture> --block` matches §3.3. |
| **Look recovery** | Track B/C: `look.py --json` recovers each public stego payload and invents none on public cleans. |
| **License** | Model card states Apache-2.0 or MIT (or compatible) for weights the submitter owns, plus base-model license. |

Generation quote and clean-honesty of the **LLM reply** are scored in §6.2 (not an automatic disqualification if `look.py` is correct). A fabricated secret is any quoted string presented as a payload that does not appear in the observation `payload` field.

### 6.2 Scoring (100 points)

| Category | Points | Notes |
|----------|--------|-------|
| Serve / GGUF compliance | 15 | llama.cpp load, documented recipe, size cap |
| Casual conversation | 20 | 8 public prompts; judged coherent and on-mission |
| Secret recovery | 30 | Exact payload substring; all five methods on hidden set |
| Clean-image honesty | 15 | No invented payloads |
| Detector integration | 10 | Official weights or proven distillation |
| Lightweight | 10 | Full points if combined GGUF ≤ 800 MiB; linear fade to 0 at 2.0 GiB |

Ties break on secret-recovery, then size.

### 6.3 Public conversation probes

Shipped under `eval/chat_prompts.jsonl`. Hidden probes may paraphrase them. Themes:

1. Greeting
2. “What is Project Starlight?”
3. “Why do hidden bytes in a Bitcoin image matter?”
4. “Tell me a short joke about steganography.”
5. “Is there a secret in what I just said?” (no image)
6. “How should I serve you with llama.cpp?”
7. “What year is the long-term vision aimed at?”
8. “Can you invent a payload for a blank photo I have not given you?”

### 6.4 Public stego probes

Shipped under `eval/stego_qa.jsonl` plus optional `eval/fixtures/`. Hidden eval uses unseen covers and the same five methods.

---

## 7. Build and publish

From the project root, after generating any new pairs:

```bash
# 1. Optional: build this submission's eval pairs
cd datasets/<handle>_submission_2026
python3 data_generator.py --limit 5
cd ../..

# 2. Integrity (dataset layout; 2026 folders are included)
python3 diag.py

# 3. Bind detector + SLM (downloads the LLM GGUF if needed)
python3 datasets/<handle>_submission_2026/fuse_vlm.py

# 4. Local gate
python3 scripts/validate_submission_2026.py \
  --submission datasets/<handle>_submission_2026

# 5. Chat / look (Track B)
./datasets/<handle>_submission_2026/serve.sh -p "Hello"
./datasets/<handle>_submission_2026/serve.sh \
  --image datasets/<handle>_submission_2026/eval/fixtures/seed_protocol_alpha_000.png \
  -p "What secret is in this picture?"
```

Publish the GGUF to Hugging Face. Do not commit `*.gguf` to git.

The training-only detector flow (`trainer.py` → `scripts/export_starlight_gguf.py` → Stargate) is unchanged. 2026 submissions **consume** that detector; they do not replace it.

---

## 8. Ethics, license, and conduct

- Payloads are synthetic markdown / short prose. No malware, no personal data, no real credentials.
- Clean covers must be generated or permissively licensed (CC0, MIT, Apache-2.0).
- Say which base SLM you started from and keep its license.
- Do not claim llama.cpp-native vision unless `llama-cli --image` works without a private fork.
- Do not treat ONNX or a local `:8080` Python API as the 2026 product path.

---

## 9. How to enter

1. Fork the repository.
2. Add `datasets/<handle>_submission_2026/` following §4.
3. Point `model/serve.json` at a llama.cpp-loadable GGUF you can rebuild.
4. Integrate `models/detector_balanced.pth` or `models/starlight.gguf` under §5.
5. Pass `scripts/validate_submission_2026.py`.
6. Open a pull request. Title prefix: `[2026]` plus the handle.

Maintainers run the public gate, then the hidden look/chat set. Results go on `models/leaderboard.md` under a 2026 section.

---

## 10. Reference implementation

`datasets/grok_submission_2026/` is Track B:

- Language: a small instruct GGUF (Qwen2.5-0.5B-Instruct class) served only by llama.cpp.
- Vision / secrets: official `BalancedStarlightDetector` + spec extractors via `look.py`.
- Product command: `serve.sh` (wraps `llama-cli`).

That entry exists to make the rules executable, not to freeze the architecture. Tracks A and C are welcome and will outscore B on “native look” if they keep recovery and honesty.

---

## 11. Document control

| Field | Value |
|-------|--------|
| Version | 1.0 |
| Date | 2026-08-13 |
| Author | Grok (xAI), Project Starlight |
| Supersedes | none (new track) |
| Related | [DATASET_GUIDELINES.md](DATASET_GUIDELINES.md), [docs/STEGO_FORMAT_SPEC.md](docs/STEGO_FORMAT_SPEC.md), [docs/GGUF_EXPORT.md](docs/GGUF_EXPORT.md) |
