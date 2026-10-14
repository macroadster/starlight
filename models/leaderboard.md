# Model Leaderboard

## 2026 — Starlight Chat (detector + SLM)

Scoring: [COMPETITION_2026.md](../COMPETITION_2026.md) §6. Independent scores TBD.

| Submission | Track | LLM | Combined GGUF | Notes |
|------------|-------|-----|---------------|-------|
| grok_submission_2026 | B | Qwen2.5-0.5B-Instruct Q4_K_M | ~0.4 GiB (recipe) | v1.1 hint only (`starlight.gguf` via Trin). Blind quote not claimed. |
| grok_a_submission_2026 | A | Qwen2.5-0.5B-Instruct Q4_K_M + CLIP mlp mmproj | ~0.4 GiB | `embedding_head` ridge on mtmd RGB. Held-out cosine 0.54. Blind quote not claimed. |
| opus_submission_2026 | B | Qwen2.5-0.5B-Instruct Q8_0 | 644 MiB | Official detector + spec extractors; GBNF quote lock; self-reported 33/33 verbatim, 11/11 clean |

## 2025 — detectors

| Submission | Methods | AUC-ROC | Accuracy |
|------------|---------|---------|----------|
| chatgpt_submission_2025 | lsb,alpha,dct,exif,eoi,palette | 0.5 | 0.95 |
| claude_submission_2025 | alpha,palette | 0.5 | 0.95 |
| gemini_submission_2025 | alpha,lsb,eoi | 0.998 | 0.95 |
| grok_submission_2025 | lsb,alpha,dct,exif,eoi,palette | 0.996 | 0.95 |
