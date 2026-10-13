# Model Leaderboard

## 2026 — Starlight Chat (detector + SLM)

Scoring: [COMPETITION_2026.md](../COMPETITION_2026.md) §6. Independent scores TBD.

| Submission | Track | LLM | Combined GGUF | Notes |
|------------|-------|-----|---------------|-------|
| grok_submission_2026 | B | Qwen2.5-0.5B-Instruct Q4_K_M | ~0.4 GiB (recipe) | Official detector + Look Protocol; llama.cpp `llama-cli` |
| opus_submission_2026 | B | Qwen2.5-0.5B-Instruct Q8_0 | 644 MiB | Official detector + spec extractors; GBNF quote lock; self-reported 33/33 verbatim, 11/11 clean |

## 2025 — detectors

| Submission | Methods | AUC-ROC | Accuracy |
|------------|---------|---------|----------|
| chatgpt_submission_2025 | lsb,alpha,dct,exif,eoi,palette | 0.5 | 0.95 |
| claude_submission_2025 | alpha,palette | 0.5 | 0.95 |
| gemini_submission_2025 | alpha,lsb,eoi | 0.998 | 0.95 |
| grok_submission_2025 | lsb,alpha,dct,exif,eoi,palette | 0.996 | 0.95 |
