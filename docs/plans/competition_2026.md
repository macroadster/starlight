# Agent plan: Starlight Chat 2026

**Status:** Rules v1.1 (2026-10-14). Grok Track B reference in `datasets/grok_submission_2026/` emits a hint only. Track A recipe is `datasets/grok_a_submission_2026/` (CLIP `mlp` mmproj fit to `embedding_head`).  
**Rules:** [COMPETITION_2026.md](../../COMPETITION_2026.md)

## Goal

Ship a competition track whose winning artifact is a **llama.cpp-servable GGUF** that can talk and, given a picture, quote embedded secrets. The official detector (`models/detector_balanced.pth` / `models/starlight.gguf`) is the look backbone.

## Done in this pass

- Competition rules (tracks A/B/C, Look Protocol, gates, scoring)
- Repo-level validator `scripts/validate_submission_2026.py`
- Grok reference submission (Track B)
- Track A recipe `datasets/grok_a_submission_2026/` (mmproj loads; blind quote not claimed)
- Pointers from README, USAGE, DATASET_GUIDELINES, Makefile

## Next (other agents)

- Track C: distill detector into a native vision tower + parity table
- Optional LoRA on the 0.5B SLM using `conversation_sft.jsonl`
- Hidden eval set (not in git)
- Fill `models/leaderboard.md` 2026 section after independent scoring
