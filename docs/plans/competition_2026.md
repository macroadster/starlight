# Agent plan: Starlight Chat 2026

**Status:** Rules published; Grok Track B reference entry in `datasets/grok_submission_2026/`.  
**Rules:** [COMPETITION_2026.md](../../COMPETITION_2026.md)

## Goal

Ship a competition track whose winning artifact is a **llama.cpp-servable GGUF** that can talk and, given a picture, quote embedded secrets. The official detector (`models/detector_balanced.pth` / `models/starlight.gguf`) is the look backbone.

## Done in this pass

- Competition rules (tracks A/B/C, Look Protocol, gates, scoring)
- Repo-level validator `scripts/validate_submission_2026.py`
- Grok reference submission (Track B)
- Pointers from README, USAGE, DATASET_GUIDELINES, Makefile

## Next (other agents)

- Track A: llama.cpp `mtmd` projector from detector streams
- Track C: distill detector into a native vision tower + parity table
- Optional LoRA on the 0.5B SLM using `conversation_sft.jsonl`
- Hidden eval set (not in git)
- Fill `models/leaderboard.md` 2026 section after independent scoring
