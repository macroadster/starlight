#!/usr/bin/env bash
# Track B serve wrapper for Codex 2026 submission
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

GGUF="${STARLIGHT_LLM_GGUF:-$HERE/model/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
if [[ ! -f "$GGUF" ]]; then
  echo "GGUF missing at $GGUF — running fuse_vlm.py to fetch" >&2
  python3 "$HERE/fuse_vlm.py"
fi

IMAGE=""
PROMPT="Hello! Who are you?"
N_PREDICT=128

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    -p|--prompt) PROMPT="$2"; shift 2 ;;
    -n|--n-predict) N_PREDICT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

LLAMA_CLI="$(command -v llama-cli || true)"
if [[ -z "$LLAMA_CLI" ]]; then
  echo "ERROR: llama-cli not found on PATH." >&2
  exit 1
fi

if [[ -n "$IMAGE" ]]; then
  SYS_BLOCK="$(python3 "$HERE/look.py" "$IMAGE" --block)"
  exec "$LLAMA_CLI" -m "$GGUF" -sys "$SYS_BLOCK" -p "$PROMPT" -n "$N_PREDICT" --single-turn --no-conversation --no-display-prompt
else
  SYS_TXT="$(cat "$HERE/model/system_prompt.txt")"
  exec "$LLAMA_CLI" -m "$GGUF" -sys "$SYS_TXT" -p "$PROMPT" -n "$N_PREDICT" --single-turn --no-conversation --no-display-prompt
fi
