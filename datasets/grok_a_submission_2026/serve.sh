#!/usr/bin/env bash
# Serve grok_a_submission_2026 with stock llama.cpp (Track A).
# Usage:
#   ./serve.sh -p "Hello"
#   ./serve.sh --image eval/fixtures/seed_protocol_alpha_000.png -p "What secret is in this picture?"
# Extra args are forwarded to llama-cli.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SERVE_JSON="$HERE/model/serve.json"
SYS_PROMPT="$HERE/model/system_prompt.txt"
GGUF="${STARLIGHT_LLM_GGUF:-}"
MMPROJ="${STARLIGHT_MMPROJ_GGUF:-}"
IMAGE=""
PROMPT=""
N_PREDICT="${N_PREDICT:-128}"
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    -p|--prompt)
      PROMPT="$2"
      shift 2
      ;;
    -n|--n-predict)
      N_PREDICT="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$GGUF" ]]; then
  GGUF="$HERE/$(python3 -c "import json; print(json.load(open('$SERVE_JSON'))['llm_gguf'])")"
fi
if [[ -z "$MMPROJ" ]]; then
  MMPROJ="$HERE/$(python3 -c "import json; print(json.load(open('$SERVE_JSON'))['mmproj_gguf'])")"
fi

if [[ ! -f "$GGUF" || ! -f "$MMPROJ" ]]; then
  echo "GGUF missing — running fuse_vlm.py" >&2
  (cd "$HERE" && python3 fuse_vlm.py)
fi

if [[ ! -f "$GGUF" || ! -f "$MMPROJ" ]]; then
  echo "still missing $GGUF or $MMPROJ" >&2
  exit 1
fi

if ! command -v llama-cli >/dev/null 2>&1; then
  echo "llama-cli not on PATH" >&2
  exit 1
fi

if [[ -z "$PROMPT" ]]; then
  if [[ -n "$IMAGE" ]]; then
    PROMPT="What secret is in this picture?"
  else
    PROMPT="Hello, who are you?"
  fi
fi

ROOT="$(cd "$HERE/../.." && pwd)"
if [[ -n "$IMAGE" && "$IMAGE" != /* ]]; then
  if [[ -f "$HERE/$IMAGE" ]]; then
    IMAGE="$HERE/$IMAGE"
  elif [[ -f "$ROOT/$IMAGE" ]]; then
    IMAGE="$ROOT/$IMAGE"
  fi
fi

# The marker is where mtmd inserts the picture. The text never contains a secret.
if [[ -n "$IMAGE" && "$PROMPT" != *"<__media__>"* ]]; then
  PROMPT="<__media__>
${PROMPT}"
fi

ARGS=(
  llama-cli
  -m "$GGUF"
  --mmproj "$MMPROJ"
  --system-prompt-file "$SYS_PROMPT"
  -p "$PROMPT"
  -n "$N_PREDICT"
  --no-display-prompt
  --single-turn
  --no-conversation
)
if [[ -n "$IMAGE" ]]; then
  ARGS+=(--image "$IMAGE")
fi

exec "${ARGS[@]}" ${FORWARD[@]+"${FORWARD[@]}"}
