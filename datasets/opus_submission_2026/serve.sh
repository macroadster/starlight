#!/usr/bin/env bash
# Serve opus_submission_2026 with stock llama.cpp (Track B).
#   ./serve.sh -p "Hello, who are you?"
#   ./serve.sh --image eval/fixtures/seed_opus_alpha_000.png -p "What secret is in this picture?"
# With --image, look.py runs the official detector and writes a quote-lock GBNF
# so llama-cli copies the payload verbatim. --raw keeps llama-cli's own banner/stats.
# Unknown args are forwarded to llama-cli.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SERVE_JSON="$HERE/model/serve.json"
SYS_PROMPT="$HERE/model/system_prompt.txt"
GGUF="${STARLIGHT_LLM_GGUF:-}"
IMAGE=""
PROMPT=""
N_PREDICT=""
RAW=0
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    -p|--prompt) PROMPT="$2"; shift 2 ;;
    -n|--n-predict) N_PREDICT="$2"; shift 2 ;;
    --raw) RAW=1; shift ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) FORWARD+=("$1"); shift ;;
  esac
done

if [[ -z "$GGUF" ]]; then
  GGUF="$HERE/$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['llm_gguf'])" "$SERVE_JSON")"
fi
if [[ ! -f "$GGUF" ]]; then
  echo "LLM GGUF missing at $GGUF; running fuse_vlm.py" >&2
  python3 "$HERE/fuse_vlm.py" >&2
fi
[[ -f "$GGUF" ]] || { echo "still no GGUF at $GGUF" >&2; exit 1; }
command -v llama-cli >/dev/null 2>&1 || { echo "llama-cli not on PATH" >&2; exit 1; }

TMP="$(mktemp -d -t starlight-opus.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
SYS_FILE="$TMP/system.txt"
GRAMMAR_ARGS=()

if [[ -n "$IMAGE" ]]; then
  if [[ "$IMAGE" != /* ]]; then
    for base in "$PWD" "$HERE" "$ROOT"; do
      if [[ -f "$base/$IMAGE" ]]; then IMAGE="$base/$IMAGE"; break; fi
    done
  fi
  [[ -f "$IMAGE" ]] || { echo "image not found: $IMAGE" >&2; exit 1; }
  (cd "$ROOT" && python3 "$HERE/look.py" "$IMAGE" --block --grammar-out "$TMP/quote.gbnf") > "$SYS_FILE"
  if [[ "${STARLIGHT_NO_LOCK:-0}" != 1 ]]; then
    GRAMMAR_ARGS=(--grammar-file "$TMP/quote.gbnf")
  fi
  PROMPT="${PROMPT:-What secret is in this picture?}"
  N_PREDICT="${N_PREDICT:-4096}"
else
  cp "$SYS_PROMPT" "$SYS_FILE"
  PROMPT="${PROMPT:-Hello, who are you?}"
  N_PREDICT="${N_PREDICT:-192}"
fi

CMD=(llama-cli -m "$GGUF" --system-prompt-file "$SYS_FILE" -p "$PROMPT" -n "$N_PREDICT"
  --temp "${STARLIGHT_TEMP:-0.3}" --seed "${STARLIGHT_SEED:-2142}"
  --no-display-prompt --single-turn --no-conversation
  ${GRAMMAR_ARGS[@]+"${GRAMMAR_ARGS[@]}"} ${FORWARD[@]+"${FORWARD[@]}"})

if [[ "$RAW" == 1 ]]; then
  exec "${CMD[@]}"
fi

"${CMD[@]}" 2>"$TMP/stderr.log" > "$TMP/stdout.log" || { cat "$TMP/stdout.log" "$TMP/stderr.log" >&2; exit 1; }
python3 - "$TMP/stdout.log" "$PROMPT" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
marker = "> " + sys.argv[2]
start = text.find(marker)
reply = text[start + len(marker):] if start >= 0 else text
reply = re.split(r"\n\[ Prompt: [^\]]*\]", reply)[0]
reply = re.sub(r"\n+Exiting\.\.\.\s*$", "", reply)
print(reply.strip())
PY
