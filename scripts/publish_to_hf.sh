#!/bin/bash
# publish_to_hf.sh — Upload GGUF primary artifacts to Hugging Face Hub
#
# Primary path for Stargate/Trin: starlight.gguf + starlight_gguf_map.json
# Optional secondary: .onnx / .pth if present (never required)
#
# Usage:
#   HF_TOKEN=hf_... ./scripts/publish_to_hf.sh
#   HF_REPO=macroadster/starlight-prod HF_TOKEN=... ./scripts/publish_to_hf.sh
#
# Never commit tokens.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (env overrides)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

GGUF_PATH="${GGUF_PATH:-models/starlight.gguf}"
MAP_PATH="${MAP_PATH:-models/starlight_gguf_map.json}"
MODEL_CARD_PATH="${MODEL_CARD_PATH:-models/model_card.md}"
# HF_REPO preferred; REPO_NAME kept for backward compatibility
HF_REPO="${HF_REPO:-${REPO_NAME:-macroadster/starlight-prod}}"
REPO_NAME="$HF_REPO"
HF_TOKEN="${HF_TOKEN:-}"

# Optional secondary artifacts (uploaded only if present)
ONNX_PATH="${ONNX_PATH:-models/detector_balanced.onnx}"
PTH_PATH="${PTH_PATH:-models/detector_balanced.pth}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------------------------------------------------------------
check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        exit 1
    fi

    if ! python3 -c "import huggingface_hub" &> /dev/null; then
        log_error "huggingface_hub not installed. Install with: pip install huggingface_hub"
        exit 1
    fi

    if [[ -z "$HF_TOKEN" ]]; then
        log_error "HF_TOKEN environment variable not set"
        log_error "  export HF_TOKEN=hf_...   # never commit this"
        exit 1
    fi

    log_info "Dependencies OK (huggingface_hub present, HF_TOKEN set)"
}

# ---------------------------------------------------------------------------
check_primary_artifacts() {
    log_info "Checking primary GGUF artifacts..."
    local missing=0

    if [[ ! -f "$GGUF_PATH" ]]; then
        log_error "Required GGUF not found: $GGUF_PATH"
        log_error "  Export first: python3 scripts/export_starlight_gguf.py \\"
        log_error "    --input models/detector_balanced.pth \\"
        log_error "    --output models/starlight.gguf \\"
        log_error "    --name-map models/starlight_gguf_map.json"
        missing=1
    else
        log_info "  Found $GGUF_PATH ($(du -h "$GGUF_PATH" | awk '{print $1}'))"
    fi

    if [[ ! -f "$MAP_PATH" ]]; then
        log_error "Required name map not found: $MAP_PATH"
        missing=1
    else
        log_info "  Found $MAP_PATH"
    fi

    if [[ ! -f "$MODEL_CARD_PATH" ]]; then
        log_error "Required model card not found: $MODEL_CARD_PATH"
        missing=1
    else
        log_info "  Found $MODEL_CARD_PATH (→ README.md on HF)"
    fi

    if [[ "$missing" -ne 0 ]]; then
        exit 1
    fi
}

# ---------------------------------------------------------------------------
publish_to_hf() {
    log_info "Publishing to Hugging Face Hub: $HF_REPO"
    log_info "  Primary: $GGUF_PATH → starlight.gguf"
    log_info "  Sidecar: $MAP_PATH → starlight_gguf_map.json"
    log_info "  Card:    $MODEL_CARD_PATH → README.md"

    if [[ -f "$ONNX_PATH" ]]; then
        log_info "  Optional ONNX present: $ONNX_PATH (will upload as secondary)"
    else
        log_warn "  Optional ONNX not present — skipped (GGUF is primary)"
    fi
    if [[ -f "$PTH_PATH" ]]; then
        log_info "  Optional PTH present: $PTH_PATH (will upload as secondary)"
    else
        log_warn "  Optional PTH not present — skipped"
    fi

    # Pass paths via env to avoid shell-injection into Python; token only via env
    GGUF_PATH="$GGUF_PATH" \
    MAP_PATH="$MAP_PATH" \
    MODEL_CARD_PATH="$MODEL_CARD_PATH" \
    ONNX_PATH="$ONNX_PATH" \
    PTH_PATH="$PTH_PATH" \
    HF_REPO="$HF_REPO" \
    HF_TOKEN="$HF_TOKEN" \
    python3 << 'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
repo_id = os.environ["HF_REPO"]
gguf = Path(os.environ["GGUF_PATH"])
name_map = Path(os.environ["MAP_PATH"])
card = Path(os.environ["MODEL_CARD_PATH"])
onnx = Path(os.environ.get("ONNX_PATH", "models/detector_balanced.onnx"))
pth = Path(os.environ.get("PTH_PATH", "models/detector_balanced.pth"))

api = HfApi(token=token)

try:
    create_repo(repo_id, private=False, token=token, exist_ok=True)
    print(f"Repo ready: {repo_id}")
except Exception as e:
    print(f"create_repo note: {e}", file=sys.stderr)

# Required primary uploads
uploads = [
    (gguf, "starlight.gguf"),
    (name_map, "starlight_gguf_map.json"),
    (card, "README.md"),
]

# Optional secondary (do not require)
if onnx.is_file():
    uploads.append((onnx, "detector_balanced.onnx"))
if pth.is_file():
    uploads.append((pth, "detector_balanced.pth"))

for local_path, repo_path in uploads:
    if not local_path.is_file():
        print(f"ERROR: missing {local_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Uploading {local_path} → {repo_path} ...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=repo_id,
        token=token,
    )
    print(f"  OK: {repo_path}")

print("All uploads completed successfully")
print(f"Repository: https://huggingface.co/{repo_id}")
print(f"GGUF URL:   https://huggingface.co/{repo_id}/resolve/main/starlight.gguf")
print(f"Map URL:    https://huggingface.co/{repo_id}/resolve/main/starlight_gguf_map.json")
PY
}

# ---------------------------------------------------------------------------
main() {
    echo "============================================================"
    echo "Starlight HF Publisher (GGUF primary)"
    echo "============================================================"
    echo "Repo:  $HF_REPO"
    echo "GGUF:  $GGUF_PATH"
    echo "Map:   $MAP_PATH"
    echo "Card:  $MODEL_CARD_PATH"
    echo "============================================================"

    check_dependencies
    check_primary_artifacts
    # Scan pipeline intentionally skipped — publish artifacts only.
    # Optional offline eval: python3 scanner.py ... (not required for HF).
    publish_to_hf

    echo ""
    echo "============================================================"
    log_info "Publication completed successfully"
    log_info "Repository: https://huggingface.co/$HF_REPO"
    log_info "Stargate download:"
    log_info "  https://huggingface.co/$HF_REPO/resolve/main/starlight.gguf"
    log_info "  https://huggingface.co/$HF_REPO/resolve/main/starlight_gguf_map.json"
    echo "============================================================"
}

main "$@"
