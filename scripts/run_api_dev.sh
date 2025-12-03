#!/usr/bin/env bash
set -euo pipefail

# Simple dev runner for the FastAPI service without containers.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export BLOCKS_DIR="${BLOCKS_DIR:-$ROOT_DIR/blocks}"
export STARGATE_API_KEY="${STARGATE_API_KEY:-demo-api-key}"
export ALLOW_ANONYMOUS_SCAN="${ALLOW_ANONYMOUS_SCAN:-true}"
export PORT="${PORT:-8080}"

echo "Starting Starlight API on port ${PORT} (blocks at ${BLOCKS_DIR})"
python -m uvicorn bitcoin_api:app --reload --host 0.0.0.0 --port "${PORT}"
