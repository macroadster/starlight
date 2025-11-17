#!/bin/bash

# publish_to_hf.sh - Automated script to scan datasets, update model card, and publish to Hugging Face Hub

set -e

# Configuration
MODEL_PATH="models/detector_balanced.onnx"
CONFIG_PATH="model/config.json"
INFERENCE_PATH="scripts/inference.py"
MODEL_CARD_PATH="models/model_card.md"
REPO_NAME="macroadster/starlight"
HF_TOKEN="${HF_TOKEN:-}"  # Set via environment variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
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
        exit 1
    fi

    log_info "Dependencies OK"
}

# Run dataset scan
run_scan() {
    log_info "Running dataset scan..."

    if [[ ! -f "scan_datasets.sh" ]]; then
        log_error "scan_datasets.sh not found"
        exit 1
    fi

    # Run scan and capture output
    SCAN_OUTPUT=$(bash scan_datasets.sh -m "$MODEL_PATH" 2>&1)
    SCAN_EXIT_CODE=$?

    if [[ $SCAN_EXIT_CODE -ne 0 ]]; then
        log_error "Dataset scan failed"
        echo "$SCAN_OUTPUT"
        exit 1
    fi

    log_info "Dataset scan completed"
    echo "$SCAN_OUTPUT"
}

# Parse scan results
parse_results() {
    log_info "Parsing scan results..."

    # Extract overall metrics using Python
    OVERALL_STATS=$(python3 -c "
import re
import sys

output = sys.stdin.read()

# Find overall performance section
overall_match = re.search(r'OVERALL PERFORMANCE:(.*?)PERFORMANCE ASSESSMENT:', output, re.DOTALL)
if not overall_match:
    print('ERROR: Could not find overall performance section')
    sys.exit(1)

stats_text = overall_match.group(1)

# Extract numbers
fp_match = re.search(r'False Positives \(Clean\)\s*\│\s*(\d+)\s*│\s*([\d.]+)', stats_text)
tp_match = re.search(r'True Positives \(Stego\)\s*\│\s*(\d+)\s*│\s*([\d.]+)', stats_text)

if not fp_match or not tp_match:
    print('ERROR: Could not parse metrics')
    sys.exit(1)

fp_count = fp_match.group(1)
fp_rate = fp_match.group(2)
tp_count = tp_match.group(1)
detection_rate = tp_match.group(2)

print(f'{fp_rate}:{detection_rate}')
" <<< "$SCAN_OUTPUT")

    if [[ "$OVERALL_STATS" == ERROR* ]]; then
        log_error "$OVERALL_STATS"
        exit 1
    fi

    IFS=':' read -r FP_RATE DETECTION_RATE <<< "$OVERALL_STATS"

    log_info "Parsed metrics - FP Rate: ${FP_RATE}%, Detection Rate: ${DETECTION_RATE}%"
}

# Update model card
update_model_card() {
    log_info "Updating model card..."

    if [[ ! -f "$MODEL_CARD_PATH" ]]; then
        log_error "Model card not found: $MODEL_CARD_PATH"
        exit 1
    fi

    # Update performance section
    python3 -c "
import re

with open('$MODEL_CARD_PATH', 'r') as f:
    content = f.read()

# Update metrics
content = re.sub(
    r'\\| Accuracy \\| [\\d.]+% \\|',
    f'| Accuracy | ${DETECTION_RATE}% |',
    content
)

content = re.sub(
    r'\\| AUC-ROC \\| [\\d.]+ \\|',
    f'| AUC-ROC | TBD |',
    content
)

content = re.sub(
    r'\\| F1 Score \\| [\\d.]+ \\|',
    f'| F1 Score | TBD |',
    content
)

content = re.sub(
    r'\\| Extraction BER \\| [\\d.]+ \\|',
    f'| Extraction BER | N/A |',
    content
)

with open('$MODEL_CARD_PATH', 'w') as f:
    f.write(content)

print('Model card updated')
" || {
        log_error "Failed to update model card"
        exit 1
    }

    log_info "Model card updated with latest metrics"
}

# Publish to Hugging Face Hub
publish_to_hf() {
    log_info "Publishing to Hugging Face Hub..."

    # Check if files exist
    for file in "$MODEL_PATH" "$CONFIG_PATH" "$INFERENCE_PATH" "$MODEL_CARD_PATH"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done

    # Create/update repo
    python3 -c "
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token='$HF_TOKEN')

try:
    # Try to create repo (will succeed if it doesn't exist)
    create_repo('$REPO_NAME', private=False, token='$HF_TOKEN')
    print('Created new repo')
except:
    print('Repo already exists')

# Upload files
files_to_upload = [
    ('$MODEL_PATH', 'model.onnx'),
    ('$CONFIG_PATH', 'config.json'),
    ('$INFERENCE_PATH', 'inference.py'),
    ('$MODEL_CARD_PATH', 'README.md')
]

for local_path, repo_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id='$REPO_NAME',
        token='$HF_TOKEN'
    )
    print(f'Uploaded {local_path} to {repo_path}')

print('All files uploaded successfully')
" || {
        log_error "Failed to publish to Hugging Face Hub"
        exit 1
    }

    log_info "Successfully published to $REPO_NAME"
}

# Main execution
main() {
    echo "============================================================"
    echo "🚀 Starlight HF Hub Publisher"
    echo "============================================================"

    check_dependencies
    run_scan
    parse_results
    update_model_card
    publish_to_hf

    echo ""
    echo "============================================================"
    log_info "✅ Publication completed successfully!"
    log_info "Repository: https://huggingface.co/$REPO_NAME"
    echo "============================================================"
}

# Run main function
main "$@"
