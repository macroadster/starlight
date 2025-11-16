#!/bin/bash

# scan_datasets.sh - Comprehensive dataset scanning script
# Loops through all submission datasets and validation set to provide accuracy summaries

set -e

# Default model path
DEFAULT_MODEL="models/detector_balanced.onnx"

# Parse command line arguments
MODEL_PATH=""
SHOW_DETAILS=false
WORKERS=$(nproc 2>/dev/null || echo 4)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model PATH     Path to ONNX model file (default: $DEFAULT_MODEL)"
    echo "  -d, --details        Show detailed file-by-file results"
    echo "  -w, --workers N     Number of parallel workers (default: $WORKERS)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default conservative model"
    echo "  $0 -m models/detector_balanced.onnx   # Use specific model"
    echo "  $0 -d                                 # Show detailed results"
    echo "  $0 -m models/detector.onnx -w 8       # Custom model with 8 workers"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--details)
            SHOW_DETAILS=true
            shift
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Use default model if none specified
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$DEFAULT_MODEL"
fi

# Check if model exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check if scanner exists
if [[ ! -f "scanner.py" ]]; then
    echo "Error: scanner.py not found in current directory"
    exit 1
fi

echo "============================================================"
echo "🔍 Project Starlight - Dataset Scanning Tool"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Workers: $WORKERS"
echo "Details: $SHOW_DETAILS"
echo ""

# Function to scan a directory and parse results
scan_directory() {
    local dir_path="$1"
    local dir_type="$2"
    local dataset_name="$3"
    
    if [[ ! -d "$dir_path" ]]; then
        echo "⚠️  Directory not found: $dir_path"
        return
    fi
    
    echo "📁 Scanning $dataset_name ($dir_type)..."
    
    # Run scanner and capture output
    local scan_output
    scan_output=$(python3 scanner.py "$dir_path" --model "$MODEL_PATH" --workers "$WORKERS" --json 2>/dev/null)
    
    # Parse JSON results
    local total_files=0
    local detected_files=0
    local clean_files=0
    local errors=0
    
    # Count files using jq if available, otherwise use Python
    if command -v jq >/dev/null 2>&1; then
        total_files=$(echo "$scan_output" | jq 'length' 2>/dev/null || echo 0)
        detected_files=$(echo "$scan_output" | jq '[.[] | select(.is_stego == true)] | length' 2>/dev/null || echo 0)
        clean_files=$(echo "$scan_output" | jq '[.[] | select(.is_stego == false)] | length' 2>/dev/null || echo 0)
        errors=$(echo "$scan_output" | jq '[.[] | select(has("error"))] | length' 2>/dev/null || echo 0)
    else
        # Fallback to Python for JSON parsing
        total_files=$(python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(len(data))
except:
    print(0)
" <<< "$scan_output")
        
        detected_files=$(python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(sum(1 for item in data if item.get('is_stego')))
except:
    print(0)
" <<< "$scan_output")
        
        clean_files=$(python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(sum(1 for item in data if not item.get('is_stego') and not item.get('error')))
except:
    print(0)
" <<< "$scan_output")
        
        errors=$(python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(sum(1 for item in data if item.get('error')))
except:
    print(0)
" <<< "$scan_output")
    fi
    
    # Calculate percentages
    local detection_rate=0
    local false_positive_rate=0
    
    if [[ $total_files -gt 0 ]]; then
        if [[ "$dir_type" == "stego" ]]; then
            # For stego directories, detection rate is detected/total
            detection_rate=$(echo "scale=1; $detected_files * 100 / $total_files" | bc -l 2>/dev/null || echo 0)
        else
            # For clean directories, false positive rate is detected/total
            false_positive_rate=$(echo "scale=1; $detected_files * 100 / $total_files" | bc -l 2>/dev/null || echo 0)
        fi
    fi
    
    # Store results in global arrays
    if [[ "$dir_type" == "stego" ]]; then
        STEGO_RESULTS+=("$dataset_name:$total_files:$detected_files:$detection_rate:$errors")
    else
        CLEAN_RESULTS+=("$dataset_name:$total_files:$detected_files:$false_positive_rate:$errors")
    fi
    
    # Show details if requested
    if [[ "$SHOW_DETAILS" == "true" ]]; then
        echo "   📊 Total files: $total_files"
        echo "   ✅ Detected: $detected_files"
        if [[ "$dir_type" == "stego" ]]; then
            echo "   📈 Detection rate: ${detection_rate}%"
        else
            echo "   ❌ False positives: $detected_files (${false_positive_rate}%)"
        fi
        if [[ $errors -gt 0 ]]; then
            echo "   ⚠️  Errors: $errors"
        fi
        
        # Show detected files if any
        if [[ $detected_files -gt 0 ]]; then
            echo "   🔍 Detected files:"
            echo "$scan_output" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for item in data:
        if item.get('is_stego'):
            filename = item.get('file_path', 'unknown')
            stego_type = item.get('stego_type', 'unknown')
            confidence = item.get('confidence', 0)
            print(f'     - {filename.split(\"/\")[-1]} ({stego_type}, {confidence:.1%})')
except:
    pass
"
        fi
        echo ""
    fi
}

# Arrays to store results
CLEAN_RESULTS=()
STEGO_RESULTS=()

# Find all submission directories
echo "🔍 Discovering datasets..."
for submission_dir in datasets/*_submission_*; do
    if [[ -d "$submission_dir" ]]; then
        dataset_name=$(basename "$submission_dir")
        
        # Scan clean directory
        scan_directory "$submission_dir/clean" "clean" "$dataset_name"
        
        # Scan stego directory  
        scan_directory "$submission_dir/stego" "stego" "$dataset_name"
    fi
done

# Also scan validation set
if [[ -d "datasets/val" ]]; then
    scan_directory "datasets/val/clean" "clean" "validation"
    scan_directory "datasets/val/stego" "stego" "validation"
fi

echo ""
echo "============================================================"
echo "📊 SUMMARY RESULTS"
echo "============================================================"

# Function to print summary table
print_summary() {
    local title="$1"
    local results_array_name="$2"
    local metric_name="$3"
    
    # Get the array by name using eval
    eval "local results=(\"\${${results_array_name}[@]}\")"
    
    echo ""
    echo "$title:"
    echo "┌─────────────────────────────────┬────────┬─────────┬──────────┬───────┐"
    echo "│ Dataset                        │ Files  │ $metric_name │ Rate (%) │ Errors│"
    echo "├─────────────────────────────────┼────────┼─────────┼──────────┼───────┤"
    
    for result in "${results[@]}"; do
        IFS=':' read -r dataset total detected rate errors <<< "$result"
        printf "│ %-31s │ %6d │ %7d │ %8.1f │ %5d │\n" "$dataset" "$total" "$detected" "$rate" "$errors"
    done
    
    echo "└─────────────────────────────────┴────────┴─────────┴──────────┴───────┘"
}

# Print clean results summary
print_summary "🧹 CLEAN DIRECTORIES (False Positives)" "CLEAN_RESULTS" "False Pos"

# Print stego results summary  
print_summary "🎯 STEGO DIRECTORIES (Detection)" "STEGO_RESULTS" "Detected"

# Calculate overall statistics
echo ""
echo "📈 OVERALL PERFORMANCE:"

# Calculate totals
total_clean_files=0
total_clean_fps=0
total_stego_files=0
total_stego_detected=0

for result in "${CLEAN_RESULTS[@]}"; do
    IFS=':' read -r dataset total detected rate errors <<< "$result"
    total_clean_files=$((total_clean_files + total))
    total_clean_fps=$((total_clean_fps + detected))
done

for result in "${STEGO_RESULTS[@]}"; do
    IFS=':' read -r dataset total detected rate errors <<< "$result"
    total_stego_files=$((total_stego_files + total))
    total_stego_detected=$((total_stego_detected + detected))
done

# Calculate overall rates
overall_fp_rate=0
overall_detection_rate=0

if [[ $total_clean_files -gt 0 ]]; then
    overall_fp_rate=$(echo "scale=2; $total_clean_fps * 100 / $total_clean_files" | bc -l 2>/dev/null || echo 0)
fi

if [[ $total_stego_files -gt 0 ]]; then
    overall_detection_rate=$(echo "scale=2; $total_stego_detected * 100 / $total_stego_files" | bc -l 2>/dev/null || echo 0)
fi

echo "┌─────────────────────────────────┬──────────┬─────────────┐"
echo "│ Metric                         │ Count    │ Rate (%)    │"
echo "├─────────────────────────────────┼──────────┼─────────────┤"
printf "│ Total Clean Files              │ %8d │ %11.2f │\n" "$total_clean_files" "$overall_fp_rate"
printf "│ Total Stego Files              │ %8d │ %11.2f │\n" "$total_stego_files" "$overall_detection_rate"
echo "├─────────────────────────────────┼──────────┼─────────────┤"
printf "│ False Positives (Clean)        │ %8d │ %11.2f │\n" "$total_clean_fps" "$overall_fp_rate"
printf "│ True Positives (Stego)         │ %8d │ %11.2f │\n" "$total_stego_detected" "$overall_detection_rate"
echo "└─────────────────────────────────┴──────────┴─────────────┘"

echo ""
echo "🎯 PERFORMANCE ASSESSMENT:"

# Performance assessment
if (( $(echo "$overall_fp_rate < 1.0" | bc -l 2>/dev/null || echo 0) )); then
    echo "✅ False positive rate: EXCELLENT (< 1%)"
elif (( $(echo "$overall_fp_rate < 5.0" | bc -l 2>/dev/null || echo 0) )); then
    echo "✅ False positive rate: GOOD (< 5%)"
else
    echo "❌ False positive rate: NEEDS IMPROVEMENT (> 5%)"
fi

if (( $(echo "$overall_detection_rate > 95.0" | bc -l 2>/dev/null || echo 0) )); then
    echo "✅ Detection rate: EXCELLENT (> 95%)"
elif (( $(echo "$overall_detection_rate > 85.0" | bc -l 2>/dev/null || echo 0) )); then
    echo "✅ Detection rate: GOOD (> 85%)"
else
    echo "❌ Detection rate: NEEDS IMPROVEMENT (< 85%)"
fi

echo ""
echo "============================================================"
echo "✅ Scan completed successfully!"
echo "============================================================"
