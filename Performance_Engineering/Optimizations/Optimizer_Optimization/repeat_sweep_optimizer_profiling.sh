#!/bin/bash
set -euo pipefail

# run
# ./run_sweep_optimized.sh --repeat 3 --template base_config.yaml --dataset ../../../DataPreprocessing/train.jsonl --csv-prefix optimizer_sweep --steps 600 --extra "--drop-cache --clean"

# Default values
REPEAT=1
STEPS=500
WAIT=20
CSV_BASE="results"
OUTPUT_DIR="./tmp_cfgs"
TEMPLATE="base_config.yaml"
DATASET="../../../DataPreprocessing/train.jsonl"
EXTRA_ARGS=()

# Help message
usage() {
    echo "Usage: $0 [--repeat N] [--steps N] [--wait N] [--template PATH] [--dataset PATH] [--csv-prefix NAME] [--output-dir DIR] [--extra '...']"
    exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat)       REPEAT="$2"; shift 2 ;;
        --steps)        STEPS="$2"; shift 2 ;;
        --wait)         WAIT="$2"; shift 2 ;;
        --template)     TEMPLATE="$2"; shift 2 ;;
        --dataset)      DATASET="$2"; shift 2 ;;
        --csv-prefix)   CSV_BASE="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --extra)        IFS=' ' read -r -a EXTRA_ARGS <<< "$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown argument: $1"; usage ;;
    esac
done

# Make sure key files exist
[[ -f "$TEMPLATE" ]] || { echo "Template not found: $TEMPLATE"; exit 1; }
[[ -f "$DATASET" ]] || { echo "Dataset not found: $DATASET"; exit 1; }

# Run loop
for ((i=1; i<=REPEAT; i++)); do
    RUN_CSV="${CSV_BASE}_run${i}.csv"
    echo -e "\n▶▶▶ Sweep Run $i / $REPEAT | Output: $RUN_CSV"

    python sweep_run_optimizer.py \
        --template "$TEMPLATE" \
        --dataset "$DATASET" \
        --steps "$STEPS" \
        --csv "$RUN_CSV" \
        --output-dir "$OUTPUT_DIR" \
        --wait "$WAIT" \
        "${EXTRA_ARGS[@]}"
done

