#!/bin/bash

# =============================================================================
# Node Benchmark Submitter Script (8-GPU Models)
# =============================================================================
# This script is a unified entry point for submitting benchmark jobs
# for large models that require 8 GPUs (full node).
#
# USAGE:
#   bash submitter.sh --task <TASK_NAME> [--reforce]
#
# EXAMPLES:
#   bash submitter.sh --task aime24
#   bash submitter.sh --task aime25
#   bash submitter.sh --task math500
#   bash submitter.sh --task amc23
#   bash submitter.sh --task aime24_math
#   bash submitter.sh --task aime25_math
#   bash submitter.sh --task aime24 --reforce
#
# SUPPORTED TASKS:
#   - aime24:  AIME 2024 (Avg@32) [qwen-eval env]
#   - aime25:  AIME 2025 (Avg@32) [qwen-eval env]
#   - math500: MATH500 (Avg@4, temperature=0.6, top_p=0.95)    [qwen-eval env]
#   - amc23:   AMC23 (Avg@16, temperature=0.6, top_p=0.95)     [qwen-eval env]
#   - aime24_math: AIME 2024 (Avg@32, qwen2.5-math)            [qwen-eval env]
#   - aime25_math: AIME 2025 (Avg@32, qwen2.5-math)            [qwen-eval env]
#
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
TASK_NAME=""
REFORCE_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                TASK_NAME="$2"
                shift 2
            else
                echo "[submitter.sh] ERROR: --task requires a value"
                exit 1
            fi
            ;;
        --task=*)
            TASK_NAME="${1#*=}"
            shift
            ;;
        --reforce)
            REFORCE_FLAG="--reforce"
            shift
            ;;
        *)
            echo "[submitter.sh] WARNING: Unknown argument: $1"
            shift
            ;;
    esac
done

# Validate task argument
if [[ -z "$TASK_NAME" ]]; then
    echo "[submitter.sh] ERROR: --task argument is required."
    echo ""
    echo "Usage: bash submitter.sh --task <TASK_NAME> [--reforce]"
    echo ""
    echo "Supported tasks:"
    echo "  - aime24:  AIME 2024 (Avg@32)"
    echo "  - aime25:  AIME 2025 (Avg@32)"
    echo "  - math500: MATH500 (Avg@4)"
    echo "  - amc23:   AMC23 (Avg@16)"
    echo "  - aime24_math: AIME 2024 (Avg@32, qwen2.5-math)"
    echo "  - aime25_math: AIME 2025 (Avg@32, qwen2.5-math)"
    exit 1
fi

# Route to appropriate evaluation script with correct conda environment
case "$TASK_NAME" in
    aime24|aime25)
        echo "[submitter.sh] Activating conda environment: qwen-eval"
        source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval
        echo "[submitter.sh] Running AIME evaluation: $TASK_NAME"
        echo "[submitter.sh] Settings: Avg@32, temp=0.6, top_p=0.95, max_model_len=16384, max_tokens=15260, 8 GPUs"
        python3 "${SCRIPT_DIR}/evaluate_aime_2.py" --task "$TASK_NAME" --n_gpu 8 --apply_chat $REFORCE_FLAG
        ;;
    math500)
        echo "[submitter.sh] Activating conda environment: qwen-eval"
        source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval
        echo "[submitter.sh] Running MATH500 evaluation"
        echo "[submitter.sh] Settings: Avg@4, temperature=0.6, top_p=0.95, num_shots=0, apply_chat_template"
        python3 "${SCRIPT_DIR}/evaluate_math.py" --task math500 $REFORCE_FLAG
        ;;
    amc23)
        echo "[submitter.sh] Activating conda environment: qwen-eval"
        source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval
        echo "[submitter.sh] Running AMC23 evaluation"
        echo "[submitter.sh] Settings: Avg@16, temperature=0.6, top_p=0.95, num_shots=0, apply_chat_template"
        python3 "${SCRIPT_DIR}/evaluate_math.py" --task amc23 $REFORCE_FLAG
        ;;
    aime24_math)
        echo "[submitter.sh] Activating conda environment: qwen-eval"
        source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval
        echo "[submitter.sh] Running AIME24 evaluation (qwen2.5-math)"
        echo "[submitter.sh] Settings: Avg@32, temperature=0.6, top_p=0.95, num_shots=0, apply_chat_template"
        python3 "${SCRIPT_DIR}/evaluate_math.py" --task aime24 $REFORCE_FLAG
        ;;
    aime25_math)
        echo "[submitter.sh] Activating conda environment: qwen-eval"
        source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval
        echo "[submitter.sh] Running AIME25 evaluation (qwen2.5-math)"
        echo "[submitter.sh] Settings: Avg@32, temperature=0.6, top_p=0.95, num_shots=0, apply_chat_template"
        python3 "${SCRIPT_DIR}/evaluate_math.py" --task aime25 $REFORCE_FLAG
        ;;
    *)
        echo "[submitter.sh] ERROR: Unknown task: $TASK_NAME"
        echo ""
        echo "Supported tasks:"
        echo "  - aime24:  AIME 2024 (Avg@32)"
        echo "  - aime25:  AIME 2025 (Avg@32)"
        echo "  - math500: MATH500 (Avg@4)"
        echo "  - amc23:   AMC23 (Avg@16)"
        echo "  - aime24_math: AIME 2024 (Avg@32, qwen2.5-math)"
        echo "  - aime25_math: AIME 2025 (Avg@32, qwen2.5-math)"
        exit 2
        ;;
esac

echo "[submitter.sh] Done!"
