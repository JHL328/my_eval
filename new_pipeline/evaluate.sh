#!/bin/bash

# =============================================================================
# Evaluation Manager Script
# =============================================================================
# 
# This script manages the evaluation of multiple language models on various
# benchmark tasks (e.g., mmlu, bbh, gpqa) using the custom evaluation pipeline.
# It supports automatic monitoring, job restarting, and post-processing.
#
# USAGE:
#   sbatch RL-eval/new_pipeline/evaluate.sh --task <TASK_NAME>
#
# PARAMETERS:
#   --task <TASK_NAME>  : The benchmark task to evaluate (required)
#                         Supported tasks include:
#                         - mmlu_flan_cot_fewshot_pass16
#                         - mmlu_pro_pass16
#                         - bbh_pass16
#                         - mmlu
#
# EXAMPLES:
#   # Evaluate all models on the BBH benchmark
#   sbatch RL-eval/new_pipeline/evaluate.sh --task bbh_pass16
#
#   # Evaluate all models on MMLU
#   sbatch RL-eval/new_pipeline/evaluate.sh --task mmlu
#
#
# OUTPUT:
#   - Main Python output: /mnt/weka/home/haolong.jia/eval/runs/<TASK_NAME>.out
#   - Main Python errors: /mnt/weka/home/haolong.jia/eval/runs/<TASK_NAME>.err
#   - Shell script logs: /mnt/weka/home/haolong.jia/eval/runs/eval_manager_<JOB_ID>.out/err
#   - Individual model logs: /mnt/sharefs/users/haolong.jia/result/<TASK_NAME>/logs/
#   - Results: /mnt/sharefs/users/haolong.jia/result/<TASK_NAME>/result.json
#
# FEATURES:
#   - Automatically skips models that have already been evaluated
#   - Submits parallel Slurm jobs for each model and task batch
#   - Monitors output directory for progress and completion
#   - Automatically restarts evaluation if idle for too long
#   - Post-processes results and generates overall result.json
#   - Robust to crashes and can resume incomplete tasks
#
# =============================================================================

#SBATCH --job-name=eval_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/eval_manager_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/eval_manager_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=main

cd /mnt/weka/home/haolong.jia/eval/RL-eval || { echo "Failed to change directory"; exit 1; }

source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# --- 优化参数解析，动态生成 python 日志文件名 ---
TASK_NAME="default_task"
MODEL_TYPE="base"
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--task" ]]; then
        TASK_NAME="$arg"
    elif [[ "$arg" == --task=* ]]; then
        TASK_NAME="${arg#*=}"
    elif [[ "$prev_arg" == "--type" ]]; then
        MODEL_TYPE="$arg"
    elif [[ "$arg" == --type=* ]]; then
        MODEL_TYPE="${arg#*=}"
    fi
    prev_arg="$arg"
done

if [[ -z "$TASK_NAME" || "$TASK_NAME" == "default_task" ]]; then
    echo "Error: --task argument not provided or invalid. Usage: sbatch <script> --task <task_name>"
    exit 1
fi

PYTHON_OUT_DIR="/mnt/weka/home/haolong.jia/eval/runs"
mkdir -p "$PYTHON_OUT_DIR"
#PYTHON_OUT_FILE="${PYTHON_OUT_DIR}/${TASK_NAME}.out"
#PYTHON_ERR_FILE="${PYTHON_OUT_DIR}/${TASK_NAME}.err"

echo "Running evaluation for task: ${TASK_NAME}"
#echo "Python output will be saved to: ${PYTHON_OUT_FILE}"
#echo "Python errors will be saved to: ${PYTHON_ERR_FILE}"

# set output directory based on TASK_NAME and MODEL_TYPE
if [[ "$TASK_NAME" == "mmlu_flan_cot_fewshot_pass16" ]]; then
    if [[ "$MODEL_TYPE" == "sft" ]]; then
        OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_flan_pass16_sft"
    else
        OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_flan_pass16"
    fi
elif [[ "$TASK_NAME" == "mmlu_pro_pass16" ]]; then
    if [[ "$MODEL_TYPE" == "sft" ]]; then
        OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_pro_pass16_sft"
    else
        OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_pro_pass16"
    fi
elif [[ "$TASK_NAME" == "bbh_pass16" ]]; then
    if [[ "$MODEL_TYPE" == "sft" ]]; then
        OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/bbh_pass16_sft"
    else
        OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/bbh_pass16"
    fi
elif [[ "$TASK_NAME" == "mmlu" ]]; then
    if [[ "$MODEL_TYPE" == "sft" ]]; then
        OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_sft"
    else
        OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu"
    fi
else
    echo "Error: Unknown task name '$TASK_NAME' for setting OUTPUT_DIR in evaluate.sh."
    exit 1
fi

echo "🔍 Monitoring directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" # ensure directory exists

# Function to check if all models are complete based on result.json files
check_all_models_complete() {
    # Get all model names from correct model map based on MODEL_TYPE
    local python_output=$(python3 -c "
import sys
sys.path.append('new_pipeline')
from model import get_model_map_by_type
model_map = get_model_map_by_type('$MODEL_TYPE')
for model_path, model_name in model_map.items():
    print(model_name)
")
    
    # Convert python output to array
    local all_model_names=()
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            all_model_names+=("$line")
        fi
    done <<< "$python_output"
    
    if [[ "${#all_model_names[@]}" -eq 0 ]]; then
        echo "Warning: No models found in Model_map"
        return 1
    fi
    
    local incomplete_count=0
    local incomplete_models=()
    
    for model_name in "${all_model_names[@]}"; do
        local model_dir="$OUTPUT_DIR/$model_name"
        local result_file="$model_dir/result.json"
        if [[ ! -f "$result_file" ]]; then
            ((incomplete_count++))
            incomplete_models+=("$model_name")
        fi
    done
    
    if [[ "$incomplete_count" -eq 0 ]]; then
        echo "✅ All ${#all_model_names[@]} models are complete!"
        return 0 # True, all complete
    else
        echo "📊 $incomplete_count/${#all_model_names[@]} models still incomplete"
        # Optionally show first few incomplete models for debugging
        if [[ "${#incomplete_models[@]}" -le 10 ]]; then
            echo "Incomplete models: ${incomplete_models[*]}"
        else
            echo "First 10 incomplete models: ${incomplete_models[*]:0:10}"
        fi
        return 1 # False, not complete
    fi
}

# Check if all models are already complete before starting evaluate.py
if check_all_models_complete; then
    echo "🎉 All models were already complete. Exiting."
    exit 0
fi

# Start main evaluation task
echo "🚀 Starting evaluation..."
python -u new_pipeline/evaluate.py "$@"

# Check completion status after evaluate.py finishes
if check_all_models_complete; then
    echo "🎉 All models are complete. Exiting."
    exit 0
else
    echo "⚠️ Some models may be incomplete."
    exit 1
fi 