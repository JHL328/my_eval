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
#                         - gpqa
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
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--task" ]]; then
        TASK_NAME="$arg"
        break
    elif [[ "$arg" == --task=* ]]; then
        TASK_NAME="${arg#*=}"
        break
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

# set output directory based on TASK_NAME
if [[ "$TASK_NAME" == "mmlu_flan_cot_fewshot_pass16" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_flan_pass16"
elif [[ "$TASK_NAME" == "mmlu_pro_pass16" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_pro_pass16"
elif [[ "$TASK_NAME" == "bbh_pass16" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/bbh_pass16"
elif [[ "$TASK_NAME" == "mmlu" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu"
elif [[ "$TASK_NAME" == "gpqa" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/gpqa_pass32_new"
else
    echo "Error: Unknown task name '$TASK_NAME' for setting OUTPUT_DIR in evaluate.sh."
    exit 1
fi

echo "🔍 Monitoring directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" # ensure directory exists

# --- add monitoring ---
CHECK_INTERVAL=60   # check every 60 seconds
TIMEOUT=600         # 10 minutes = 600 seconds

# Function to check if all models are complete based on result.json files
check_all_models_complete() {
    # only count model folders (exclude job_scripts logs scripts)
    local model_dirs=($(find "$OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type d \
        ! -name 'job_scripts' \
        ! -name 'logs' \
        ! -name 'scripts'))
    if [[ "${#model_dirs[@]}" -eq 0 ]]; then
        # no model folders, means not started, not complete
        return 1
    fi
    local incomplete_count=0
    for dir in "${model_dirs[@]}"; do
        if [[ ! -f "$dir/result.json" ]]; then
            ((incomplete_count++))
        fi
    done
    if [[ "$incomplete_count" -eq 0 ]]; then
        return 0 # True, all complete
    else
        return 1 # False, not complete
    fi
}

while true; do
    # Check if all models are already complete before starting/restarting evaluate.py
    if check_all_models_complete; then
        echo "🎉 All models were already complete. Exiting."
        exit 0
    fi

    # start main evaluation task (background)
    echo "🚀 Starting evaluation..."
    START_TS=$(date +%s) # record start time
    # run evaluate
    python -u new_pipeline/evaluate.py "$@" &
    EVAL_PID=$!

    last_change=""

    while kill -0 $EVAL_PID 2>/dev/null; do
        sleep $CHECK_INTERVAL
        now=$(date +%s)

        # Check for overall completion while evaluate.py is still running
        if check_all_models_complete; then
            echo "🎉 All models seem to be complete!"
            kill $EVAL_PID 2>/dev/null # Kill evaluate.py gracefully
            wait $EVAL_PID 2>/dev/null # Wait for it to terminate
            exit 0 # Exit the manager script successfully
        fi

        # only count new/modified csvs since last start
        new_change=$(find "$OUTPUT_DIR" -name "*.csv" -printf "%T@\n" 2>/dev/null | awk -v st="$START_TS" '$1 > st' | sort -n | tail -1)
        if [[ -n "$new_change" ]] && { [[ -z "$last_change" ]] || (( $(echo "$new_change > $last_change" | bc -l) )); }; then
            last_change=$new_change
        fi

        if [[ -n "$last_change" ]]; then
            idle_time=$(echo "$now - $last_change" | bc)
            if (( $(echo "$idle_time > $TIMEOUT" | bc -l) )); then
                echo "⏳ No new csv for $TIMEOUT seconds, restarting evaluation..."
                kill $EVAL_PID 2>/dev/null
                wait $EVAL_PID 2>/dev/null
                # kill all related slurm tasks (if any)
                squeue -u $USER -n '*mmlu*' -h -o '%i' | xargs -r scancel
                break
            fi
        else
            echo "DEBUG: Waiting for new CSV generated after script start."
            idle_time=0
        fi
        echo "DEBUG: last_change=$last_change, now=$now, idle_time=$idle_time"
    done

    # If evaluate.py exited on its own (kill -0 failed), check if it completed successfully
    if check_all_models_complete; then
        echo "🎉 evaluate.py exited successfully and all models are complete. Exiting."
        exit 0
    else
        echo "⚠️ evaluate.py exited unexpectedly or incomplete. Restarting..."
        # No need to scancel here, as it would have been done by timeout or if evaluate.py handles it on crash.
        # The outer loop will just restart evaluate.py.
    fi

done 