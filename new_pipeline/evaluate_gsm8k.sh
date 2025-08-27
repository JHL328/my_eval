#!/bin/bash

# =============================================================================
# Universal Evaluation Batch Script for GSM8K and Math500
# =============================================================================
# This script is the unified Slurm batch entry for both GSM8K and Math500 tasks.
# It submits a manager job that will further submit per-model jobs for evaluation.
#
# USAGE:
#   sbatch evaluate_gsm8k.sh --task <TASK_NAME> [other options]
#
# PARAMETERS:
#   --task <TASK_NAME>   : Task to run, either 'gsm8k' or 'math500' (default: gsm8k)
#   --reforce            : (Optional) Force rerun all models even if result.csv exists
#   (other options are passed to evaluate_gsm8k.py)
#
# EXAMPLES:
#   sbatch evaluate_gsm8k.sh --task gsm8k
#   sbatch evaluate_gsm8k.sh --task math500 --reforce
#
# OUTPUT:
#   - Main manager logs: /mnt/weka/home/haolong.jia/eval/runs/<task>_manager.out/err
#   - Per-model logs:    /mnt/sharefs/users/haolong.jia/result/<task>_passXX/<model>/slurm.out/err
#   - Results:           /mnt/sharefs/users/haolong.jia/result/<task>_passXX/passk.json
# =============================================================================

#SBATCH --job-name=eval_manager_${SLURM_JOB_ID}
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/eval_manager_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/eval_manager_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=main

cd /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline || { echo "Failed to change directory"; exit 1; }

source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval

echo "🚀 Running evaluation batch manager for task: $@"

# Parse task argument
TASK_NAME=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--task" ]]; then
        TASK_NAME="$arg"
        break
    fi
    prev_arg="$arg"
done

# Route to appropriate script based on task
if [[ "$TASK_NAME" == "math500" ]]; then
    echo "Routing to evaluate_math500.py for Math500 evaluation"
    # Filter out --task argument when calling evaluate_math500.py
    ARGS_WITHOUT_TASK=""
    skip_next=false
    for arg in "$@"; do
        if [[ "$skip_next" == true ]]; then
            skip_next=false
            continue
        fi
        if [[ "$arg" == "--task" ]]; then
            skip_next=true
            continue
        fi
        ARGS_WITHOUT_TASK="$ARGS_WITHOUT_TASK $arg"
    done
    python -u evaluate_math500.py --submit_jobs $ARGS_WITHOUT_TASK
else
    python -u evaluate_gsm8k.py --submit_jobs "$@"
fi
