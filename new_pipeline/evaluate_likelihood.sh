#!/bin/bash

# =============================================================================
# Likelihood Evaluation Manager Script
# =============================================================================
# 
# This script manages the evaluation of multiple language models on various
# benchmark tasks using lm-evaluation-harness.
#
# USAGE:
#   sbatch RL-eval/new_pipeline/evaluate_likelihood.sh --task <TASK_NAME>
#
# PARAMETERS:
#   --task <TASK_NAME>  : The benchmark task to evaluate (required)
#                         Supported tasks include:
#                         - drop
#                         - arc_easy
#                         - arc_challenge
#                         - hellaswag
#                         - piqa
#                         - winogrande
#                         - triviaqa
#                         - nq_open
#                         - commonsense_qa
#                         - agieval
#                         - openbookqa
#                         - social_iqa
#                         - truthfulqa_mc2
#
# EXAMPLES:
#   # Evaluate all models on the DROP benchmark
#   sbatch RL-eval/new_pipeline/evaluate_likelihood.sh --task drop
#
#   # Evaluate all models on ARC-Easy
#   sbatch RL-eval/new_pipeline/evaluate_likelihood.sh --task arc_easy
#
#   # Alternative syntax with equals sign
#   sbatch RL-eval/new_pipeline/evaluate_likelihood.sh --task=hellaswag
#
# OUTPUT:
#   - Main Python output: /mnt/weka/home/haolong.jia/eval/runs/<TASK_NAME>.out
#   - Main Python errors: /mnt/weka/home/haolong.jia/eval/runs/<TASK_NAME>.err
#   - Shell script logs: /mnt/weka/home/haolong.jia/eval/runs/eval_manager_<JOB_ID>.out/err
#   - Individual model logs: /mnt/sharefs/users/haolong.jia/result/<TASK_NAME>/logs/<MODEL_NAME>.out/err
#   - Results: /mnt/sharefs/users/haolong.jia/result/<TASK_NAME>/result.json
#
# FEATURES:
#   - Automatically skips models that have already been evaluated
#   - Submits parallel Slurm jobs for each model
#   - Waits for all jobs to complete before generating summary
#   - Automatically post-processes results and generates overall result.json
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

# Save original arguments
ORIGINAL_ARGS=("$@")

# Parse --task parameter from command line arguments
TASK_NAME="default_task"
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                TASK_NAME="$2"
                shift 2
            else
                shift
            fi
            ;;
        --task=*)
            TASK_NAME="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Build dynamic output file names based on task
PYTHON_OUT_FILE="/mnt/weka/home/haolong.jia/eval/runs/${TASK_NAME}.out"
PYTHON_ERR_FILE="/mnt/weka/home/haolong.jia/eval/runs/${TASK_NAME}.err"

echo "Running evaluation for task: ${TASK_NAME}"
echo "Python output will be saved to: ${PYTHON_OUT_FILE}"
echo "Python errors will be saved to: ${PYTHON_ERR_FILE}"

# Run Python script with output redirection using original arguments
python -u new_pipeline/evaluate_likelihood.py "${ORIGINAL_ARGS[@]}" > "${PYTHON_OUT_FILE}" 2> "${PYTHON_ERR_FILE}"

# Check exit status and report
if [ $? -eq 0 ]; then
    echo "Evaluation manager completed successfully for task: ${TASK_NAME}"
else
    echo "Evaluation manager failed for task: ${TASK_NAME}"
    exit 1
fi
