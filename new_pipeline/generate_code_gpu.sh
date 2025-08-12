#!/bin/bash

#SBATCH --job-name=codegen_gpu_${TASK_NAME}
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/eval_manager_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/eval_manager_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --partition=main
#SBATCH --time=4:00:00

cd /mnt/weka/home/haolong.jia/eval/RL-eval || { echo "Failed to change directory"; exit 1; }

source /mnt/weka/home/haolong.jia/miniconda3/bin/activate evalplus-eval

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

PYTHON_OUT_FILE="/mnt/weka/home/haolong.jia/eval/runs/${TASK_NAME}_gpu.out"
PYTHON_ERR_FILE="/mnt/weka/home/haolong.jia/eval/runs/${TASK_NAME}_gpu.err"

echo "Running all steps (generation, sanitize, evaluate) on GPU node for task: ${TASK_NAME}"
echo "Python output will be saved to: ${PYTHON_OUT_FILE}"
echo "Python errors will be saved to: ${PYTHON_ERR_FILE}"

# Run all steps: Generate code samples, sanitize, and evaluate
echo "=========================================="
echo "Running Code Generation + Sanitize + Evaluate"
echo "=========================================="
python -u new_pipeline/evaluate_code_simple.py "${ORIGINAL_ARGS[@]}" > "${PYTHON_OUT_FILE}" 2> "${PYTHON_ERR_FILE}"

if [ $? -eq 0 ]; then
    echo "All steps completed successfully for task: ${TASK_NAME}"
    echo "Note: Each model job runs generation + sanitize + evaluate on the same GPU node."
else
    echo "Pipeline failed for task: ${TASK_NAME}"
    exit 1
fi