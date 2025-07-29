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

echo "Running code generation (Step 1) for task: ${TASK_NAME}"
echo "Python output will be saved to: ${PYTHON_OUT_FILE}"
echo "Python errors will be saved to: ${PYTHON_ERR_FILE}"

# Call the main Python orchestration script with --step generation flag
python -u new_pipeline/evaluate_code.py "${ORIGINAL_ARGS[@]}" --step generation > "${PYTHON_OUT_FILE}" 2> "${PYTHON_ERR_FILE}"

if [ $? -eq 0 ]; then
    echo "Code generation (Step 1) completed successfully for task: ${TASK_NAME}"
    echo "Note: Individual model CPU jobs will be submitted by each GPU job automatically."
else
    echo "Code generation (Step 1) failed for task: ${TASK_NAME}"
    exit 1
fi