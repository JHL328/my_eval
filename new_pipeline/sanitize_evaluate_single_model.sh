#!/bin/bash

# These are now set via command line when submitting the job
# #SBATCH --job-name=san_eval_${MODEL_NAME}
# #SBATCH --output=/mnt/sharefs/users/haolong.jia/result/${TASK_NAME}/logs/${MODEL_NAME}_sanitize.out
# #SBATCH --error=/mnt/sharefs/users/haolong.jia/result/${TASK_NAME}/logs/${MODEL_NAME}_sanitize.err
#SBATCH --partition=cpuonly
#SBATCH --qos=cpuonly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00

cd /mnt/weka/home/haolong.jia/eval/RL-eval || { echo "Failed to change directory"; exit 1; }

source /mnt/weka/home/haolong.jia/miniconda3/bin/activate evalplus-eval

# Save original arguments
ORIGINAL_ARGS=("$@")

# Parse parameters from command line arguments
TASK_NAME="default_task"
MODEL_NAME="default_model"
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
        --model)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                MODEL_NAME="$2"
                shift 2
            else
                shift
            fi
            ;;
        --model=*)
            MODEL_NAME="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Running sanitize and evaluate for model: ${MODEL_NAME} on task: ${TASK_NAME}"
echo "Start time: $(date)"

# Call the main Python orchestration script with --step sanitize_evaluate and --model flags
python -u new_pipeline/evaluate_code.py "${ORIGINAL_ARGS[@]}" --step sanitize_evaluate --model "${MODEL_NAME}"

if [ $? -eq 0 ]; then
    echo "Sanitize and evaluate completed successfully for model: ${MODEL_NAME}"
    echo "End time: $(date)"
else
    echo "Sanitize and evaluate failed for model: ${MODEL_NAME}"
    echo "End time: $(date)"
    exit 1
fi