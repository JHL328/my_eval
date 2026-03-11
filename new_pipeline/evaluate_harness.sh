#!/bin/bash

#SBATCH --job-name=eval_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/eval_manager_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/eval_manager_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main

cd /mnt/weka/home/haolong.jia/eval/RL-eval || { echo "❌ Failed to change directory"; exit 1; }

EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-base}"
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate "${EVAL_CONDA_ENV}"

ORIGINAL_ARGS=("$@")
# the task can be mmlu_redux_generative or ifeval
TASK_NAME="mmlu_redux_generative"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            if [[ -n "$2" && "$2" != --* ]]; then
                TASK_NAME="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        --task=*)
            TASK_NAME="${1#*=}"
            shift 1
            ;;
        *)
            shift 1
            ;;
    esac
done

OUT_FILE="/mnt/weka/home/haolong.jia/eval/runs/${TASK_NAME}.out"
ERR_FILE="/mnt/weka/home/haolong.jia/eval/runs/${TASK_NAME}.err"

echo "▶️ running evaluate_harness.py --task ${TASK_NAME}"
python -u new_pipeline/evaluate_harness.py "${ORIGINAL_ARGS[@]}" >"${OUT_FILE}" 2>"${ERR_FILE}"
STATUS=$?

if [[ ${STATUS} -eq 0 ]]; then
    echo "✅ completed evaluate_harness, logs located at ${OUT_FILE}"
else
    echo "❌ evaluate_harness failed, please check ${ERR_FILE}"
fi

exit ${STATUS}
