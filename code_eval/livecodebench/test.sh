#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --job-name=lm_eval

# Load necessary modules and environment
module load cuda/12.4
source ~/miniconda3/bin/activate livecode_eval_env

# Accept ROOT_DIR as the first parameter and MODEL_DIR as the second parameter
ROOT_DIR="$1"
MODEL_DIR="$2"

# Change to livecode_bench directory relative to ROOT_DIR
cd "${ROOT_DIR}/livecode_bench"

# Create output directories relative to ROOT_DIR
mkdir -p "${ROOT_DIR}/printout/output_file"
mkdir -p "${ROOT_DIR}/printout/error_file"

# Set NCCL to debug only warnings
export NCCL_DEBUG=WARN

# Disable NVML if not required and set additional env variables
export PYTORCH_NO_NVML=1
export HF_ALLOW_CODE_EVAL=1
export CUDA_LAUNCH_BLOCKING=1

current_time=$(date +%Y%m%d_%H%M%S)

# Redirect all stdout and stderr to your log files
exec >> "${ROOT_DIR}/printout/output_file/output_eval_${SLURM_JOB_ID}_${current_time}.out" 2>> "${ROOT_DIR}/printout/error_file/error_eval_${SLURM_JOB_ID}_${current_time}.err"

OUTPUT_DIR="${ROOT_DIR}"

python -m lcb_runner.runner.main --model Qwen/CodeQwen1.5-7B-Chat --model_path "${MODEL_DIR}" --output_name "qwen2" --scenario codegeneration --evaluate --tensor_parallel_size 4 --output_dir "${OUTPUT_DIR}"
saved_eval_all_file="${OUTPUT_DIR}/log.json"
python -m lcb_runner.evaluation.compute_scores --eval_all_file "${saved_eval_all_file}" --start_date 2024-05-01
