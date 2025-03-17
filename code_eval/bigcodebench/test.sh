#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --job-name=lm_eval

# Load necessary modules and environment
module load cuda/12.4
source ~/miniconda3/bin/activate bigcodebench-eval

# Accept the root path as the first argument; if not provided, use the script's directory.
ROOT_DIR="$1"
# Accept the model path as the second argument; this should be an absolute path.
MODEL_DIR="$2"

# Create output directories relative to the ROOT_DIR
mkdir -p "${ROOT_DIR}/printout/output_file"
mkdir -p "${ROOT_DIR}/printout/error_file"

# Set NCCL and other environment variables
export NCCL_DEBUG=WARN
export PYTORCH_NO_NVML=1
export HF_ALLOW_CODE_EVAL=1
export CUDA_LAUNCH_BLOCKING=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

set -e

current_time=$(date +%Y%m%d_%H%M%S)

# Redirect stdout and stderr to log files within the root directory
exec >> "${ROOT_DIR}/printout/output_file/output_eval_${SLURM_JOB_ID}_${current_time}.out" 2>> "${ROOT_DIR}/printout/error_file/error_eval_${SLURM_JOB_ID}_${current_time}.err"

# Define other parameters relative to ROOT_DIR
TP=4

OUTPUT_DIR="${ROOT_DIR}"

mkdir -p "${OUTPUT_DIR}"

run_benchmark() {
  SPLIT=$1
  SUBSET=$2

  # Generate code completions
  python generate.py \
    --model "${MODEL_DIR}" \
    --split "${SPLIT}" \
    --subset "${SUBSET}" \
    --greedy \
    --bs 1 \
    --temperature 0 \
    --n_samples 1 \
    --resume \
    --backend vllm \
    --tp "${TP}" \
    --save_path "${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion.jsonl"
    #--chat_mode

  # Sanitize and calibrate the generated samples
  python sanitize.py \
    --samples "${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion.jsonl" \
    --calibrate

  # Evaluate the sanitized and calibrated completions
  python evaluate.py \
    --split "${SPLIT}" \
    --subset "${SUBSET}" \
    --no-gt \
    --samples "${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion-sanitized-calibrated.jsonl"
}

# Run benchmarks for different configurations
run_benchmark instruct full
run_benchmark complete full
run_benchmark instruct hard
run_benchmark instruct full