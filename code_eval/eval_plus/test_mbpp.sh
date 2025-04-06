#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --job-name=lm_eval

# Load necessary modules and environment
module load cuda/12.4
source ~/miniconda3/bin/activate evalplus_env
cd /mbz/users/liyuan/Qwen2.5-Coder/qwencoder-eval/instruct/eval_plus


# Set NCCL to debug only warnings
export NCCL_DEBUG=WARN

# Disable NVML if not required and set additional env variables
export PYTORCH_NO_NVML=1
export HF_ALLOW_CODE_EVAL=1
export CUDA_LAUNCH_BLOCKING=1

# Set additional environment variables
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

export PATH=./vllm/bin:$PATH
export PYTHONPATH=$PYTHONPATH:./eval_plus/evalplus

current_time=$(date +%Y%m%d_%H%M%S)

TP=4
ROOT_DIR="$1"
# Accept the model path as the second argument; this should be an absolute path.
MODEL_DIR="$2"
MODEL="$3"

# Create output directories relative to the ROOT_DIR
mkdir -p "${ROOT_DIR}/printout/output_file"
mkdir -p "${ROOT_DIR}/printout/error_file"

exec >> "${ROOT_DIR}/printout/output_file/output_humaneval_${SLURM_JOB_ID}_${current_time}.out" 2>> "${ROOT_DIR}/printout/error_file/error_humaneval_${SLURM_JOB_ID}_${current_time}.err"

mkdir -p ${ROOT_DIR}/mbpp/${MODEL}/response

pip install datamodel_code_generator anthropic mistralai google-generativeai


echo "EvalPlus: ${MODEL_DIR}, ROOT_DIR ${ROOT_DIR}"

python generate.py \
  --model_type qwen2 \
  --model_size chat \
  --model_path ${MODEL_DIR} \
  --bs 16 \
  --temperature 0.6 \
  --n_samples 1 \
  --greedy \
  --root  ${ROOT_DIR}/mbpp/${MODEL}/response \
  --dataset mbpp \
  --tensor-parallel-size ${TP}

python -m evalplus.sanitize --samples   ${ROOT_DIR}/mbpp/${MODEL}/response


evalplus.evaluate \
  --dataset mbpp \
  --samples ${ROOT_DIR}/mbpp/${MODEL}/response > ${ROOT_DIR}/mbpp/${MODEL}/raw_mbpp_results.txt

evalplus.evaluate \
  --dataset mbpp \
  --samples ${ROOT_DIR}/mbpp/${MODEL}/response-sanitized > ${ROOT_DIR}/mbpp/${MODEL}/mbpp_results.txt
