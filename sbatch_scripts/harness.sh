#!/bin/bash
#SBATCH --job-name=eval_harness
#SBATCH --partition=higherprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

MODEL_NAME_OR_PATH=$1
source /mbz/users/yuqi.wang/miniconda3/bin/activate harness-eval

# srun bash -c "lm-eval --model_args=\"pretrained=$MODEL_NAME_OR_PATH,revision=main,dtype=auto\" \
# --tasks=$2 \
# --batch_size=auto \
# --log_samples \
# --apply_chat_template \
# --gen_kwargs temperature=0,top_p=1 \
# --output_path=$3/$2"

srun bash -c "lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_OR_PATH,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $2 \
    --batch_size auto \
    --log_samples \
    --apply_chat_template \
    --gen_kwargs temperature=0,top_p=1 \
    --num_fewshot $3 \
    --output_path=$4/$2"