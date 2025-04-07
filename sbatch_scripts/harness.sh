#!/bin/bash
#SBATCH --job-name=eval_harness
#SBATCH --partition=higherprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --exclude=mbz-h100-018

MODEL_NAME_OR_PATH=$1
source /mbz/users/yuqi.wang/miniconda3/bin/activate harness-eval

TASK=$2
if [[ $TASK = "ifeval" ]]; then
    srun bash -c "lm_eval --model vllm \
        --model_args pretrained=$MODEL_NAME_OR_PATH,tensor_parallel_size=8,dtype=bfloat16,max_model_len=$4,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks $2 \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $5 \
        --num_fewshot $3 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path=$6/$2"
else
    srun bash -c "lm_eval --model vllm \
        --model_args pretrained=$MODEL_NAME_OR_PATH,tensor_parallel_size=8,dtype=bfloat16,max_model_len=$4,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks $2 \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $5 \
        --num_fewshot $3 \
        --output_path=$6/$2"
fi