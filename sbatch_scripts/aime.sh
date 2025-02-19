#!/bin/bash
#SBATCH --job-name=eval_aime
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err


MODEL_NAME_OR_PATH=$1
source /mbz/users/yuqi.wang/miniconda3/bin/activate qwen-eval
cd ./qwen2.5-math/evaluation
# it can support multi-gpu
srun bash -c "bash sh/eval.sh qwen25-math-cot $MODEL_NAME_OR_PATH"