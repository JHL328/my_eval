#!/bin/bash
#SBATCH --job-name=eval_qwen2.5_math
#SBATCH --partition=higherprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err


source /mbz/users/yuqi.wang/miniconda3/bin/activate qwen-eval
cd ./qwen2.5-math/evaluation
# it can support multi-gpu
srun bash -c "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && bash sh/eval.sh $1 $2 $3 $4 $5"