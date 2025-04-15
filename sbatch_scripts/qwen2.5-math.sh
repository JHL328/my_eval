#!/bin/bash
#SBATCH --job-name=eval_qwen2.5_math
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err


source /mbz/users/yuqi.wang/miniconda3/bin/activate qwen-eval
cd ./qwen2.5-math/evaluation
# it can support multi-gpu
model_size=$10
if [ $model_size -le 3 ]; then
    srun bash -c "export CUDA_VISIBLE_DEVICES=0 && bash sh/eval.sh $1 $2 $3 $4 $5 $6 $7 $8 $9"
elif [ $model_size -le 7 ]; then
    srun bash -c "export CUDA_VISIBLE_DEVICES=0,1 && bash sh/eval.sh $1 $2 $3 $4 $5 $6 $7 $8 $9"
elif [ $model_size -le 14 ]; then
    srun bash -c "export CUDA_VISIBLE_DEVICES=0,1,2,3 && bash sh/eval.sh $1 $2 $3 $4 $5 $6 $7 $8 $9"
else
    srun bash -c "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && bash sh/eval.sh $1 $2 $3 $4 $5 $6 $7 $8 $9"
fi
