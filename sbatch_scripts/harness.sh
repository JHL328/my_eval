#!/bin/bash
#SBATCH --job-name=eval_aime
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

MODEL_NAME_OR_PATH=$1
source /mbz/users/yuqi.wang/miniconda3/bin/activate harness-eval

srun bash -c "lm-eval --model_args=\"pretrained=$MODEL_NAME_OR_PATH,revision=main,dtype=auto\" \
--tasks=$2 \
--batch_size=auto \
--log_samples \
--output_path=/mbz/users/yuqi.wang/RL-Eval/results/$2"
