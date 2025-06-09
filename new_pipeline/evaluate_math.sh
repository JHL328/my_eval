#!/bin/bash
#SBATCH --job-name=gsm8k_batch
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/gsm8k_batch.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/gsm8k_batch.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

cd /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval
which python
export TOKENIZERS_PARALLELISM=false

python3 -u evaluate_math.py
