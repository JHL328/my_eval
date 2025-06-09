#!/bin/bash

#SBATCH --job-name=drop_eval_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/drop_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/drop_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=main

cd /mnt/weka/home/haolong.jia/eval/RL-eval || { echo "Failed to change directory"; exit 1; }

source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

python -u new_pipeline/evaluate_likelihood.py "$@"
