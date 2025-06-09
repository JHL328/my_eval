#!/bin/bash

#SBATCH --job-name=gpqa_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/gpqa_manager.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/gpqa_manager.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=main

cd /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline || { echo "Failed to change directory"; exit 1; }

source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval

python -u evaluate_gpqa.py --submit_jobs "$@"
