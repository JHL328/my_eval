#!/bin/bash

#SBATCH --job-name=humaneval_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/humaneval_manager.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/humaneval_manager.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=main

cd /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline || { echo "Failed to change directory"; exit 1; }

source /mnt/weka/home/haolong.jia/miniconda3/bin/activate evalplus-eval

python -u evaluate_humaneval.py "$@"
