#!/bin/bash
#SBATCH --job-name=pass_k_result
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/pass.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/pass.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --exclude=fs-mbz-gpu-[156,407,034,015,039,031,160,497,161,114,638,765,794,664,140,864,027,117,007,088,099,361,358,413,425,420,472,436,496,487,958,951,612,655,591,670,706,693,811,971]
#SBATCH --partition=main

# Create runs directory if it doesn't exist
mkdir -p /mnt/weka/home/haolong.jia/eval/runs

# cd the working directory
cd /mnt/weka/home/haolong.jia/eval/RL-eval || { echo "Failed to change directory"; exit 1; }

# activate the environment
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# execute the pass_k_result script, pass all command line arguments
python -u pipeline/pass_k_result.py "$@"


############### Usage ###############
#### sbatch pipeline/pass_k_result.sh --benchmark /mnt/sharefs/users/haolong.jia/result/mmlu_flan_cot_fewshot_pass16 --list
#### sbatch pipeline/pass_k_result.sh --benchmark /mnt/sharefs/users/haolong.jia/result/mmlu_flan_cot_fewshot_pass16 --model Llama-3.2-1B
#### sbatch pipeline/pass_k_result.sh --benchmark /mnt/sharefs/users/haolong.jia/result/mmlu_flan_cot_fewshot_pass16
