#!/bin/bash
#SBATCH --job-name=eval_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/manager_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/manager_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=fs-mbz-gpu-[156,407,034,015,039,031,160,497,161,114,638,765,794,664,140,864,027,117,007,088,099,361,358,413,425,420,472,436,496,487,958,951,612,655,591,670,706,693,811,971]
#SBATCH --partition=main

# cd the working directory
cd /mnt/weka/home/haolong.jia/eval/RL-eval || { echo "Failed to change directory"; exit 1; }

# activate the environment
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# execute the manager script, pass all command line arguments
python -u pipeline/main.py "$@"


############### Usage ###############
#### sbatch pipeline/main.sh --benchmark math500_pass1 --prompt_type cot --gpus_per_task 1 --num_node 1
