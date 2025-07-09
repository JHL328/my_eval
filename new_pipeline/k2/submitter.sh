#!/bin/bash

#SBATCH --job-name=k2-eval-submitter
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/k2_submitter_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/k2_submitter_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --time=24:00:00

# --- 参数解析 ---
# command line is like: sbatch submitter.sh --task bbh
TASK=""
if [[ "$1" == "--task" ]] && [[ -n "$2" ]]; then
    TASK="$2"
else
    echo "Error: Please specify the task using --task [bbh|mmlu|mmlu_flan|mmlu_pro|gsm8k|math500]"
    exit 1
fi

# --- 环境设置 ---
echo "--- Setting up environment ---"
cd /mnt/weka/home/haolong.jia/eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# --- 运行主调度脚本 ---
echo "--- Starting K2+ evaluation dispatcher for task: $TASK ---"
python -u RL-eval/new_pipeline/k2/k2_evaluate.py --eval_task "$TASK"

echo "--- Dispatcher finished ---"
