#!/bin/bash

#SBATCH --job-name=eval_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/mmlu.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/mmlu.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=main

cd /mnt/weka/home/haolong.jia/eval/RL-eval || { echo "Failed to change directory"; exit 1; }

source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# --- 动态设置监控目录 ---
TASK_NAME=""
for arg in "$@"; do
    if [[ "$arg" == --task* ]]; then
        TASK_NAME="${arg#*=}"
        if [[ "$TASK_NAME" == "--task" ]]; then
            # This is a simple parser for "--task name" format
            eval set -- "$@"
            while [ $# -gt 0 ]; do
                if [ "$1" = "--task" ]; then
                    TASK_NAME="$2"
                    break
                fi
                shift
            done
        fi
        break
    fi
done

if [[ -z "$TASK_NAME" ]]; then
    echo "Error: --task argument not provided."
    exit 1
fi

# 根据 TASK_NAME 设置 OUTPUT_DIR
if [[ "$TASK_NAME" == "mmlu_flan_cot_fewshot_pass16" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_flan_pass16"
elif [[ "$TASK_NAME" == "mmlu_pro_pass16" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_pro_pass16"
elif [[ "$TASK_NAME" == "bbh_pass16" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/bbh_pass16"
elif [[ "$TASK_NAME" == "mmlu" ]]; then
    OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu"
else
    echo "Error: Unknown task name '$TASK_NAME' for setting OUTPUT_DIR in evaluate.sh."
    exit 1
fi

echo "Monitoring directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" # 确保目录存在

# --- 增加监控 ---
CHECK_INTERVAL=60   # 每隔60秒检查一次
TIMEOUT=600         # 10分钟=600秒

while true; do
    # 启动主评估任务（后台）
    python -u new_pipeline/evaluate.py "$@" &
    EVAL_PID=$!

    last_change=$(find "$OUTPUT_DIR" -name "*.csv" -printf "%T@\n" 2>/dev/null | sort -n | tail -1)
    if [[ -z "$last_change" ]]; then
        last_change=$(date +%s)
    fi

    while kill -0 $EVAL_PID 2>/dev/null; do
        sleep $CHECK_INTERVAL
        now=$(date +%s)
        new_change=$(find "$OUTPUT_DIR" -name "*.csv" -printf "%T@\n" 2>/dev/null | sort -n | tail -1)
        if [[ -z "$new_change" ]]; then
            new_change=$last_change
        fi
        if (( $(echo "$new_change > $last_change" | bc -l) )); then
            last_change=$new_change
        fi
        # 检查是否所有模型都完成
        incomplete=$(find "$OUTPUT_DIR" -maxdepth 2 -type d -exec test ! -f {}/result.json \; -print | wc -l)
        # The above line is a bit complex, let's break it down:
        # find "$OUTPUT_DIR" -maxdepth 2 -type d : find directories in OUTPUT_DIR, but don't go too deep
        # -exec test ! -f {}/result.json \; : for each directory found, check if a file named result.json does NOT exist
        # -print : print the name of directories that satisfy the condition
        # wc -l : count the number of such directories
        
        # A simpler check might be to count the number of models in Model_map and compare with completed ones.
        # But this check is more robust as it does not depend on model.py
        
        if [[ "$incomplete" -le 1 ]]; then # <= 1 because the root dir itself is counted
            echo "All models seem to be complete!"
            kill $EVAL_PID 2>/dev/null
            wait $EVAL_PID 2>/dev/null
            exit 0
        fi
        idle_time=$(echo "$now - $last_change" | bc)
        if (( $(echo "$idle_time > $TIMEOUT" | bc -l) )); then
            echo "No new csv for $TIMEOUT seconds, restarting evaluation..."
            kill $EVAL_PID 2>/dev/null
            wait $EVAL_PID 2>/dev/null
            # 杀掉所有相关slurm任务（如果有）
            squeue -u $USER -n '*mmlu*' -h -o '%i' | xargs -r scancel
            break
        fi
    done

done 