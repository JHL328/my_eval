#!/bin/bash

# MMLU-Redux evaluation batch submission script

# Parse command line arguments
MODEL_TYPE=${1:-base}  # default to base models
ACTION=${2:-submit}     # submit or summarize

echo "=================================="
echo "MMLU-Redux Evaluation Script"
echo "Model Type: $MODEL_TYPE"
echo "Action: $ACTION"
echo "=================================="

# Activate conda environment
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval

# Change to script directory
cd /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline

if [ "$ACTION" == "submit" ]; then
    echo "Submitting MMLU-Redux evaluation jobs for $MODEL_TYPE models..."
    python3 evaluate_mmlu_redux.py --submit_jobs --type $MODEL_TYPE
elif [ "$ACTION" == "summarize" ]; then
    echo "Summarizing MMLU-Redux results for $MODEL_TYPE models..."
    python3 evaluate_mmlu_redux.py --summarize --type $MODEL_TYPE
else
    echo "Unknown action: $ACTION"
    echo "Usage: $0 [base|sft] [submit|summarize]"
    exit 1
fi

echo "=================================="
echo "Script completed!"
echo "==================================">