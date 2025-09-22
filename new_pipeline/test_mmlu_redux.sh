#!/bin/bash
#SBATCH --job-name=test_mmlu_redux
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/test_mmlu_redux_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/test_mmlu_redux_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --time=1:00:00

# =============================================================================
# Test Script for MMLU Redux - Direct lighteval approach
# =============================================================================
# This script bypasses evaluate_lighteval.py and calls lighteval directly
# to avoid the multiprocessing issue
#
# USAGE:
#   sbatch test_mmlu_redux.sh
# =============================================================================

echo "🧪 Starting MMLU Redux evaluation test (Direct approach)"
echo "========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# Change to lighteval directory
cd /mnt/weka/home/haolong.jia/eval/RL-eval/lighteval || { echo "Failed to change directory"; exit 1; }

# Test parameters
MODEL_PATH="/mnt/sharefs/users/haolong.jia/RL-model/sft/lonely_cone_0/checkpoint-27358"
OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_redux_sft_test/direct_test"

echo "Test Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Subjects: abstract_algebra, college_mathematics, elementary_mathematics"
echo "  Samples: 20 per subject (for quick testing)"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run lighteval directly (without --save-details to avoid Arrow error)
echo "Running lighteval with 3 MMLU subjects..."
echo "=========================================="

python -m lighteval vllm \
    "model_name=$MODEL_PATH,dtype=auto,trust_remote_code=true,tensor_parallel_size=1,gpu_memory_utilization=0.9,add_special_tokens=true" \
    "lighteval|mmlu_redux_2:abstract_algebra|0,lighteval|mmlu_redux_2:college_mathematics|0,lighteval|mmlu_redux_2:elementary_mathematics|0" \
    --output-dir="$OUTPUT_DIR" \
    --dataset-loading-processes=1 \
    --max-samples=20 2>&1 | tee ${OUTPUT_DIR}/evaluation.log

EVAL_STATUS=$?

echo ""
echo "Evaluation completed with status: $EVAL_STATUS"

# Check output
OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/mmlu_redux_sft_test/${MODEL_NAME}"
echo ""
echo "Checking output directory: $OUTPUT_DIR"
echo "========================================="

if [ -d "$OUTPUT_DIR" ]; then
    echo "Output files:"
    ls -la $OUTPUT_DIR/

    # Check for model config
    if [ -f "$OUTPUT_DIR/model_config.yaml" ]; then
        echo ""
        echo "Model config created ✓"
    fi

    # Check for job script
    if [ -f "$OUTPUT_DIR/mmlu_redux_${MODEL_NAME}.sh" ]; then
        echo "Job script created ✓"
    fi

    # Wait a moment for results processing
    echo ""
    echo "Waiting for results processing..."
    sleep 10

    # Display metrics if available
    if [ -f "$OUTPUT_DIR/metrics.txt" ]; then
        echo ""
        echo "Metrics:"
        echo "--------"
        cat $OUTPUT_DIR/metrics.txt
    else
        echo ""
        echo "Metrics file not yet generated (check job logs)"
    fi

    # Check results.json
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo ""
        echo "Results file exists ✓"
        echo "File size: $(stat -c%s "$OUTPUT_DIR/results.json") bytes"

        # Extract scores for tested subjects
        echo ""
        echo "Extracting scores for tested subjects:"
        python -c "
import json
import os

results_file = '$OUTPUT_DIR/results.json'
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)

    if 'results' in data:
        for task_name, metrics in data['results'].items():
            if 'mmlu_redux' in task_name:
                subject = task_name.split(':')[-1].replace('|0', '')
                score = metrics.get('accuracy', metrics.get('acc', 'N/A'))
                print(f'  {subject}: {score}')
"
    else
        echo ""
        echo "⏳ Results file not yet generated"
        echo "Check the running job with: squeue -u $USER"
        echo "Monitor logs at: $OUTPUT_DIR/slurm.out"
    fi

    # Show job status
    echo ""
    echo "Checking SLURM job status..."
    squeue -u $USER -n "mmlu_redux_${MODEL_NAME}" 2>/dev/null || echo "No active job found"

else
    echo "❌ Output directory not found"
fi

echo ""
echo "🏁 Test script completed!"
echo ""
echo "To monitor the evaluation job:"
echo "  squeue -u $USER"
echo "  tail -f $OUTPUT_DIR/slurm.out"
echo ""
echo "Main test log: /mnt/weka/home/haolong.jia/eval/runs/test_mmlu_redux_${SLURM_JOB_ID}.out"