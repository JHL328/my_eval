#!/bin/bash
#SBATCH --job-name=test_lighteval_yaml
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/test_lighteval_yaml_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/test_lighteval_yaml_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --time=2:00:00

# =============================================================================
# Test Script using evaluate_lighteval.py with YAML Configuration
# =============================================================================
# This script tests the YAML configuration system with evaluate_lighteval.py
#
# USAGE:
#   sbatch test_lighteval_yaml_pipeline.sh
# =============================================================================

echo "🧪 Starting lighteval YAML pipeline test"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Change to pipeline directory
cd /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline

# Test parameters
MODEL_PATH="/mnt/sharefs/users/haolong.jia/RL-model/sft/lonely_cone_0/checkpoint-27358"
MODEL_NAME="test_lonely_cone"
CONFIG_FILE="configs/gpqa_diamond_test_config.yaml"

echo "Test Configuration:"
echo "  Config File: $CONFIG_FILE"
echo "  Model Path: $MODEL_PATH"
echo "  Model Name: $MODEL_NAME"
echo ""

# Check the config file
echo "Configuration file contents:"
echo "============================="
head -n 20 $CONFIG_FILE
echo "..."
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# Run evaluation using evaluate_lighteval.py with YAML config
echo ""
echo "Starting evaluation with evaluate_lighteval.py..."
echo "================================================="

python evaluate_lighteval.py \
    --config $CONFIG_FILE \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --type sft

EVAL_STATUS=$?

echo ""
echo "Evaluation completed with status: $EVAL_STATUS"

# Check output
OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/gpqa_diamond_sft_test/${MODEL_NAME}"
echo ""
echo "Checking output directory: $OUTPUT_DIR"
echo "========================================"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Output files:"
    ls -la $OUTPUT_DIR/

    # Display metrics if available
    if [ -f "$OUTPUT_DIR/metrics.txt" ]; then
        echo ""
        echo "Metrics:"
        cat $OUTPUT_DIR/metrics.txt
    fi

    # Check results.json
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo ""
        echo "Results file exists ✓"
        echo "File size: $(stat -c%s "$OUTPUT_DIR/results.json") bytes"
    fi
else
    echo "❌ Output directory not found"
fi

echo ""
echo "🏁 Test completed!"
echo "Check logs at: /mnt/weka/home/haolong.jia/eval/runs/test_lighteval_yaml_${SLURM_JOB_ID}.out"