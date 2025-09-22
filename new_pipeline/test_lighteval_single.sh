#!/bin/bash
#SBATCH --job-name=test_lighteval_gpqa
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/test_lighteval_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/test_lighteval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --time=2:00:00

# =============================================================================
# Test Script for Single Model Evaluation using lighteval with YAML Config
# =============================================================================
# This script tests a single SFT model on GPQA Diamond task using YAML configuration
#
# USAGE:
#   sbatch test_lighteval_single.sh
# =============================================================================

echo "🧪 Starting lighteval single model test"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Test parameters
MODEL_PATH="/mnt/sharefs/users/haolong.jia/RL-model/sft/lonely_cone_0/checkpoint-27358"
MODEL_NAME="test_lonely_cone"
TASK="gpqa_diamond"
OUTPUT_DIR="/mnt/sharefs/users/haolong.jia/result/${TASK}_sft_test/${MODEL_NAME}"
CONFIG_FILE="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/configs/gpqa_diamond_test_config.yaml"

echo "Test Configuration:"
echo "  Config File: $CONFIG_FILE"
echo "  Model Path: $MODEL_PATH"
echo "  Model Name: $MODEL_NAME"
echo "  Task: $TASK"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Activate conda environment
echo "Activating conda environment..."
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# Change to lighteval directory
echo "Changing to lighteval directory..."
cd /mnt/weka/home/haolong.jia/eval/RL-eval/lighteval

# Check if lighteval is accessible
echo "Checking lighteval installation..."
python -c "import lighteval; print(f'lighteval version: {lighteval.__version__ if hasattr(lighteval, \"__version__\") else \"unknown\"}')" || echo "Warning: Could not check lighteval version"

# Create model-specific config from template
echo "Creating model-specific configuration..."
MODEL_CONFIG_FILE="${OUTPUT_DIR}/model_config.yaml"
python -c "
import yaml
import os

# Load the test config
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

# Create model-specific config
model_config = {
    'model_parameters': config['model_parameters'].copy()
}

# Set the model path
model_config['model_parameters']['model_name'] = '${MODEL_PATH}'

# Ensure output directory exists
os.makedirs('${OUTPUT_DIR}', exist_ok=True)

# Write model config
with open('${MODEL_CONFIG_FILE}', 'w') as f:
    yaml.dump(model_config, f, default_flow_style=False)

print(f'Model config created at: ${MODEL_CONFIG_FILE}')
"

# Run lighteval evaluation with YAML config
echo ""
echo "Starting evaluation with YAML configuration..."
echo "=============================================="

python -m lighteval vllm \
    "${MODEL_CONFIG_FILE}" \
    "lighteval|gpqa:diamond|0" \
    --output-dir="${OUTPUT_DIR}" \
    --save-details \
    --dataset-loading-processes=8 \
    --max-samples=5

EVAL_STATUS=$?

echo ""
echo "Evaluation completed with status: $EVAL_STATUS"

# Post-process: Find and move the results file from nested directory structure
echo "Post-processing results..."

# Find the actual results JSON file (lighteval creates it in a nested path)
RESULTS_FILE=$(find "${OUTPUT_DIR}/results" -name "results_*.json" -type f 2>/dev/null | head -1)

if [ -n "$RESULTS_FILE" ]; then
    echo "Found results file: $RESULTS_FILE"
    # Copy to the expected location
    cp "$RESULTS_FILE" "${OUTPUT_DIR}/results.json"
    echo "Copied to: ${OUTPUT_DIR}/results.json"

    # Process results to generate metrics.txt
    echo "Processing results to generate metrics..."
    python -c "
import json
import os
import glob

results_path = '${OUTPUT_DIR}/results.json'
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)

    print('Results structure:')
    print(f'  Keys: {list(data.keys())}')

    if 'results' in data:
        print(f'  Results keys: {list(data[\"results\"].keys())}')

        # Extract metrics
        metrics = {}
        for task, result in data['results'].items():
            print(f'  Task {task}:')
            if isinstance(result, dict):
                print(f'    Metrics: {list(result.keys())}')
                # For GPQA, look for pass@k metrics
                if 'gpqa' in '${TASK}'.lower():
                    for metric_name, metric_value in result.items():
                        if 'pass@k' in metric_name.lower() or 'pass_at_k' in metric_name.lower():
                            metrics[task] = metric_value
                            print(f'    {metric_name}: {metric_value}')
                            break
                # For other tasks, look for accuracy
                else:
                    for metric_name, metric_value in result.items():
                        if 'accuracy' in metric_name.lower() or 'acc' in metric_name.lower() or 'exact_match' in metric_name.lower():
                            metrics[task] = metric_value
                            print(f'    {metric_name}: {metric_value}')
                            break

        # Write metrics.txt
        metrics_path = '${OUTPUT_DIR}/metrics.txt'
        with open(metrics_path, 'w') as f:
            if 'gpqa' in '${TASK}'.lower():
                # For GPQA, write pass@1 score
                for task, score in metrics.items():
                    f.write(f'pass@1: {score:.4f}\n')
            else:
                for task, score in metrics.items():
                    f.write(f'accuracy: {score:.4f}\n')

        print(f'')
        print(f'Metrics saved to {metrics_path}')

        # Display final metric
        if metrics:
            avg_score = sum(metrics.values()) / len(metrics)
            print(f'')
            if 'gpqa' in '${TASK}'.lower():
                print(f'🎯 Final Pass@1: {avg_score:.4f}')
            else:
                print(f'🎯 Final Accuracy: {avg_score:.4f}')
else:
    print('❌ Results file not found')
"
    echo "✅ Results processed successfully"
else
    echo "❌ No results file found in ${OUTPUT_DIR}/results/"
fi

# List output files
echo ""
echo "Output files:"
ls -la $OUTPUT_DIR/

echo ""
echo "🏁 Test completed!"
echo "Check logs at: /mnt/weka/home/haolong.jia/eval/runs/test_lighteval_${SLURM_JOB_ID}.out"