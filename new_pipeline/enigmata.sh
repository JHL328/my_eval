#!/bin/bash

#SBATCH --job-name=enigmata_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/enigmata_manager_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/enigmata_manager_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --time=2:00:00

# =============================================================================
# Enigmata Evaluation Manager
# =============================================================================
# This script iterates over all models of a given type (base/sft) defined in model.py
# and submits a separate GPU job for each model to run test_enigmata.py.
#
# Usage: sbatch enigmata.sh --model-type <base|sft>
# =============================================================================

cd /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline || { echo "Failed to change directory"; exit 1; }

# Activate environment to access model.py dependencies
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval

MODEL_TYPE="sft"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Starting Enigmata Manager for model type: $MODEL_TYPE"

# ---------------------------------------------------------
# Extract model list using inline Python (no temp file)
# ---------------------------------------------------------
# Use a heredoc to store the python script in a variable, then pass to python -c
read -r -d '' PYTHON_SCRIPT <<EOF
import sys
from pathlib import Path
# Add current directory to path to find model.py
sys.path.append(str(Path(".").resolve()))
from model import get_model_map_by_type

try:
    # Get models based on the passed type
    model_type_arg = "${MODEL_TYPE}"
    print(f"DEBUG: Fetching models for type: {model_type_arg}", file=sys.stderr)
    model_map = get_model_map_by_type(model_type_arg)
    
    if not model_map:
        print(f"WARNING: No models found for type {model_type_arg}", file=sys.stderr)
        
    for path, name in model_map.items():
        print(f"{path}|{name}")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo "🔍 Fetching model list..."
# Execute python script and read output into array
mapfile -t MODEL_ENTRIES < <(python -c "$PYTHON_SCRIPT")

if [ ${#MODEL_ENTRIES[@]} -eq 0 ]; then
    echo "❌ No models found or error occurred in model retrieval."
    exit 1
fi

echo "✅ Found ${#MODEL_ENTRIES[@]} models to process."

# ---------------------------------------------------------
# Submit Job for each model
# ---------------------------------------------------------
for entry in "${MODEL_ENTRIES[@]}"; do
    # Split "path|name"
    MODEL_PATH="${entry%|*}"
    MODEL_NAME="${entry#*|}"
    
    JOB_NAME="enig_${MODEL_NAME}"
    
    echo "📨 Submitting job for model: $MODEL_NAME"
    
    # Directly submit the python script which contains #SBATCH configuration
    # We only specify the job name here; resources are defined in test_enigmata.py
    sbatch \
        --job-name="${JOB_NAME}" \
        --chdir="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline" \
        test_enigmata.py \
        --model "${MODEL_PATH}" \
        --model-type "${MODEL_TYPE}" \
        --run-eval

done

echo "✅ All submission jobs dispatched."
