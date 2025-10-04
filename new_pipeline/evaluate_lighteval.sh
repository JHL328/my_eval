#!/bin/bash

# =============================================================================
# Evaluation Batch Script for MMLU Redux and GPQA Diamond using lighteval
# =============================================================================
# This script is the SLURM batch entry for lighteval-based evaluations.
# It submits a manager job that will further submit per-model jobs.
#
# USAGE:
#   sbatch evaluate_lighteval.sh --task <TASK_NAME> [--type <MODEL_TYPE>] [--reforce]
#
# PARAMETERS:
#   --task <TASK_NAME>   : Task to run, either 'mmlu_redux' or 'gpqa_diamond'
#   --type <MODEL_TYPE>  : Model type, either 'base' or 'sft' (default: sft)
#   --reforce           : Force rerun even if results exist
#
# EXAMPLES:
#   sbatch evaluate_lighteval.sh --task mmlu_redux --type sft
#   sbatch evaluate_lighteval.sh --task gpqa_diamond --type sft --reforce
#
# OUTPUT:
#   - Manager logs: /mnt/weka/home/haolong.jia/eval/runs/lighteval_manager_<job_id>.out/err
#   - Per-model logs: /mnt/sharefs/users/haolong.jia/result/<task>_sft/<model>/slurm.out/err
#   - Results: /mnt/sharefs/users/haolong.jia/result/<task>_sft/summary.json
# =============================================================================

#SBATCH --job-name=lighteval_manager
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/lighteval_manager_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/lighteval_manager_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --time=1:00:00

# Change to pipeline directory
cd /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline || { echo "Failed to change directory"; exit 1; }

# Activate conda environment
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

echo "🚀 Running lighteval evaluation manager for task: $@"
echo "Current directory: $(pwd)"
echo "Python: $(which python)"

# Run the manager script with submit_jobs flag
python -u evaluate_lighteval.py --submit_jobs "$@"

echo "✅ Lighteval evaluation manager completed"