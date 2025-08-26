#!/bin/bash

# =============================================================================
# Universal Benchmark Submitter Script (No Arg Passthrough)
# =============================================================================
# This script is a unified entry point for submitting all benchmark batch jobs.
# It dispatches to the correct batch script (evaluate_gsm8k.sh, evaluate_likelihood.sh, evaluate.sh)
# based on the --task argument, and passes only hardcoded arguments for each task.
#
# USAGE:
#   bash submitter.sh --task <TASK_NAME>
#
# EXAMPLES:
#   bash submitter.sh --task gsm8k
#   bash submitter.sh --task math500
#   bash submitter.sh --task drop
#   bash submitter.sh --task arc_challenge
#   bash submitter.sh --task mmlu_flan_cot_fewshot_pass16
#
# All sbatch commands and arguments are hardcoded per task below.
# =============================================================================

# Parse --task and --type arguments
TASK_NAME=""
MODEL_TYPE="base"  # Default to base
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                TASK_NAME="$2"
                shift 2
            else
                shift
            fi
            ;;
        --task=*)
            TASK_NAME="${1#*=}"
            shift
            ;;
        --type)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                MODEL_TYPE="$2"
                shift 2
            else
                shift
            fi
            ;;
        --type=*)
            MODEL_TYPE="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -z "$TASK_NAME" ]]; then
    echo "[submitter.sh] ERROR: --task argument is required."
    echo "Usage: bash submitter.sh --task <TASK_NAME> [--type <base|sft>]"
    exit 1
fi

# Validate --type argument
if [[ "$MODEL_TYPE" != "base" && "$MODEL_TYPE" != "sft" ]]; then
    echo "[submitter.sh] ERROR: --type must be 'base' or 'sft' (got: $MODEL_TYPE)"
    exit 1
fi

# One-task-one-branch, all sbatch commands and arguments are hardcoded below

##############################
###### Code Tasks ###########
##############################
if [[ "$TASK_NAME" == "mbpp" ]]; then
    # Two-stage submission: GPU for generation, then CPU for sanitize/evaluate
    echo "[submitter.sh] Submitting two-stage pipeline for mbpp"
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/generate_code_gpu.sh --task mbpp"
elif [[ "$TASK_NAME" == "humaneval" ]]; then
    # Two-stage submission: GPU for generation, then CPU for sanitize/evaluate
    echo "[submitter.sh] Submitting two-stage pipeline for humaneval"
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/generate_code_gpu.sh --task humaneval"
elif [[ "$TASK_NAME" == "mbpp_old" ]]; then
    # Old single-job submission for backward compatibility
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_code.sh --task mbpp"
elif [[ "$TASK_NAME" == "humaneval_old" ]]; then
    # Old single-job submission for backward compatibility
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_code.sh --task humaneval"
    
##############################
###### MATH Tasks ###########
##############################
elif [[ "$TASK_NAME" == "gsm8k" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_gsm8k.sh --task gsm8k"
elif [[ "$TASK_NAME" == "math500" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_gsm8k.sh --task math500"

##############################
###### Likelihood eval #######
##############################
elif [[ "$TASK_NAME" == "drop" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task drop"
elif [[ "$TASK_NAME" == "arc_easy" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task arc_easy"
elif [[ "$TASK_NAME" == "arc_challenge" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task arc_challenge"
elif [[ "$TASK_NAME" == "hellaswag" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task hellaswag"
elif [[ "$TASK_NAME" == "piqa" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task piqa"
elif [[ "$TASK_NAME" == "winogrande" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task winogrande"
elif [[ "$TASK_NAME" == "triviaqa" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task triviaqa"
elif [[ "$TASK_NAME" == "nq_open" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task nq_open"
elif [[ "$TASK_NAME" == "commonsense_qa" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task commonsense_qa"
elif [[ "$TASK_NAME" == "agieval" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task agieval"
elif [[ "$TASK_NAME" == "openbookqa" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task openbookqa"
elif [[ "$TASK_NAME" == "social_iqa" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task social_iqa"
elif [[ "$TASK_NAME" == "truthfulqa_mc2" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_likelihood.sh --task truthfulqa_mc2"
##############################
#### MMLU and BBH Tasks ######
##############################
elif [[ "$TASK_NAME" == "mmlu_flan_cot_fewshot_pass16" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate.sh --task mmlu_flan_cot_fewshot_pass16 --type $MODEL_TYPE --force"
elif [[ "$TASK_NAME" == "mmlu_pro_pass16" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate.sh --task mmlu_pro_pass16 --type $MODEL_TYPE --force"
elif [[ "$TASK_NAME" == "bbh_pass16" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate.sh --task bbh_pass16 --type $MODEL_TYPE --force"
elif [[ "$TASK_NAME" == "mmlu" ]]; then
    CMD="sbatch /mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate.sh --task mmlu --type $MODEL_TYPE --force"
##############################
###### GQA Tasks #############
##############################
else
    echo "[submitter.sh] ERROR: Unknown or unsupported task: $TASK_NAME"
    echo "Supported tasks: mbpp, humaneval, mbpp_old, humaneval_old, gsm8k, math500, drop, arc_easy, arc_challenge, hellaswag, piqa, winogrande, triviaqa, nq_open, commonsense_qa, agieval, openbookqa, social_iqa, truthfulqa_mc2, mmlu_flan_cot_fewshot_pass16, mmlu_pro_pass16, bbh_pass16, mmlu, gpqa"
    echo "Note: Use mbpp/humaneval for two-stage GPU+CPU pipeline, or mbpp_old/humaneval_old for single-job pipeline"
    exit 2
fi
echo "[submitter.sh] Will run: $CMD"
eval "$CMD"
