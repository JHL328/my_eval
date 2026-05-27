#!/bin/bash

# =================================================================
# Automated CSV Generation Script
# =================================================================
# This script generates CSV files for different benchmarks and metrics.
# It separates 7B and 1.5B model results into their respective directories.
#
# USAGE:
#   bash csv.sh
# =================================================================

# Path to the Python CSV generation script
CSV_SCRIPT="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/generate_plot/generate_csv.py"

# Base directory for results
RESULT_DIR="/mnt/weka/shrd/k2m/haolong.jia/result"

# Output Roots
ROOT_7B="/mnt/weka/home/haolong.jia/eval/data-engineering/evals/bbq_ablations"
ROOT_1P5B="/mnt/weka/home/haolong.jia/eval/data-engineering/evals/bbq_ablations"

# --- List of Commands to Execute ---
# Arguments: --output_root_7b <path> --output_root_1p5b <path> --subdir <subdir>
# The subdir (base, math, code) determines the subfolder within the roots.

commands=(
    # --- BBH ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/bbh_pass16/passk.json --metric pass@1 --task bbh --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    
    # --- MMLU ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mmlu/passk.json --metric pass@1 --task mmlu_std --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mmlu_flan_pass16/passk.json --metric pass@1 --task mmlu_cot --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mmlu_pro_pass16/passk.json --metric pass@1 --task mmlu_pro --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"

    # --- IFEval ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/ifeval/result.json --metric inst_level_strict_acc,none --task ifeval --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"

    # --- MATH ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --metric pass@1 --task math500 --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir math --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --metric pass@8 --task math500 --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir math --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --metric pass@16 --task math500 --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir math --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --metric pass@32 --task math500 --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir math --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --metric pass@64 --task math500 --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir math --only mix-bbq-all-mask,mix-bbq-all-baseline"
    
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --metric pass@1 --task gsm8k --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir math --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --metric pass@8 --task gsm8k --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir math --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --metric pass@16 --task gsm8k --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir math --only mix-bbq-all-mask,mix-bbq-all-baseline"

    # --- Likelihood-based evaluation (Base) ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/drop/result.json --metric f1,none --task drop --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/arc_easy/result.json --metric acc_norm,none --task arc_easy --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/arc_challenge/result.json --metric acc_norm,none --task arc_challenge --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/hellaswag/result.json --metric acc_norm,none --task hellaswag --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/piqa/result.json --metric acc_norm,none --task piqa --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/winogrande/result.json --metric acc_norm,none --task winogrande --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/triviaqa/result.json --metric exact_match,remove_whitespace --task triviaqa --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/nq_open/result.json --metric exact_match,remove_whitespace --task nq_open --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/agieval/result.json --metric acc_norm,none --task agieval --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/commonsense_qa/result.json --metric acc_norm,none --task commonsense_qa --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/openbookqa/result.json --metric acc_norm,none --task openbookqa --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/social_iqa/result.json --metric acc_norm,none --task social_iqa --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/truthfulqa_mc2/result.json --metric acc_norm,none --task truthfulqa_mc2 --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir base --only mix-bbq-all-mask,mix-bbq-all-baseline"

    # --- Code ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --metric pass@16 --task humaneval_base --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --metric pass@16 --task humaneval_plus --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --metric pass@1 --task humaneval_base --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --metric pass@1 --task humaneval_plus --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"

    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --metric pass@16 --task mbpp_base --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --metric pass@16 --task mbpp_plus --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --metric pass@1 --task mbpp_base --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --metric pass@1 --task mbpp_plus --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --metric pass@32 --task mbpp_base --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --metric pass@32 --task mbpp_plus --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --metric pass@64 --task mbpp_base --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --metric pass@64 --task mbpp_plus --output_root_7b ${ROOT_7B} --output_root_1p5b ${ROOT_1P5B} --subdir code --only mix-bbq-all-mask,mix-bbq-all-baseline"
)

# --- Execution Loop ---
for cmd in "${commands[@]}"; do
    echo "===================================================================="
    echo "🚀 Executing CSV Gen: $cmd"
    echo "===================================================================="
    eval "$cmd"
    if [ $? -eq 0 ]; then
        echo "✅ CSV Generated successfully."
    else
        echo "❌ Command failed."
    fi
    echo
done

echo "🎉 All CSV generation commands have been executed."
