#!/bin/bash

# =================================================================
# Automated Plotting Script
# =================================================================
# This script executes a series of plotting commands for different
# benchmarks and metrics.
#
# USAGE:
#   bash plot.sh
#
# To add a new plot, simply add a new command to the `commands`
# array below.
# =================================================================

# Ensure the output directory for plots exists
PLOT_DIR="/mnt/weka/home/haolong.jia/eval/RL-eval/plot"
mkdir -p "$PLOT_DIR"

# Path to the Python plotting scripts
PLOT_SCRIPT="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/generate_plot/plot.py"
SMOL_SCRIPT="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/generate_plot/smol.py"

# Base directory for results
RESULT_DIR="/mnt/sharefs/users/haolong.jia/result"

# --- List of Commands to Execute ---
# Add your plotting commands here.
commands=(
    # # --- BBH Plots (pass@k) ---
    # "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/bbh_pass16/passk.json --output ${PLOT_DIR}/bbh_pass@1.pdf --metric pass@1 --task bbh"
    # "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/bbh_pass16/passk.json --output ${PLOT_DIR}/bbh_pass@2.pdf --metric pass@2 --task bbh"
    

    # # --- MMLU (Standard) Plots (pass@k) ---
    # "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mmlu/passk.json --output ${PLOT_DIR}/mmlu_std_pass@1.pdf --metric pass@1 --task mmlu_std"

    # # --- MMLU (CoT) Plots (pass@k) ---
    # "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mmlu_flan_pass16/passk.json --output ${PLOT_DIR}/mmlu_cot_pass@1.pdf --metric pass@1 --task mmlu_cot"
    
    # # --- MMLU-Pro Plots (pass@k) ---
    # "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mmlu_pro_pass16/passk.json --output ${PLOT_DIR}/mmlu_pro_pass@1.pdf --metric pass@1 --task mmlu_pro"

    # # --- MATH & Code Plots (pass@k) ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passkback.json --output ${PLOT_DIR}/math500_pass@1.pdf --metric pass@1 --task math500"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passkback.json --output ${PLOT_DIR}/math500_pass@32.pdf --metric pass@32 --task math500"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passkback.json --output ${PLOT_DIR}/math500_pass@64.pdf --metric pass@64 --task math500"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/math-verify-passk.json --output ${PLOT_DIR}/gsm8k_pass@1.pdf --metric pass@1 --task gsm8k"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/math-verify-passk.json --output ${PLOT_DIR}/gsm8k_pass@8.pdf --metric pass@8 --task gsm8k"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/math-verify-passk.json --output ${PLOT_DIR}/gsm8k_pass@16.pdf --metric pass@16 --task gsm8k"

    # # --- Likelihood-based evaluation plots ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/drop/result.json --output ${PLOT_DIR}/drop_f1.pdf --metric f1,none --task drop"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/arc_easy/result.json --output ${PLOT_DIR}/arc_easy_acc_norm.pdf --metric acc_norm,none --task arc_easy"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/arc_challenge/result.json --output ${PLOT_DIR}/arc_challenge_acc_norm.pdf --metric acc_norm,none --task arc_challenge"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/hellaswag/result.json --output ${PLOT_DIR}/hellaswag_acc_norm.pdf --metric acc_norm,none --task hellaswag"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/piqa/result.json --output ${PLOT_DIR}/piqa_acc_norm.pdf --metric acc_norm,none --task piqa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/winogrande/result.json --output ${PLOT_DIR}/winogrande_acc_norm.pdf --metric acc_norm,none --task winogrande"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/triviaqa/result.json --output ${PLOT_DIR}/triviaqa_exact_match.pdf --metric exact_match,remove_whitespace --task triviaqa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/nq_open/result.json --output ${PLOT_DIR}/nq_open_exact_match.pdf --metric exact_match,remove_whitespace --task nq_open"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/agieval/result.json --output ${PLOT_DIR}/agieval_acc_norm.pdf --metric acc_norm,none --task agieval"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/commonsense_qa/result.json --output ${PLOT_DIR}/commonsense_qa_acc_norm.pdf --metric acc_norm,none --task commonsense_qa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/openbookqa/result.json --output ${PLOT_DIR}/openbookqa_acc_norm.pdf --metric acc_norm,none --task openbookqa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/social_iqa/result.json --output ${PLOT_DIR}/social_iqa_acc_norm.pdf --metric acc_norm,none --task social_iqa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/truthfulqa_mc2/result.json --output ${PLOT_DIR}/truthfulqa_mc2_acc_norm.pdf --metric acc_norm,none --task truthfulqa_mc2"

    # --- humaneval and mbpp based plots ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --output ${PLOT_DIR}/humaneval_base_pass16.pdf --metric pass@16 --task humaneval"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --output ${PLOT_DIR}/humaneval_plus_pass16.pdf --metric pass@16 --task humaneval+"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --output ${PLOT_DIR}/humaneval_base_pass1.pdf --metric pass@1 --task humaneval"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --output ${PLOT_DIR}/humaneval_plus_pass1.pdf --metric pass@1 --task humaneval+"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --output ${PLOT_DIR}/mbpp_base_pass16.pdf --metric pass@16 --task mbpp"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --output ${PLOT_DIR}/mbpp_plus_pass16.pdf --metric pass@16 --task mbpp+"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --output ${PLOT_DIR}/mbpp_base_pass64.pdf --metric pass@64 --task mbpp"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --output ${PLOT_DIR}/mbpp_plus_pass64.pdf --metric pass@64 --task mbpp+"

    # --- Smol (Token-based) Plots ---
    # These plots use the smol.py script which converts steps to token counts for better comparison
    # BBH Token-based plots
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/bbh_pass16/passk.json --output ${PLOT_DIR}/bbh_pass@1_tokens.pdf --metric pass@1 --task bbh"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/bbh_pass16/passk.json --output ${PLOT_DIR}/bbh_pass@2_tokens.pdf --metric pass@2 --task bbh"
    
    # # MMLU Token-based plots
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/mmlu/passk.json --output ${PLOT_DIR}/mmlu_std_pass@1_tokens.pdf --metric pass@1 --task mmlu_std"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/mmlu_flan_pass16/passk.json --output ${PLOT_DIR}/mmlu_cot_pass@1_tokens.pdf --metric pass@1 --task mmlu_cot"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/mmlu_pro_pass16/passk.json --output ${PLOT_DIR}/mmlu_pro_pass@1_tokens.pdf --metric pass@1 --task mmlu_pro"
    
    # # MATH & GSM8K Token-based plots
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passkback.json --output ${PLOT_DIR}/math500_pass@1_tokens.pdf --metric pass@1 --task math500"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passkback.json --output ${PLOT_DIR}/math500_pass@32_tokens.pdf --metric pass@32 --task math500"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passkback.json --output ${PLOT_DIR}/math500_pass@64_tokens.pdf --metric pass@64 --task math500"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --output ${PLOT_DIR}/gsm8k_pass@1_tokens.pdf --metric pass@1 --task gsm8k"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --output ${PLOT_DIR}/gsm8k_pass@8_tokens.pdf --metric pass@8 --task gsm8k"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --output ${PLOT_DIR}/gsm8k_pass@16_tokens.pdf --metric pass@16 --task gsm8k"
    
    # # Likelihood-based Token plots
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/drop/result.json --output ${PLOT_DIR}/drop_f1_tokens.pdf --metric f1,none --task drop"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/arc_easy/result.json --output ${PLOT_DIR}/arc_easy_acc_norm_tokens.pdf --metric acc_norm,none --task arc_easy"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/arc_challenge/result.json --output ${PLOT_DIR}/arc_challenge_acc_norm_tokens.pdf --metric acc_norm,none --task arc_challenge"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/hellaswag/result.json --output ${PLOT_DIR}/hellaswag_acc_norm_tokens.pdf --metric acc_norm,none --task hellaswag"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/piqa/result.json --output ${PLOT_DIR}/piqa_acc_norm_tokens.pdf --metric acc_norm,none --task piqa"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/winogrande/result.json --output ${PLOT_DIR}/winogrande_acc_norm_tokens.pdf --metric acc_norm,none --task winogrande"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/triviaqa/result.json --output ${PLOT_DIR}/triviaqa_exact_match_tokens.pdf --metric exact_match,remove_whitespace --task triviaqa"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/nq_open/result.json --output ${PLOT_DIR}/nq_open_exact_match_tokens.pdf --metric exact_match,remove_whitespace --task nq_open"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/agieval/result.json --output ${PLOT_DIR}/agieval_acc_norm_tokens.pdf --metric acc_norm,none --task agieval"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/commonsense_qa/result.json --output ${PLOT_DIR}/commonsense_qa_acc_norm_tokens.pdf --metric acc_norm,none --task commonsense_qa"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/openbookqa/result.json --output ${PLOT_DIR}/openbookqa_acc_norm_tokens.pdf --metric acc_norm,none --task openbookqa"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/social_iqa/result.json --output ${PLOT_DIR}/social_iqa_acc_norm_tokens.pdf --metric acc_norm,none --task social_iqa"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/truthfulqa_mc2/result.json --output ${PLOT_DIR}/truthfulqa_mc2_acc_norm_tokens.pdf --metric acc_norm,none --task truthfulqa_mc2"

    # --- humaneval based plots ---
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --output ${PLOT_DIR}/humaneval_base_pass16_tokens.pdf --metric pass@16 --task humaneval"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --output ${PLOT_DIR}/humaneval_plus_pass16_tokens.pdf --metric pass@16 --task humaneval"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --output ${PLOT_DIR}/humaneval_base_pass1_tokens.pdf --metric pass@1 --task humaneval"
    # "python ${SMOL_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --output ${PLOT_DIR}/humaneval_plus_pass1_tokens.pdf --metric pass@1 --task humaneval"
)

# --- Execution Loop ---
for cmd in "${commands[@]}"; do
    echo "===================================================================="
    echo "🚀 Executing: $cmd"
    echo "===================================================================="
    eval "$cmd"
    if [ $? -eq 0 ]; then
        echo "✅ Command completed successfully."
    else
        echo "❌ Command failed."
    fi
    echo
done

echo "🎉 All plotting commands have been executed."
