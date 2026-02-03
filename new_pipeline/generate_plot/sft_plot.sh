#!/bin/bash

# =================================================================
# SFT in Pretrain Plotting Script
# =================================================================
# This script generates plots for SFT in Pretrain evaluation.
# It uses --mode all to combine base and sft model results.
#
# USAGE:
#   bash sft_plot.sh
# =================================================================

# Output directory for plots
PLOT_DIR="/mnt/weka/home/haolong.jia/eval/data-engineering/evals/sft_in_pretrain"
mkdir -p "$PLOT_DIR/base" "$PLOT_DIR/math" "$PLOT_DIR/code"

# Path to the Python plotting script
PLOT_SCRIPT="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/generate_plot/plot.py"

# Base directory for results
RESULT_DIR="/mnt/weka/shrd/k2m/haolong.jia/result"

# --- List of Commands to Execute ---
commands=(
    # ========================================
    # Benchmarks with BOTH base and sft eval
    # Using --mode all to combine results
    # ========================================
    
    # --- ifeval (base + sft) ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/ifeval/result.json --passk_sft ${RESULT_DIR}/sft/ifeval/result.json --output ${PLOT_DIR}/base/ifeval.pdf --metric inst_level_strict_acc,none --task ifeval --mode all"
    
    # --- BBH (base + sft) ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/bbh_pass16/passk.json --passk_sft ${RESULT_DIR}/bbh_pass16_sft/passk.json --output ${PLOT_DIR}/base/bbh_pass@1.pdf --metric pass@1 --task bbh --mode all"
    
    # --- MMLU (base + sft) ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mmlu/passk.json --passk_sft ${RESULT_DIR}/mmlu_sft/passk.json --output ${PLOT_DIR}/base/mmlu_std_pass@1.pdf --metric pass@1 --task mmlu_std --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mmlu_flan_pass16/passk.json --passk_sft ${RESULT_DIR}/mmlu_flan_pass16_sft/passk.json --output ${PLOT_DIR}/base/mmlu_cot_pass@1.pdf --metric pass@1 --task mmlu_cot --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mmlu_pro_pass16/passk.json --passk_sft ${RESULT_DIR}/mmlu_pro_pass16_sft/passk.json --output ${PLOT_DIR}/base/mmlu_pro_pass@1.pdf --metric pass@1 --task mmlu_pro --mode all"

    # --- MATH (base + sft) ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --output ${PLOT_DIR}/math/math500_pass@1.pdf --metric pass@1 --task math500 --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --output ${PLOT_DIR}/math/math500_pass@8.pdf --metric pass@8 --task math500 --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --output ${PLOT_DIR}/math/math500_pass@16.pdf --metric pass@16 --task math500 --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --output ${PLOT_DIR}/math/math500_pass@32.pdf --metric pass@32 --task math500 --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --output ${PLOT_DIR}/math/math500_pass@64.pdf --metric pass@64 --task math500 --mode all"
    
    # --- GSM8K (base only, sft uses same dir) ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --output ${PLOT_DIR}/math/gsm8k_pass@1.pdf --metric pass@1 --task gsm8k --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --output ${PLOT_DIR}/math/gsm8k_pass@8.pdf --metric pass@8 --task gsm8k --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --output ${PLOT_DIR}/math/gsm8k_pass@16.pdf --metric pass@16 --task gsm8k --mode all"

    # --- Code (base + sft) ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --passk_sft ${RESULT_DIR}/humaneval_sft/base_passk.json --output ${PLOT_DIR}/code/humaneval_base_pass1.pdf --metric pass@1 --task humaneval --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --passk_sft ${RESULT_DIR}/humaneval_sft/base_passk.json --output ${PLOT_DIR}/code/humaneval_base_pass16.pdf --metric pass@16 --task humaneval --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --passk_sft ${RESULT_DIR}/humaneval_sft/plus_passk.json --output ${PLOT_DIR}/code/humaneval_plus_pass1.pdf --metric pass@1 --task humaneval+ --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --passk_sft ${RESULT_DIR}/humaneval_sft/plus_passk.json --output ${PLOT_DIR}/code/humaneval_plus_pass16.pdf --metric pass@16 --task humaneval+ --mode all"
    
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --passk_sft ${RESULT_DIR}/mbpp_sft/base_passk.json --output ${PLOT_DIR}/code/mbpp_base_pass1.pdf --metric pass@1 --task mbpp --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --passk_sft ${RESULT_DIR}/mbpp_sft/base_passk.json --output ${PLOT_DIR}/code/mbpp_base_pass16.pdf --metric pass@16 --task mbpp --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --passk_sft ${RESULT_DIR}/mbpp_sft/plus_passk.json --output ${PLOT_DIR}/code/mbpp_plus_pass1.pdf --metric pass@1 --task mbpp+ --mode all"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --passk_sft ${RESULT_DIR}/mbpp_sft/plus_passk.json --output ${PLOT_DIR}/code/mbpp_plus_pass16.pdf --metric pass@16 --task mbpp+ --mode all"

    # ========================================
    # Benchmarks with ONLY base eval
    # Using default --mode base
    # ========================================
    
    # --- Likelihood-based evaluation (Base only) ---
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/drop/result.json --output ${PLOT_DIR}/base/drop_f1.pdf --metric f1,none --task drop"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/arc_easy/result.json --output ${PLOT_DIR}/base/arc_easy_acc_norm.pdf --metric acc_norm,none --task arc_easy"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/arc_challenge/result.json --output ${PLOT_DIR}/base/arc_challenge_acc_norm.pdf --metric acc_norm,none --task arc_challenge"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/hellaswag/result.json --output ${PLOT_DIR}/base/hellaswag_acc_norm.pdf --metric acc_norm,none --task hellaswag"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/piqa/result.json --output ${PLOT_DIR}/base/piqa_acc_norm.pdf --metric acc_norm,none --task piqa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/winogrande/result.json --output ${PLOT_DIR}/base/winogrande_acc_norm.pdf --metric acc_norm,none --task winogrande"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/triviaqa/result.json --output ${PLOT_DIR}/base/triviaqa_exact_match.pdf --metric exact_match,remove_whitespace --task triviaqa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/nq_open/result.json --output ${PLOT_DIR}/base/nq_open_exact_match.pdf --metric exact_match,remove_whitespace --task nq_open"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/agieval/result.json --output ${PLOT_DIR}/base/agieval_acc_norm.pdf --metric acc_norm,none --task agieval"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/commonsense_qa/result.json --output ${PLOT_DIR}/base/commonsense_qa_acc_norm.pdf --metric acc_norm,none --task commonsense_qa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/openbookqa/result.json --output ${PLOT_DIR}/base/openbookqa_acc_norm.pdf --metric acc_norm,none --task openbookqa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/social_iqa/result.json --output ${PLOT_DIR}/base/social_iqa_acc_norm.pdf --metric acc_norm,none --task social_iqa"
    "python ${PLOT_SCRIPT} --passk ${RESULT_DIR}/truthfulqa_mc2/result.json --output ${PLOT_DIR}/base/truthfulqa_mc2_acc_norm.pdf --metric acc_norm,none --task truthfulqa_mc2"
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
