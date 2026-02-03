#!/bin/bash

# =================================================================
# SFT in Pretrain CSV Generation Script
# =================================================================
# This script generates CSV files for SFT in Pretrain evaluation.
# It uses --mode all to combine base and sft model results.
#
# USAGE:
#   bash sft_csv.sh
# =================================================================

# Path to the Python CSV generation script
CSV_SCRIPT="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/generate_plot/generate_csv.py"

# Base directory for results
RESULT_DIR="/mnt/weka/shrd/k2m/haolong.jia/result"

# Output Root for SFT in Pretrain
ROOT="/mnt/weka/home/haolong.jia/eval/data-engineering/evals/sft_in_pretrain"
mkdir -p "$ROOT"

# --- List of Commands to Execute ---
commands=(
    # ========================================
    # Benchmarks with BOTH base and sft eval
    # Using --mode all to combine results
    # ========================================
    
    # --- ifeval (base + sft) ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/ifeval/result.json --passk_sft ${RESULT_DIR}/sft/ifeval/result.json --metric inst_level_strict_acc,none --task ifeval --output_root_1p5b ${ROOT} --subdir base --mode all"
    
    # --- BBH (base + sft) ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/bbh_pass16/passk.json --passk_sft ${RESULT_DIR}/bbh_pass16_sft/passk.json --metric pass@1 --task bbh --output_root_1p5b ${ROOT} --subdir base --mode all"
    
    # --- MMLU (base + sft) ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mmlu/passk.json --passk_sft ${RESULT_DIR}/mmlu_sft/passk.json --metric pass@1 --task mmlu_std --output_root_1p5b ${ROOT} --subdir base --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mmlu_flan_pass16/passk.json --passk_sft ${RESULT_DIR}/mmlu_flan_pass16_sft/passk.json --metric pass@1 --task mmlu_cot --output_root_1p5b ${ROOT} --subdir base --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mmlu_pro_pass16/passk.json --passk_sft ${RESULT_DIR}/mmlu_pro_pass16_sft/passk.json --metric pass@1 --task mmlu_pro --output_root_1p5b ${ROOT} --subdir base --mode all"

    # --- MATH (base + sft) ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --metric pass@1 --task math500 --output_root_1p5b ${ROOT} --subdir math --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --metric pass@8 --task math500 --output_root_1p5b ${ROOT} --subdir math --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --metric pass@16 --task math500 --output_root_1p5b ${ROOT} --subdir math --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --metric pass@32 --task math500 --output_root_1p5b ${ROOT} --subdir math --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/math500_pass64/passk.json --passk_sft ${RESULT_DIR}/math500_pass64_sft/passk.json --metric pass@64 --task math500 --output_root_1p5b ${ROOT} --subdir math --mode all"
    
    # --- GSM8K (base only, sft uses same dir) ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --metric pass@1 --task gsm8k --output_root_1p5b ${ROOT} --subdir math --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --metric pass@8 --task gsm8k --output_root_1p5b ${ROOT} --subdir math --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/gsm8k_pass16/passk.json --metric pass@16 --task gsm8k --output_root_1p5b ${ROOT} --subdir math --mode all"

    # --- Code (base + sft) ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --passk_sft ${RESULT_DIR}/humaneval_sft/base_passk.json --metric pass@1 --task humaneval_base --output_root_1p5b ${ROOT} --subdir code --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/humaneval/base_passk.json --passk_sft ${RESULT_DIR}/humaneval_sft/base_passk.json --metric pass@16 --task humaneval_base --output_root_1p5b ${ROOT} --subdir code --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --passk_sft ${RESULT_DIR}/humaneval_sft/plus_passk.json --metric pass@1 --task humaneval_plus --output_root_1p5b ${ROOT} --subdir code --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/humaneval/plus_passk.json --passk_sft ${RESULT_DIR}/humaneval_sft/plus_passk.json --metric pass@16 --task humaneval_plus --output_root_1p5b ${ROOT} --subdir code --mode all"
    
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --passk_sft ${RESULT_DIR}/mbpp_sft/base_passk.json --metric pass@1 --task mbpp_base --output_root_1p5b ${ROOT} --subdir code --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/base_passk.json --passk_sft ${RESULT_DIR}/mbpp_sft/base_passk.json --metric pass@16 --task mbpp_base --output_root_1p5b ${ROOT} --subdir code --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --passk_sft ${RESULT_DIR}/mbpp_sft/plus_passk.json --metric pass@1 --task mbpp_plus --output_root_1p5b ${ROOT} --subdir code --mode all"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/mbpp/plus_passk.json --passk_sft ${RESULT_DIR}/mbpp_sft/plus_passk.json --metric pass@16 --task mbpp_plus --output_root_1p5b ${ROOT} --subdir code --mode all"

    # ========================================
    # Benchmarks with ONLY base eval
    # Using default --mode base
    # ========================================
    
    # --- Likelihood-based evaluation (Base only) ---
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/drop/result.json --metric f1,none --task drop --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/arc_easy/result.json --metric acc_norm,none --task arc_easy --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/arc_challenge/result.json --metric acc_norm,none --task arc_challenge --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/hellaswag/result.json --metric acc_norm,none --task hellaswag --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/piqa/result.json --metric acc_norm,none --task piqa --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/winogrande/result.json --metric acc_norm,none --task winogrande --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/triviaqa/result.json --metric exact_match,remove_whitespace --task triviaqa --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/nq_open/result.json --metric exact_match,remove_whitespace --task nq_open --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/agieval/result.json --metric acc_norm,none --task agieval --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/commonsense_qa/result.json --metric acc_norm,none --task commonsense_qa --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/openbookqa/result.json --metric acc_norm,none --task openbookqa --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/social_iqa/result.json --metric acc_norm,none --task social_iqa --output_root_1p5b ${ROOT} --subdir base"
    "python ${CSV_SCRIPT} --passk ${RESULT_DIR}/truthfulqa_mc2/result.json --metric acc_norm,none --task truthfulqa_mc2 --output_root_1p5b ${ROOT} --subdir base"
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
