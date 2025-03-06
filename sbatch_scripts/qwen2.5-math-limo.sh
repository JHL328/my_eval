#!/bin/bash
#SBATCH --job-name=eval_qwen2.5_math
#SBATCH --partition=higherprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --exclude=mbz-h100-023

source /mbz/users/yuqi.wang/miniconda3/bin/activate qwen-eval-new
cd ./eval
# it can support multi-gpu
if [[ -n $8 ]]; then
    srun bash -c "TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
    python eval.py \
    --model_name_or_path $1 \
    --data_name $2 \
    --prompt_type $3 \
    --temperature $4 \
    --start_idx 0 \
    --end_idx -1 \
    --n_sampling $5 \
    --k 1 \
    --split "test" \
    --max_tokens $6 \
    --seed 0 \
    --top_p 1 \
    --surround_with_messages \
    --output_dir $7 \
    --completions_save_dir $7 \
    --use_few_shot"
else
    srun bash -c "TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
    python eval.py \
    --model_name_or_path $1 \
    --data_name $2 \
    --prompt_type $3 \
    --temperature $4 \
    --start_idx 0 \
    --end_idx -1 \
    --n_sampling $5 \
    --k 1 \
    --split "test" \
    --max_tokens $6 \
    --seed 0 \
    --top_p 1 \
    --surround_with_messages \
    --output_dir $7 \
    --completions_save_dir $7"
fi