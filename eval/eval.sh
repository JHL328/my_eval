CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
python eval.py \
--model_name_or_path "/mbz/users/richard.fan/LLaMA-Factory/saves/qwen2.5-32b-instruct-16-gpu-5epochs/full/sft/checkpoint-315" \
--data_name "aime24" \
--prompt_type "qwen-instruct" \
--temperature 0.0 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--top_p 1 \
--surround_with_messages \
--output_dir /mbz/users/yuqi.wang/RL-eval/eval_results \
--completions_save_dir /mbz/users/yuqi.wang/RL-eval/eval_results