import os
from model import Model_map
from evalplus.data import get_human_eval_plus

def create_job_script(script_path, job_name, output_file, error_file, command_args, time_limit="8:00:00"):
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time={time_limit}
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

cd /mnt/weka/home/haolong.jia/eval
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export CUDA_LAUNCH_BLOCKING=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

{command_args}
"""
    with open(script_path, "w") as f:
        f.write(script_content)
        f.flush()
        os.fsync(f.fileno())

if __name__ == "__main__":
    # 1. 获取HumanEval任务总数
    problems = get_human_eval_plus()
    total_tasks = len(problems)
    num_splits = 8
    split_size = (total_tasks + num_splits - 1) // num_splits

    ROOT_DIR = "/mnt/sharefs/users/haolong.jia/result/humaneval"
    os.makedirs(ROOT_DIR, exist_ok=True)

    for model_path, model_name in Model_map.items():
        output_dir = os.path.join(ROOT_DIR, model_name)
        os.makedirs(output_dir, exist_ok=True)
        scripts_dir = os.path.join(output_dir, "job_scripts")
        os.makedirs(scripts_dir, exist_ok=True)
        for split_idx in range(num_splits):
            idx_start = split_idx * split_size
            idx_end = min((split_idx + 1) * split_size, total_tasks)
            job_name = f"humaneval_{model_name}_{idx_start}_{idx_end}"
            job_script = os.path.join(scripts_dir, f"{job_name}.sh")
            output_file = os.path.join(output_dir, f"output_{idx_start}_{idx_end}.out")
            error_file = os.path.join(output_dir, f"error_{idx_start}_{idx_end}.err")
            command_args = (
                f"python RL-eval/new_pipeline/evaluate_humaneval_single.py "
                f"--model_path '{model_path}' "
                f"--output_dir '{output_dir}' "
                f"--model_name '{model_name}' "
                f"--tensor_parallel_size 1 "
                f"--temperature 0.6 "
                f"--n_samples 32 "
                f"--max_tokens 1024 "
                f"--idx_start {idx_start} "
                f"--idx_end {idx_end} "
            )
            create_job_script(job_script, job_name, output_file, error_file, command_args)
            os.system(f"sbatch {job_script}")
            print(f"Submitted job for {model_name} split {idx_start}-{idx_end}")
