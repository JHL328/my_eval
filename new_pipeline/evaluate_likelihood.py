import os
import time
import sys
import argparse

# 动态导入Model_map
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model_map

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='drop', help='Task name for evaluation')
args = parser.parse_args()
task = args.task

output_dir = f"/mnt/sharefs/users/haolong.jia/result/{task}"
job_dir = os.path.join(output_dir, "job_scripts")
log_dir = os.path.join(output_dir, "logs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(job_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 你可以根据需要调整资源参数
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task}_{model_name}
#SBATCH --output={log_dir}/{model_name}.out
#SBATCH --error={log_dir}/{model_name}.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

{lm_eval_cmd}
"""

for model_path, model_name in Model_map.items():
    model_out_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    job_script = os.path.join(job_dir, f"job_{model_name}.sh")
    if task == "social_iqa":
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,gpu_memory_utilization=0.95 \
  --tasks {task} \
  --output_path {output_dir}/{model_name} \
  --batch_size auto \
  --log_samples \
  --num_fewshot 0 \
  --trust_remote_code"""
    else:
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,gpu_memory_utilization=0.95 \
  --tasks {task} \
  --output_path {output_dir}/{model_name} \
  --batch_size auto \
  --log_samples \
  --num_fewshot 0"""
    with open(job_script, "w") as f:
        f.write(SBATCH_TEMPLATE.format(
            model_name=model_name,
            model_path=model_path,
            output_dir=output_dir,
            log_dir=log_dir,
            task=task,
            lm_eval_cmd=lm_eval_cmd
        ))
    os.system(f"sbatch {job_script}")
    print(f"Submitted job for {model_name}")
    time.sleep(0.2)  
