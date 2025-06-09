import os
import subprocess
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Model_map

OUTPUT_DIR = "/mnt/sharefs/users/haolong.jia/result/gsm8k_pass16"
EVAL_SCRIPT = "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/math_eval.py"
CONDA_ACTIVATE = "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval"
DATA_NAME = "gsm8k"
GPUS_PER_TASK = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

for model_path, model_name in Model_map.items():
    model_out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    job_name = f"gsm8k_{model_name}"
    job_script = os.path.join(model_out_dir, f"{job_name}.sh")
    with open(job_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={model_out_dir}/slurm.out
#SBATCH --error={model_out_dir}/slurm.err
#SBATCH --gres=gpu:{GPUS_PER_TASK}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

cd /mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation
{CONDA_ACTIVATE}
which python
export TOKENIZERS_PARALLELISM=false
python3 -u {EVAL_SCRIPT} \
    --model_name_or_path {model_path} \
    --data_names {DATA_NAME} \
    --output_dir {model_out_dir} \
    --split test \
    --prompt_type cot \
    --num_test_sample -1 \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 16 \
    --top_p 0.95 \
    --max_tokens_per_call 4096 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --num_shots 8
""")
    subprocess.run(["sbatch", job_script])
