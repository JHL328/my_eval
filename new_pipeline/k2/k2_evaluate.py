import os
import sys
import json
import time
import re
import subprocess
import argparse
from pathlib import Path

# 确保可以从父目录导入 k2_model
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))
from k2 import k2_model

# --- settings ---
OUTPUT_BASE_DIR = "/mnt/sharefs/users/haolong.jia/result-k2"
# --- Task-specific configurations ---
TASK_CONFIGS = {
    "bbh": {
        "handler": "default_handler", # Use the default script-generation handler
        "cot_prompts_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/bbh_cot_prompts.json",
        "eval_script_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/k2/bbh.py",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, 'bbh'),
        "time_limit": "2:00:00"
    },
    "mmlu": {
        "handler": "default_handler",
        "prompts_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_prompts.json",
        "eval_script_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/k2/mmlu.py",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, 'mmlu'),
        "time_limit": "3:00:00" # MMLU has more sub-tasks, might need more time
    },
    "mmlu_flan": {
        "handler": "default_handler",
        "cot_prompts_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_cot_prompts.json",
        "eval_script_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/k2/mmlu_flan.py",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, 'mmlu_flan_cot_fewshot_pass16'),
        "time_limit": "4:00:00" # MMLU CoT is more time-consuming
    },
    "mmlu_pro": {
        "handler": "default_handler",
        "prompts_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_pro_prompts.json",
        "eval_script_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/k2/mmlu_pro.py",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, 'mmlu_pro_pass16'),
        "time_limit": "4:00:00"
    },
    "gsm8k": {
        "handler": "default_handler",
        "data_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/gsm8k_test.jsonl",
        "eval_script_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/k2/k2_math.py",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, 'gsm8k_pass16'),
        "time_limit": "12:00:00",
        "n_sampling": 16
    },
    "math500": {
        "handler": "default_handler",
        "data_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/test_data/math_test.json",
        "eval_script_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/k2/k2_math.py",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, 'math500_pass16'),
        "time_limit": "12:00:00",
        "n_sampling": 16,
        "conda_activate_path": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval"
    }
}

# Slurm job settings for the default handler
SLURM_GPUS = 8
SLURM_CPUS = 96
SLURM_PARTITION = "main"

# VLLM inference settings for the default handler
TP_SIZE = 8
N_SAMPLING = 1
# --- settings end ---

def create_job_script_for_default_handler(script_path, job_name, log_path, command_args, time_limit, activate_env_command_in_script, nodes=1):
    """Generate a specific Slurm job script for tasks using the default handler."""
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}/{job_name}.out
#SBATCH --error={log_path}/{job_name}.err
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{SLURM_GPUS}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --time={time_limit}
#SBATCH --partition={SLURM_PARTITION}

# --- environment settings ---
cd /mnt/weka/home/haolong.jia/eval
{activate_env_command_in_script}

echo "Job started on $(hostname) at $(date)"
echo "Executing command: {command_args}"
echo "---------------------------------"

# --- execute the command ---
{command_args}

echo "---------------------------------"
echo "Job finished at $(date)"
"""
    with open(script_path, 'w') as f:
        f.write(script_content)

def wait_for_jobs_to_complete(job_ids):
    """Wait for all specified Slurm jobs to complete"""
    if not job_ids:
        return
    
    print(f"\n⏳ Waiting for {len(job_ids)} Slurm jobs to complete...")
    while job_ids:
        job_ids_str = ",".join(job_ids)
        try:
            squeue_output = subprocess.check_output(
                f"squeue -h -j {job_ids_str} -o '%i %t'",
                shell=True, text=True
            )
            running_jobs = set()
            for line in squeue_output.strip().split('\n'):
                if line:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1] in ['R', 'PD', 'CG']:
                        running_jobs.add(parts[0])
            
            job_ids = [job_id for job_id in job_ids if job_id in running_jobs]
            
            if job_ids:
                print(f"  ⏰ - {len(job_ids)} jobs still running/pending. Checking again in 120 seconds...")
                time.sleep(120)
            else:
                print("   🎉- All jobs have completed.")
                break
        except subprocess.CalledProcessError:
            print("   - `squeue` command failed. Assuming all jobs have completed.")
            break

def aggregate_final_results(models, task_name, task_output_dir):
    """After all model jobs are completed, aggregate the final results for the specified task."""
    print(f"\n📝 Aggregating final results for task '{task_name}'...")
    overall_summary = {}

    for _, model_name in models.items():
        model_result_path = os.path.join(task_output_dir, model_name, "result.json")
        if os.path.exists(model_result_path):
            try:
                with open(model_result_path, 'r') as f:
                    data = json.load(f)
                    overall_summary[model_name] = data
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read or parse result.json for {model_name}: {e}")
        else:
            print(f"Warning: Final result.json not found for {model_name}. Skipping.")
    
    final_result_path = os.path.join(task_output_dir, "result.json")
    with open(final_result_path, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"\n✅ Final overall summary for '{task_name}' saved to {final_result_path}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="K2+ Model Evaluation Dispatcher")
    parser.add_argument("--eval_task", type=str, required=True, choices=list(TASK_CONFIGS.keys()),
                        help="The evaluation task to run (e.g., 'bbh', 'mmlu', 'livecodebench').")
    return parser.parse_args()

def main():
    """Main scheduler function, submits a job for each model, then monitors and aggregates"""
    args = parse_args()
    task = args.eval_task
    config = TASK_CONFIGS[task]

    print(f"--- K2+ Evaluation Submitter & Monitor for task: {task.upper()} ---")
    
    task_output_dir = config["output_dir"]
    os.makedirs(task_output_dir, exist_ok=True)
    
    models_to_run = k2_model.model_map
    print(f"Found {len(models_to_run)} models to evaluate.")

    submitted_job_ids = []

    for model_path, model_name in models_to_run.items():
        model_final_result = os.path.join(task_output_dir, model_name, "result.json")
        if os.path.exists(model_final_result):
            print(f"⏩ Skipping model '{model_name}': Final result.json already exists.")
            continue

        job_name = f"{model_name}-{task}"
        
        print(f"🚀 Submitting '{model_name}' for task '{task}' using default script handler...")

        scripts_dir = os.path.join(task_output_dir, "job_scripts")
        logs_dir = os.path.join(task_output_dir, "logs")
        os.makedirs(scripts_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        script_path = os.path.join(scripts_dir, f"{job_name}.sh")
        
        command_parts = [
            f"python -u {config['eval_script_path']}",
            f"--model_path {model_path}",
            f"--model_name {model_name}",
            f"--output_base_dir {task_output_dir}",
            f"--tp_size {config.get('tp_size', TP_SIZE)}"
        ]
        
        # If the script is k2_math.py, it requires a task_name argument
        if "k2_math.py" in config['eval_script_path']:
            command_parts.append(f"--task_name {task}")
        
        # This generic loop adds any additional parameters from the config
        # that are not part of the reserved set.
        reserved_keys = {'handler', 'eval_script_path', 'output_dir', 'time_limit', 'conda_activate_path', 'slurm_nodes'}
        for key, value in config.items():
            if key not in reserved_keys and key not in ['tp_size']:
                command_parts.append(f"--{key} {value}")

        # Default n_sampling if not provided.
        if 'n_sampling' not in config:
             command_parts.append(f"--n_sampling {N_SAMPLING}")
            
        command_args = " ".join(command_parts)
        
        default_conda_env = "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval"
        activate_env_command_in_script = config.get("conda_activate_path", default_conda_env)
        
        slurm_nodes = config.get("slurm_nodes", 1)
        create_job_script_for_default_handler(script_path, job_name, logs_dir, command_args, config['time_limit'], activate_env_command_in_script, nodes=slurm_nodes)

        # Submit the appropriate generated script
        print(f"   - Submitting script: {script_path}")
        try:
            result = subprocess.check_output(f"sbatch {script_path}", shell=True, text=True)
            match = re.search(r'Submitted batch job (\d+)', result)
            if match:
                job_id = match.group(1)
                submitted_job_ids.append(job_id)
                print(f"   - Submitted (Job ID: {job_id})")
            else:
                print(f"   - Submitted but could not parse Job ID.")
        except subprocess.CalledProcessError as e:
            print(f"   - Failed to submit: {e}")

        time.sleep(0.2)

    wait_for_jobs_to_complete(submitted_job_ids)
    
    print(f"\nStarting final aggregation for all models that were run for task '{task}'...")
    aggregate_final_results(models_to_run, task, config["output_dir"])

    print(f"\n--- ✅ All processes for task '{task}' completed! ---")

if __name__ == "__main__":
    main()
