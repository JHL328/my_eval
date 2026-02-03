#!/usr/bin/env python
"""
MATH500 Evaluation Script for 8-GPU Node Models
- Avg@4 (temperature=0.6, top_p=0.95)
- SFT-style eval with apply_chat_template (qwen25 prompt)
- num_shots=0
"""

import os
import sys
import json
import csv
import time
import subprocess
import re
import numpy as np
import pandas as pd

# Import model_map from local model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import model_map

# =====================
# Task Configurations
# =====================
TASK_CONFIGS = {
    "math500": {
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/node/math500_avg4",
        "EVAL_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/math_eval.py",
        "DATA_DIR": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/data",
        "DATA_NAME": "math500",
        "K_LIST": [1, 2, 4],
        "N_SAMPLING": 4,  # Avg@4
        "NUM_SHOTS": 0,
        "PROMPT_TYPE": "qwen25",
        "TEMPERATURE": 0.6,
        "TOP_P": 0.95,
        "MAX_TOKENS": 4096,
    },
    "amc23": {
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/node/amc23_avg16",
        "EVAL_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/math_eval.py",
        "DATA_DIR": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/data",
        "DATA_NAME": "amc23",
        "K_LIST": [1, 8, 16],
        "N_SAMPLING": 16,  # Avg@16
        "NUM_SHOTS": 0,
        "PROMPT_TYPE": "qwen25",
        "TEMPERATURE": 0.6,
        "TOP_P": 0.95,
        "MAX_TOKENS": 4096,
    },
    "aime24": {
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/node/aime24_avg32",
        "EVAL_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/math_eval.py",
        "DATA_DIR": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/data",
        "DATA_NAME": "aime24",
        "K_LIST": [1, 8, 16, 32],
        "N_SAMPLING": 32,  # Avg@32
        "NUM_SHOTS": 0,
        "PROMPT_TYPE": "qwen25",
        "TEMPERATURE": 0.6,
        "TOP_P": 0.95,
        "MAX_TOKENS": 15360,
    },
    "aime25": {
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/node/aime25_avg32",
        "EVAL_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/math_eval.py",
        "DATA_DIR": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/data",
        "DATA_NAME": "aime25",
        "K_LIST": [1, 8, 16, 32],
        "N_SAMPLING": 32,  # Avg@32
        "NUM_SHOTS": 0,
        "PROMPT_TYPE": "qwen25",
        "TEMPERATURE": 0.6,
        "TOP_P": 0.95,
        "MAX_TOKENS": 15360,
    },
}

# SLURM configuration for 8-GPU node
SLURM_CONFIG = {
    "NODES": 1,
    "NTASKS": 1,
    "CPUS_PER_TASK": 96,
    "GPUS": 8,
    "MEM": "0",
    "TIME_LIMIT": "24:00:00",
    "PARTITION": "main",
    "TENSOR_PARALLEL_SIZE": 8,
    "CONDA_ENV": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval",
    "CD_PATH": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/node",
}

# SBATCH template
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{model_name}.out
#SBATCH --error={log_dir}/{model_name}.err
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-task={gpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

cd {cd_path}
{conda_env}
which python

export TOKENIZERS_PARALLELISM=false

{eval_cmd}

echo "✅ {model_name} evaluation finished"
"""


def pass_at_k(n, c, k):
    """Calculate pass@k metric."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def is_job_done(model_out_dir):
    """Check if evaluation is already complete."""
    result_csv = os.path.join(model_out_dir, "result.csv")
    return os.path.exists(result_csv)


def submit_jobs_for_all_models(args, task_config):
    """Submit SLURM jobs for all models."""
    base_out = task_config["BASE_OUT"]
    os.makedirs(base_out, exist_ok=True)
    
    job_dir = os.path.join(base_out, "job_scripts")
    log_dir = os.path.join(base_out, "logs")
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    models_to_run = []
    models_skipped = []
    sbatch_commands = []
    submitted_slurm_job_ids = []
    
    for model_path, model_name in model_map.items():
        model_out_dir = os.path.join(base_out, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        # Check if already done
        if not args.reforce and is_job_done(model_out_dir):
            models_skipped.append(model_name)
            continue
        
        models_to_run.append(model_name)
        job_name = f"{args.task}_{model_name}"
        job_script = os.path.join(job_dir, f"job_{model_name}.sh")
        
        # Use qwen2.5-math evaluation script (SFT-style, apply_chat_template)
        eval_cmd = f"""python3 -u {task_config['EVAL_SCRIPT']} \\
    --model_name_or_path {model_path} \\
    --data_names {task_config['DATA_NAME']} \\
    --data_dir {task_config['DATA_DIR']} \\
    --output_dir {model_out_dir} \\
    --split test \\
    --prompt_type {task_config['PROMPT_TYPE']} \\
    --num_test_sample -1 \\
    --seed 0 \\
    --temperature {task_config['TEMPERATURE']} \\
    --n_sampling {task_config['N_SAMPLING']} \\
    --top_p {task_config['TOP_P']} \\
    --max_tokens_per_call {task_config['MAX_TOKENS']} \\
    --start 0 \\
    --end -1 \\
    --use_vllm \\
    --save_outputs \\
    --overwrite \\
    --num_shots {task_config['NUM_SHOTS']} \\
    --apply_chat_template"""
        
        with open(job_script, "w") as f:
            f.write(SBATCH_TEMPLATE.format(
                job_name=job_name,
                log_dir=log_dir,
                model_name=model_name,
                nodes=SLURM_CONFIG["NODES"],
                ntasks=SLURM_CONFIG["NTASKS"],
                cpus_per_task=SLURM_CONFIG["CPUS_PER_TASK"],
                gpus=SLURM_CONFIG["GPUS"],
                mem=SLURM_CONFIG["MEM"],
                time_limit=SLURM_CONFIG["TIME_LIMIT"],
                partition=SLURM_CONFIG["PARTITION"],
                cd_path=SLURM_CONFIG["CD_PATH"],
                conda_env=SLURM_CONFIG["CONDA_ENV"],
                eval_cmd=eval_cmd,
            ))
        
        sbatch_commands.append((f"sbatch {job_script}", model_name))
    
    # Print summary
    print("\n--- 📝 Summary of the model evaluation plan ---")
    if models_to_run:
        print(f"\n📊 Models to run (total {len(models_to_run)} models):")
        for model in models_to_run:
            print(f"  - {model}")
    
    if models_skipped:
        print(f"\n📌 Skipped models (total {len(models_skipped)} models, result.csv exists):")
        for model in models_skipped:
            print(f"  - {model}")
    
    if not models_to_run and not models_skipped:
        print("\nNo models to run.")
    
    # Submit jobs
    if sbatch_commands:
        print(f"\nSubmitting {len(sbatch_commands)} jobs...")
        for cmd, model_name in sbatch_commands:
            try:
                result = subprocess.check_output(cmd, shell=True, text=True)
                match = re.search(r'Submitted batch job (\d+)', result)
                if match:
                    job_id = match.group(1)
                    submitted_slurm_job_ids.append(job_id)
                    print(f"Successfully submitted job: {model_name} (Job ID: {job_id})")
                else:
                    print(f"Submitted job but could not parse job ID: {model_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit job: {model_name}")
            time.sleep(0.2)
    
    print(f"\nAll model evaluation tasks have been submitted (submitted {len(models_to_run)} models, skipped {len(models_skipped)} models).")
    return submitted_slurm_job_ids, models_to_run, models_skipped


def wait_for_jobs_completion(submitted_slurm_job_ids):
    """Wait for all SLURM jobs to complete."""
    if not submitted_slurm_job_ids:
        return
    
    print(f"\n⏳ Waiting for all {len(submitted_slurm_job_ids)} Slurm jobs to complete...")
    
    while submitted_slurm_job_ids:
        job_ids_str = ",".join(submitted_slurm_job_ids)
        try:
            squeue_output = subprocess.check_output(
                f"squeue -h -j {job_ids_str} -o '%i %t'",
                shell=True,
                text=True
            )
            
            running_jobs = set()
            for line in squeue_output.strip().split('\n'):
                if line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id = parts[0]
                        status = parts[1]
                        if status in ['R', 'PD', 'CG']:
                            running_jobs.add(job_id)
            
            submitted_slurm_job_ids = [job_id for job_id in submitted_slurm_job_ids if job_id in running_jobs]
            
            if submitted_slurm_job_ids:
                print(f"📊 {len(submitted_slurm_job_ids)} jobs still running/pending. Checking again in 60 seconds...")
                time.sleep(60)
        
        except subprocess.CalledProcessError:
            print("All jobs appear to have completed.")
            break
    
    print("\n✅ All Slurm jobs have completed!")


def postprocess_task_results(base_out, task_config):
    """Post-process results for a single dataset."""
    import glob
    import shutil
    data_name = task_config["DATA_NAME"]
    
    for model_path, model_name in model_map.items():
        model_out_dir = os.path.join(base_out, model_name)
        task_dir = os.path.join(model_out_dir, data_name)

        if not os.path.isdir(task_dir):
            continue
        
        # Find nested directories
        checkpoint_dirs = glob.glob(os.path.join(task_dir, "checkpoint-*"))
        subdirs = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        
        target_dir = None
        if checkpoint_dirs:
            target_dir = checkpoint_dirs[0]
        elif subdirs:
            target_dir = os.path.join(task_dir, subdirs[0])
        
        if target_dir and os.path.isdir(target_dir):
            # Move jsonl files
            jsonl_files = glob.glob(os.path.join(target_dir, "*.jsonl"))
            if jsonl_files:
                shutil.move(jsonl_files[0], os.path.join(model_out_dir, "sample.jsonl"))
            
            # Move json files
            json_files = glob.glob(os.path.join(target_dir, "*.json"))
            if json_files:
                shutil.move(json_files[0], os.path.join(model_out_dir, "result.json"))
        
        # Generate result.csv from sample.jsonl
        sample_jsonl = os.path.join(model_out_dir, "sample.jsonl")
        if os.path.exists(sample_jsonl):
            all_scores = []
            with open(sample_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    scores = obj.get("score", [])
                    if isinstance(scores, list):
                        row = [1 if s is True or s == 1 else 0 for s in scores]
                    else:
                        row = [1 if scores is True or scores == 1 else 0]
                    all_scores.append(row)
            
            csv_path = os.path.join(model_out_dir, "result.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_scores)
            
            # Calculate pass@k
            if all_scores:
                passk_dict = {}
                for k in task_config["K_LIST"]:
                    pass_at_k_scores = []
                    for em_list in all_scores:
                        n = len(em_list)
                        c = sum(em_list)
                        pass_at_k_scores.append(pass_at_k(n, c, k))
                    passk_dict[f"pass@{k}"] = float(np.mean(pass_at_k_scores))
                
                total_correct = sum(sum(row) for row in all_scores)
                total_attempts = sum(len(row) for row in all_scores)
                exact_match = total_correct / total_attempts if total_attempts > 0 else 0.0
                avg_k = task_config["N_SAMPLING"]
                
                metrics_path = os.path.join(model_out_dir, "metrics.txt")
                with open(metrics_path, "w") as f:
                    f.write(f"exact_match: {exact_match:.4f}\n")
                    f.write(f"avg@{avg_k}: {exact_match:.4f}\n")
                    for k in task_config["K_LIST"]:
                        f.write(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}\n")


def summarize_results(task_name, base_out, task_config):
    """Summarize pass@k results for all models."""
    all_results = {}
    
    for model_path, model_name in model_map.items():
        model_dir = os.path.join(base_out, model_name)
        csv_path = os.path.join(model_dir, "result.csv")
        
        if not os.path.exists(csv_path):
            continue
        
        data = pd.read_csv(csv_path, header=None).values
        all_samples = data.tolist()
        
        results = {}
        total_correct = sum(sum(row) for row in all_samples)
        total_attempts = sum(len(row) for row in all_samples)
        avg_score = total_correct / total_attempts if total_attempts > 0 else 0.0
        results[f"avg@{task_config['N_SAMPLING']}"] = float(avg_score)
        for k in task_config["K_LIST"]:
            pass_at_k_scores = []
            for em_list in all_samples:
                n = len(em_list)
                c = sum(em_list)
                pass_at_k_scores.append(pass_at_k(n, c, k))
            results[f"pass@{k}"] = float(np.mean(pass_at_k_scores))
        
        all_results[model_name] = results
    
    # Save result.json
    result_json = os.path.join(base_out, "result.json")
    with open(result_json, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ All models pass@k results saved to: {result_json}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Math Evaluation for 8-GPU Node Models")
    parser.add_argument(
        "--task",
        type=str,
        default="math500",
        choices=["math500", "amc23", "aime24", "aime25"],
        help="Task name: math500, amc23, aime24, or aime25",
    )
    parser.add_argument("--reforce", action="store_true",
                        help="If set, rerun evaluation even if result.csv exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task_config = TASK_CONFIGS[args.task]

    # Job submission mode
    print(f"=== Math Evaluation: {args.task} ===")
    print(f"N_SAMPLING: {task_config['N_SAMPLING']} (Avg@{task_config['N_SAMPLING']})")
    print(f"Temperature: {task_config['TEMPERATURE']}, Top-p: {task_config['TOP_P']}")
    print(f"Output: {task_config['BASE_OUT']}")
    print(f"GPUs: {SLURM_CONFIG['GPUS']} (tensor_parallel_size={SLURM_CONFIG['TENSOR_PARALLEL_SIZE']})")

    # Submit jobs
    submitted_ids, models_run, models_skipped_list = submit_jobs_for_all_models(args, task_config)

    # Wait for completion
    wait_for_jobs_completion(submitted_ids)

    # Post-process results
    print("\n📝 Post-processing results...")
    postprocess_task_results(task_config["BASE_OUT"], task_config)

    # Summarize results
    summarize_results(args.task, task_config["BASE_OUT"], task_config)

    print("\n🎉 All tasks completed!")
