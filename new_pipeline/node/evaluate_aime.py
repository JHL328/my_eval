#!/usr/bin/env python
"""
AIME24/25 Evaluation Script for 8-GPU Node Models
Supports Avg@32 evaluation using lm-evaluation-harness
"""

import os
import sys
import time
import argparse
import subprocess
import json
import re
from pathlib import Path
import shutil
import numpy as np

# Add harness root to sys.path to import tasks.aime.utils
harness_root = "/mnt/weka/home/haolong.jia/eval/RL-eval/lm-evaluation-harness"
if harness_root not in sys.path:
    sys.path.append(harness_root)

try:
    from lm_eval.tasks.aime import utils as aime_utils
except ImportError:
    print(f"⚠️ Warning: Could not import lm_eval.tasks.aime.utils from {harness_root}")
    aime_utils = None

# Import model_map from local model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import model_map

# =====================
# Task Configurations
# =====================
TASK_CONFIGS = {
    "aime24": {
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/node/aime24_avg32",
        "HARNESS_TASK": "aime24_avg32",  # Use the new yaml with repeats=32
        "N_SAMPLING": 32,  # Avg@32
        "K_LIST": [1, 2, 4, 8, 16, 32],
        "TEMPERATURE": 0.7,
        "TOP_P": 0.95,
        "MAX_GEN_TOKS": 12288,  # Leave 4096 for prompt (Total 16384)
        "NUM_FEWSHOT": 0,
    },
    "aime25": {
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/node/aime25_avg32",
        "HARNESS_TASK": "aime25_avg32",  # Use the new yaml with repeats=32
        "N_SAMPLING": 32,  # Avg@32
        "K_LIST": [1, 2, 4, 8, 16, 32],
        "TEMPERATURE": 0.7,
        "TOP_P": 0.95,
        "MAX_GEN_TOKS": 12288,  # Leave 4096 for prompt (Total 16384)
        "NUM_FEWSHOT": 0,
    },
}

# SLURM configuration for 8-GPU node
SLURM_CONFIG = {
    "NODES": 1,
    "NTASKS": 1,
    "CPUS_PER_TASK": 96,
    "GPUS": 8,
    "MEM": "0",  # Use all available memory
    "TIME_LIMIT": "24:00:00",
    "PARTITION": "main",
    "TENSOR_PARALLEL_SIZE": 8,
    "CONDA_ENV": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval",
    "CD_PATH": "/mnt/weka/home/haolong.jia/eval/RL-eval/lm-evaluation-harness",
}

# SBATCH template for 8-GPU node
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

{lm_eval_cmd}

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
    metrics_txt = os.path.join(model_out_dir, "metrics.txt")
    return os.path.exists(metrics_txt)


def post_process_aime_results(task_name, output_dir, task_config):
    """Post-process AIME evaluation results to compute Avg@k metrics."""
    output_path = Path(output_dir)
    overall_summary = {}
    n_sampling = task_config["N_SAMPLING"]

    if aime_utils is None:
        print("❌ Error: aime_utils not loaded. Cannot evaluate correctness.")
        return

    def normalize_resps(raw_resps):
        if raw_resps is None:
            return []
        if isinstance(raw_resps, str):
            return [raw_resps]
        if isinstance(raw_resps, list):
            flattened = []
            for item in raw_resps:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        return []

    for model_path, model_name in model_map.items():
        model_dir = output_path / model_name
        if not model_dir.exists():
            print(f"⚠️ {model_name} output directory not found, skip")
            continue

        # Find sample files
        sample_files = sorted(model_dir.rglob("samples_*.jsonl"))
        if not sample_files:
            print(f"⚠️ {model_name} no sample files found, skip")
            continue

        print(f"\n📂 {model_name}: located {len(sample_files)} sample file(s)")

        # Parse samples and compute metrics
        doc_results = {}  # doc_id -> list of exact_match values

        for sample_path in sample_files:
            try:
                with sample_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        doc_id = record.get("doc_id", 0)
                        doc_content = record.get("doc")
                        raw_resps = record.get("resps")
                        if raw_resps is None:
                            raw_resps = record.get("filtered_resps")

                        if raw_resps is None or doc_content is None:
                            continue

                        responses = normalize_resps(raw_resps)
                        if not responses:
                            continue
                        if len(responses) > n_sampling:
                            responses = responses[:n_sampling]

                        sample_results = []
                        for resp in responses:
                            metrics = aime_utils.process_results(doc_content, [resp])
                            sample_results.append(metrics.get("exact_match", 0))

                        if doc_id not in doc_results:
                            doc_results[doc_id] = []
                        doc_results[doc_id].extend(sample_results)

            except OSError as exc:
                print(f"  ⚠️ Failed to read {sample_path}: {exc}")
                continue

        if not doc_results:
            print(f"  ⚠️ {model_name} has no valid samples. Skipping.")
            continue

        # Compute pass@k metrics
        passk_dict = {}

        for k in task_config["K_LIST"]:
            if k > n_sampling:
                continue
            pass_at_k_scores = []
            for doc_id, em_list in doc_results.items():
                n = len(em_list)
                c = sum(em_list)
                if n >= k:
                    sample_pass_k = pass_at_k(n, c, k)
                    pass_at_k_scores.append(sample_pass_k)
            if pass_at_k_scores:
                passk_dict[f"pass@{k}"] = float(np.mean(pass_at_k_scores))

        # Compute overall exact match (average accuracy)
        total_correct = sum(sum(em_list) for em_list in doc_results.values())
        total_attempts = sum(len(em_list) for em_list in doc_results.values())
        exact_match = total_correct / total_attempts if total_attempts > 0 else 0.0
        avg_k = n_sampling

        overall_summary[model_name] = {
            f"avg@{avg_k}": exact_match,
            "exact_match": exact_match,
            "total_questions": len(doc_results),
            "total_samples": total_attempts,
            **passk_dict
        }

        print(f"  → {model_name}: exact_match={exact_match:.4f}, pass@1={passk_dict.get('pass@1', 0):.4f}")
        
        # Save per-model metrics
        metrics_path = model_dir / "metrics.txt"
        with metrics_path.open("w") as f:
            f.write(f"exact_match: {exact_match:.4f}\n")
            f.write(f"avg@{avg_k}: {exact_match:.4f}\n")
            for k in task_config["K_LIST"]:
                if f"pass@{k}" in passk_dict:
                    f.write(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}\n")
        
        # Move and rename result files for cleaner structure
        subdirs = [p for p in model_dir.iterdir() if p.is_dir()]
        for subdir in subdirs:
            result_src = next(subdir.glob("results_*.json"), None)
            if result_src is not None:
                dest = model_dir / "harness_result.json"
                if dest.exists():
                    dest.unlink()
                shutil.move(str(result_src), dest)
            
            sample_src = next(subdir.glob("samples_*.jsonl"), None)
            if sample_src is not None:
                dest = model_dir / "samples.jsonl"
                if dest.exists():
                    dest.unlink()
                shutil.move(str(sample_src), dest)
            
            try:
                shutil.rmtree(subdir)
            except OSError:
                pass
    
    if not overall_summary:
        print("⚠️ No metrics were computed; result.json will not be written.")
        return
    
    # Save overall result.json
    overall_result_path = output_path / "result.json"
    with overall_result_path.open("w", encoding="utf-8") as fh:
        json.dump(overall_summary, fh, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved aggregated metrics to {overall_result_path}")


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
        
        # Build lm_eval command - the yaml file already has repeats=32 configured
        # So we only need a single run, and the harness will automatically run 32 times
        # max_model_len=16384 sets the total context window.
        # The YAML config sets max_tokens=12288 for generation.
        # This leaves 16384 - 12288 = 4096 tokens for the prompt.
        lm_eval_cmd = f"""lm_eval --model vllm \\
  --model_args pretrained={model_path},tensor_parallel_size={SLURM_CONFIG['TENSOR_PARALLEL_SIZE']},dtype=bfloat16,gpu_memory_utilization=0.95,max_model_len=16384,trust_remote_code=True \\
  --tasks {task_config['HARNESS_TASK']} \\
  --output_path {model_out_dir} \\
  --batch_size auto \\
  --log_samples \\
  --num_fewshot {task_config['NUM_FEWSHOT']} \\
  --trust_remote_code"""
        
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
                lm_eval_cmd=lm_eval_cmd,
            ))
        
        sbatch_commands.append((f"sbatch {job_script}", model_name))
    
    # Print summary
    print("\n--- 📝 Summary of the model evaluation plan ---")
    if models_to_run:
        print(f"\n📊 Models to run (total {len(models_to_run)} models):")
        for model in models_to_run:
            print(f"  - {model}")
    
    if models_skipped:
        print(f"\n📌 Skipped models (total {len(models_skipped)} models, result.json exists):")
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


def parse_args():
    parser = argparse.ArgumentParser(description="AIME24/25 Evaluation for 8-GPU Node Models")
    parser.add_argument("--task", type=str, default="aime24", choices=["aime24", "aime25"],
                        help="Task name: aime24 or aime25")
    parser.add_argument("--reforce", action="store_true",
                        help="If set, rerun evaluation even if result.json exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task_config = TASK_CONFIGS[args.task]
    
    print(f"=== AIME Evaluation: {args.task} ===")
    print(f"N_SAMPLING: {task_config['N_SAMPLING']} (Avg@{task_config['N_SAMPLING']})")
    print(f"Temperature: {task_config['TEMPERATURE']}, Top-p: {task_config['TOP_P']}")
    print(f"Output: {task_config['BASE_OUT']}")
    print(f"GPUs: {SLURM_CONFIG['GPUS']} (tensor_parallel_size={SLURM_CONFIG['TENSOR_PARALLEL_SIZE']})")
    
    # Submit jobs
    submitted_ids, models_run, models_skipped_list = submit_jobs_for_all_models(args, task_config)
    
    # Wait for completion
    wait_for_jobs_completion(submitted_ids)
    
    # Post-process results
    print("\n📝 Post-processing AIME results...")
    post_process_aime_results(args.task, task_config["BASE_OUT"], task_config)
    
    print("\n🎉 All tasks completed!")
