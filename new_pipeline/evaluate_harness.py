#!/usr/bin/env python
"""Evaluation script for lm-evaluation-harness tasks (MMLU Redux, IfEval, etc.)."""

import os
import time
import sys
import argparse
import subprocess
import json
import re
from pathlib import Path
import shutil
from typing import Dict, Optional
from typing import Dict, Optional

def post_process(task, output_dir, model_map):
    """Post-process harness outputs to compute aggregated scores."""

    output_path = Path(output_dir)

    if task == "mmlu_redux_generative":
        overall_summary = {}

        for model_path, model_name in model_map.items():
            model_dir = output_path / model_name
            if not model_dir.exists():
                print(f"⚠️ {model_name} output directory not found, skip")
                continue

            sample_files = sorted(model_dir.rglob("samples*.jsonl"))
            print(f"\n📂 {model_name}: located {len(sample_files)} sample file(s)")

            total_samples = 0
            exact_match_sum = 0.0

            for sample_path in sample_files:
                file_samples = 0
                file_sum = 0.0
                try:
                    with sample_path.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as exc:
                                print(f"  ⚠️ {sample_path.name}: JSON decode error ({exc}) - skip line")
                                continue

                            exact_match = record.get("exact_match")
                            if exact_match is None:
                                continue
                            try:
                                val = float(exact_match)
                            except (TypeError, ValueError):
                                print(f"  ⚠️ {sample_path.name}: invalid exact_match value {exact_match}")
                                continue

                            total_samples += 1
                            file_samples += 1
                            exact_match_sum += val
                            file_sum += val
                except OSError as exc:
                    print(f"  ⚠️ Failed to read {sample_path}: {exc}")
                    continue

                print(
                    f"  • {sample_path.name}: samples={file_samples}, exact_match_sum={file_sum:.4f}"
                )

            if total_samples == 0:
                print(f"  ⚠️ {model_name} has no valid samples. Skipping score computation.")
                continue

            score = exact_match_sum / total_samples
            overall_summary[model_name] = {
                "exact_match": score,
                "total_samples": total_samples,
                "exact_match_sum": exact_match_sum,
                "sample_files": len(sample_files),
            }

            print(
                f"  → {model_name}: score={score:.6f} (sum={exact_match_sum:.4f}, total={total_samples})"
            )

        if not overall_summary:
            print("⚠️ No scores were computed; result.json will not be updated.")
            return

        overall_result_path = output_path / "result.json"
        with overall_result_path.open("w", encoding="utf-8") as fh:
            json.dump(overall_summary, fh, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved aggregated metrics to {overall_result_path}")

    elif task == "ifeval":
        metrics_keys = [
            "prompt_level_strict_acc,none",
            "prompt_level_strict_acc_std_err,none",
            "inst_level_strict_acc,none",
            "inst_level_strict_acc_stderr,none",
            "prompt_level_loose_acc,none",
            "prompt_level_loose_acc_stderr,none",
            "inst_level_loose_acc,none",
            "inst_level_loose_acc_stderr,none",
        ]

        overall_summary: Dict[str, Dict[str, Optional[float]]] = {}

        for model_path, model_name in model_map.items():
            model_dir = output_path / model_name
            if not model_dir.exists():
                print(f"⚠️ {model_name} output directory not found, skip")
                continue

            subdirs = [p for p in model_dir.iterdir() if p.is_dir()]
            for subdir in subdirs:
                result_src = next(subdir.glob("results_*.json"), None)
                if result_src is not None:
                    dest = model_dir / "result.json"
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

            result_path = model_dir / "result.json"
            if not result_path.exists():
                print(f"⚠️ {model_name} result.json not found after moving files, skip")
                continue

            try:
                with result_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:
                print(f"⚠️ Failed to read {result_path}: {exc}")
                continue

            metrics_source = {}
            if isinstance(data, dict):
                results = data.get("results")
                if isinstance(results, dict):
                    candidate = results.get("ifeval")
                    if isinstance(candidate, dict):
                        metrics_source = candidate
                    else:
                        for value in results.values():
                            if isinstance(value, dict):
                                metrics_source = value
                                break

            model_metrics = {key: metrics_source.get(key) for key in metrics_keys}
            overall_summary[model_name] = model_metrics

        if not overall_summary:
            print("⚠️ No IfEval metrics were collected; result.json will not be written.")
            return

        overall_result_path = output_path / "result.json"
        with overall_result_path.open("w", encoding="utf-8") as fh:
            json.dump(overall_summary, fh, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved IfEval metrics to {overall_result_path}")

    else:
        print(f"⚠️ No post-processing implemented for task {task}")
    



# dynamically import Model_map
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Model_map, get_model_map_by_type


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='mmlu_redux_generative', help='Task name for evaluation')
parser.add_argument('--model-type', type=str, default='sft', choices=['base', 'sft'], help='Model type')
args = parser.parse_args()

task = args.task
output_dir = f"/mnt/sharefs/users/haolong.jia/result/{task}"
job_dir = os.path.join(output_dir, "job_scripts")
log_dir = os.path.join(output_dir, "logs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(job_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Select model map based on type
if args.model_type == 'sft':
    model_map = get_model_map_by_type('sft')
else:
    model_map = Model_map

# prepare stage: initialize the lists
models_to_run = []
models_skipped = []
sbatch_commands = []
submitted_slurm_job_ids = []

# SBATCH template
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task}_{model_name}
#SBATCH --output={log_dir}/{model_name}.out
#SBATCH --error={log_dir}/{model_name}.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --mem=160G

cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-sft

{lm_eval_cmd}

echo "✅ {model_name} evaluation finished (post-processing disabled)"
"""

for model_path, model_name in model_map.items():
    model_out_dir = os.path.join(output_dir, model_name)
    result_json_path = os.path.join(model_out_dir, "result.json")

    # check if result.json already exists
    if os.path.exists(result_json_path):
        models_skipped.append(model_name)
        continue

    # if the model is not skipped, add to the list of models to run
    models_to_run.append(model_name)

    # create the model output directory
    os.makedirs(model_out_dir, exist_ok=True)
    job_script = os.path.join(job_dir, f"job_{model_name}.sh")

    # build the lm_eval command
    if task == "mmlu_redux_generative":
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.95,max_model_len=8192,trust_remote_code=True \
  --tasks mmlu_redux_generative \
  --output_path {model_out_dir} \
  --gen_kwargs temperature=0.6,top_p=0.95 \
  --batch_size auto \
  --log_samples \
  --num_fewshot 0 \
  --trust_remote_code"""
    elif task == "ifeval":
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.95,max_model_len=8192,trust_remote_code=True \
  --tasks ifeval \
  --output_path {model_out_dir} \
  --gen_kwargs temperature=0.6,top_p=0.95,do_sample=True,max_gen_toks=4096 \
  --batch_size auto \
  --log_samples \
  --num_fewshot 0 \
  --trust_remote_code"""

        # Add apply_chat_template for sft models
        if args.model_type == 'sft':
            lm_eval_cmd += " \\\n  --apply_chat_template"

    # write the job script
    with open(job_script, "w") as f:
        f.write(SBATCH_TEMPLATE.format(
            model_name=model_name,
            model_out_dir=model_out_dir,
            log_dir=log_dir,
            task=task,
            lm_eval_cmd=lm_eval_cmd
        ))

    # add the submission command to the list
    sbatch_commands.append((f"sbatch {job_script}", model_name))

# print the summary information
print("\n--- 📝 summary of the model evaluation plan ---")
if models_to_run:
    print(f"\n📊models to run (total {len(models_to_run)} models):")
    for model in models_to_run:
        print(f"  - {model}")

if models_skipped:
    print(f"\n📌 skipped models (total {len(models_skipped)} models, because result.json already exists):")
    for model in models_skipped:
        print(f"  - {model}")

if not models_to_run and not models_skipped:
    print("\nno models to run.")

# submit the jobs and capture job IDs
if sbatch_commands:
    print(f"\nsubmitting {len(sbatch_commands)} jobs...")
    for cmd, model_name in sbatch_commands:
        try:
            result = subprocess.check_output(cmd, shell=True, text=True)
            match = re.search(r'Submitted batch job (\d+)', result)
            if match:
                job_id = match.group(1)
                submitted_slurm_job_ids.append(job_id)
                print(f"successfully submitted job: {model_name} (Job ID: {job_id})")
            else:
                print(f"submitted job but could not parse job ID: {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"failed to submit job: {model_name}")
        time.sleep(0.2)

# print initial completion information
print(f"\nall model evaluation tasks have been submitted (submitted {len(models_to_run)} models, skipped {len(models_skipped)} models).")

# wait for all jobs to complete
if submitted_slurm_job_ids:
    print(f"\n⏳ waiting for all {len(submitted_slurm_job_ids)} Slurm jobs to complete...")

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
                print(f"📊 {len(submitted_slurm_job_ids)} jobs still running/pending. Checking again in 30 seconds...")
                time.sleep(30)

        except subprocess.CalledProcessError:
            print("All jobs appear to have completed.")
            break

    print("\n✅ All Slurm jobs have completed!")

# Post-process outputs to compute metrics
print("\n📝 Post-processing harness outputs...")
post_process(task, output_dir, model_map)

print("\n🎉 All tasks completed!")
