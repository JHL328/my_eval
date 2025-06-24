import os
import time
import sys
import argparse
import subprocess
import json
import re
import multiprocessing
import numpy as np
# from evalplus import evaluate # We don't call this directly in python anymore
# from evalplus import sanitize # We don't call this directly in python anymore

# Import necessary parts from evalplus for pass@k calculation
from evalplus.eval import estimate_pass_at_k, PASS # Import PASS for status check

# dynamic import model list
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model_map

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Task name for evaluation: humaneval or mbpp')
args = parser.parse_args()
task = args.task

output_dir = f"/mnt/sharefs/users/haolong.jia/result/{task}"
job_dir = os.path.join(output_dir, "job_scripts")
log_dir = os.path.join(output_dir, "logs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(job_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# parameters
N_SAMPLES = 64 # For faster debugging, revert to 64 for actual runs
TEMPERATURE = 0.6
TP_SIZE = 1 # Tensor Parallel size for LLM instantiation in generate_code.py
MAX_TOKENS = 1024 # Max tokens for generation

models_to_run = []
models_skipped = []
sbatch_commands = []
submitted_slurm_job_ids = []

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task}_{model_name}
#SBATCH --output={log_dir}/{model_name}.out
#SBATCH --error={log_dir}/{model_name}.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
# SBATCH --partition=main
#SBATCH --mem=180G

cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate evalplus-eval

# Step 1: Generate code samples
echo "[Step 1] Generating code samples for {model_name}..."
python new_pipeline/generate_code.py \
  --model_path {model_path} \
  --dataset {task} \
  --n_samples {n_samples} \
  --temperature {temperature} \
  --tensor_parallel_size {tp_size} \
  --output_dir {raw_samples_dir} \
  --max_tokens {max_tokens}
if [ $? -ne 0 ]; then
    echo "Error: Step 1 (generate_code.py) failed for {model_name}. Exiting."
    exit 1
fi
echo "Step 1 completed. Listing contents of raw_samples_dir: {raw_samples_dir}"


# Step 2: Sanitize the generated samples
echo "[Step 2] Sanitizing code samples for {model_name}..."
python -m evalplus.sanitize --samples {raw_samples_dir}/samples.jsonl
if [ $? -ne 0 ]; then
    echo "Error: Step 2 (evalplus.sanitize) failed for {model_name}. Exiting."
    exit 1
fi
SAN_SAMPLES_JSONL="{raw_samples_dir}/samples-sanitized.jsonl"
SAMPLES_TO_EVAL="{raw_samples_dir}/samples.jsonl" # Default to raw samples
if [ -f "$SAN_SAMPLES_JSONL" ] && [ -s "$SAN_SAMPLES_JSONL" ]; then
    echo "Sanitized file $SAN_SAMPLES_JSONL exists and is not empty. Using it for evaluation."
    SAMPLES_TO_EVAL="$SAN_SAMPLES_JSONL"
else
    echo "Sanitized file $SAN_SAMPLES_JSONL does NOT exist or is empty. Using raw samples for evaluation."
fi
echo "Will use $SAMPLES_TO_EVAL for evaluation."

# Step 3: Evaluate the samples
echo "[Step 3] Evaluating code samples for {model_name} using $SAMPLES_TO_EVAL ..."
python -m evalplus.evaluate \
  --dataset {task} \
  --samples "$SAMPLES_TO_EVAL" > {results_txt_path}
if [ $? -ne 0 ]; then
    echo "Error: Step 3 (evalplus.evaluate) failed for {model_name}. Exiting."
    exit 1
fi
echo "✅ Step 3 completed for {model_name}. Results redirected to {results_txt_path}. The full results JSON will be at $SAMPLES_TO_EVAL.eval_results.json"
"""

for model_path, model_name in Model_map.items():
    model_out_dir = os.path.join(output_dir, model_name)
    # Path for the raw text output from evalplus.evaluate
    results_txt_path = os.path.join(model_out_dir, "results.txt") 
    raw_samples_dir = os.path.join(model_out_dir, "raw_samples")
    
    # Check if the final aggregated JSON for this model type already implies completion,
    # or if the specific results.txt exists. For simplicity, we check results.txt.
    if os.path.exists(results_txt_path): # If raw text results exist, assume job was run.
        # Potentially, one might want to check if base_passk.json AND plus_passk.json for this model already exist
        # but for now, skipping if raw results.txt is present is a simpler skip logic.
        models_skipped.append(model_name)
        continue
    models_to_run.append(model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(raw_samples_dir, exist_ok=True)
    job_script = os.path.join(job_dir, f"job_{model_name}.sh")

    sbatch_script_content = SBATCH_TEMPLATE.format(
        model_name=model_name,
        model_path=model_path,
        log_dir=log_dir,
        task=task,
        n_samples=N_SAMPLES,
        temperature=TEMPERATURE,
        tp_size=TP_SIZE,
        max_tokens=MAX_TOKENS,
        raw_samples_dir=raw_samples_dir,
        results_txt_path=results_txt_path # Pass the new path to the template
    )
    with open(job_script, "w") as f:
        f.write(sbatch_script_content)
    sbatch_commands.append((f"sbatch {job_script}", model_name))

print("\n--- 📝 summary of the code evaluation plan ---")
if models_to_run:
    print(f"\n📊models to run (total {len(models_to_run)} models):")
    for model in models_to_run:
        print(f"  - {model}")
if models_skipped:
    print(f"\n📌 skipped models (total {len(models_skipped)} models, because results.txt already exists):")
    for model in models_skipped:
        print(f"  - {model}")
if not models_to_run and not models_skipped:
    print("\nno models to run.")

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
                print(f"submitted job but could not parse job ID for {model_name}: {result.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"failed to submit job {model_name}: {e}. Output: {e.output}")
        time.sleep(0.2)

print(f"\nall code evaluation tasks have been submitted (submitted {len(models_to_run)} models, skipped {len(models_skipped)} models).")

if submitted_slurm_job_ids:
    print(f"\n⏳ waiting for all {len(submitted_slurm_job_ids)} Slurm jobs to complete...")
    active_jobs = set(submitted_slurm_job_ids)
    while active_jobs:
        job_ids_str = ",".join(active_jobs)
        current_running_or_pending = set()
        try:
            squeue_output = subprocess.check_output(
                f"squeue -h -j {job_ids_str} -o '%i %t'", 
                shell=True, 
                text=True,
                stderr=subprocess.STDOUT 
            )
            for line in squeue_output.strip().split('\n'):
                if line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id, status = parts[0], parts[1]
                        if status in ['R', 'PD', 'CG', 'CF']: 
                            current_running_or_pending.add(job_id)
            
            active_jobs = current_running_or_pending
            if active_jobs:
                print(f"📊 {len(active_jobs)} jobs still active: {sorted(list(active_jobs))}. Checking again in 30 seconds...")
                time.sleep(30)
            else:
                print("All submitted Slurm jobs have finished processing according to squeue.")
                break 
        except subprocess.CalledProcessError as e:
            if "Invalid job id specified" in e.output or "No jobs in partition" in e.output:
                 print("squeue reported no active jobs (or invalid job ID), assuming completion.")
                 active_jobs.clear()
            else:
                print(f"Error querying squeue: {e.output}. Will retry in 30 seconds...")
            time.sleep(30) 
    print("\n✅ All Slurm jobs have completed or are no longer tracked!")

print("\n📝 Generating overall pass@k JSON files...")
overall_base_summary = {}
overall_plus_summary = {}

# We iterate using Model_map to ensure all defined models are processed
ALL_PASS_K_VALUES = [1, 4, 8, 10, 16, 32, 64] # Define desired K values

for _, model_name in Model_map.items(): 
    model_out_dir = os.path.join(output_dir, model_name)
    raw_samples_dir = os.path.join(model_out_dir, "raw_samples")

    # Determine the actual samples.jsonl path used for evaluation
    sanitized_samples_jsonl_path = os.path.join(raw_samples_dir, "samples-sanitized.jsonl")
    raw_samples_jsonl_path = os.path.join(raw_samples_dir, "samples.jsonl")

    samples_to_eval_path_for_json = ""
    if os.path.exists(sanitized_samples_jsonl_path) and os.path.getsize(sanitized_samples_jsonl_path) > 0:
        samples_to_eval_path_for_json = sanitized_samples_jsonl_path
    elif os.path.exists(raw_samples_jsonl_path) and os.path.getsize(raw_samples_jsonl_path) > 0:
        samples_to_eval_path_for_json = raw_samples_jsonl_path
    else:
        print(f"Warning: Neither raw nor sanitized samples.jsonl found for {model_name}. Skipping pass@k calculation for this model.")
        # Fill with None if no data found
        final_base_passk = {f"pass@{k}": None for k in ALL_PASS_K_VALUES}
        final_plus_passk = {f"pass@{k}": None for k in ALL_PASS_K_VALUES}
        overall_base_summary[model_name] = final_base_passk
        overall_plus_summary[model_name] = final_plus_passk
        continue

    eval_results_json_path = f"{samples_to_eval_path_for_json}.eval_results.json"

    if not os.path.exists(eval_results_json_path):
        print(f"Warning: Evaluation results JSON not found: {eval_results_json_path}. Skipping pass@k calculation for this model.")
        # Fill with None if no results found
        final_base_passk = {f"pass@{k}": None for k in ALL_PASS_K_VALUES}
        final_plus_passk = {f"pass@{k}": None for k in ALL_PASS_K_VALUES}
        overall_base_summary[model_name] = final_base_passk
        overall_plus_summary[model_name] = final_plus_passk
        continue

    # Load the full evaluation results from JSON
    with open(eval_results_json_path, 'r') as f:
        full_eval_results = json.load(f)

    # Initialize data structures for pass@k calculation
    total_samples_per_task = []
    base_correct_per_task = []
    plus_correct_per_task = []

    if "eval" in full_eval_results:
        for task_id, task_results in full_eval_results["eval"].items():
            num_samples_for_task = len(task_results)
            total_samples_per_task.append(num_samples_for_task)

            base_correct_count = 0
            plus_correct_count = 0
            for res in task_results:
                if res["base_status"] == PASS:
                    base_correct_count += 1
                if res["base_status"] == PASS and res["plus_status"] == PASS:
                    plus_correct_count += 1
            base_correct_per_task.append(base_correct_count)
            plus_correct_per_task.append(plus_correct_count)

    total_samples_per_task_np = np.array(total_samples_per_task)
    base_correct_per_task_np = np.array(base_correct_per_task)
    plus_correct_per_task_np = np.array(plus_correct_per_task)

    final_base_passk = {}
    for k_val in ALL_PASS_K_VALUES:
        if total_samples_per_task_np.min() >= k_val:
            pass_rate = estimate_pass_at_k(total_samples_per_task_np, base_correct_per_task_np, k_val).mean()
            final_base_passk[f"pass@{k_val}"] = pass_rate
        else:
            final_base_passk[f"pass@{k_val}"] = None # Not enough samples to calculate

    final_plus_passk = {}
    for k_val in ALL_PASS_K_VALUES:
        if (total_samples_per_task_np >= k_val).all(): # For plus, all tasks must have enough samples
            pass_rate = estimate_pass_at_k(total_samples_per_task_np, plus_correct_per_task_np, k_val).mean()
            final_plus_passk[f"pass@{k_val}"] = pass_rate
        else:
            final_plus_passk[f"pass@{k_val}"] = None # Not enough samples to calculate

    overall_base_summary[model_name] = final_base_passk
    overall_plus_summary[model_name] = final_plus_passk

# Save base pass@k results
base_passk_json_path = os.path.join(output_dir, "base_passk.json")
with open(base_passk_json_path, "w") as f:
    json.dump(overall_base_summary, f, indent=2, ensure_ascii=False)
print(f"✅ Base pass@k results saved to {base_passk_json_path}")

# Save plus pass@k results
plus_passk_json_path = os.path.join(output_dir, "plus_passk.json")
with open(plus_passk_json_path, "w") as f:
    json.dump(overall_plus_summary, f, indent=2, ensure_ascii=False)
print(f"✅ Plus pass@k results saved to {plus_passk_json_path}")

print(f"\n🎉 All tasks completed! Processed {len(Model_map)} models for pass@k summary.")
