#!/usr/bin/env python3
"""
Simplified evaluate_code.py - only 'all' mode for GPU execution
All steps (generation, sanitize, evaluate) run on the same GPU node
"""

import os
import time
import sys
import argparse
import subprocess
import json
import re
import numpy as np
from evalplus.eval import estimate_pass_at_k, PASS

# Import optimization config
sys.path.insert(0, '/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline')
from optimize_config import get_sanitize_command, get_evaluate_command, get_optimal_config

# Import model list
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
try:
    from model import Model_map
except ImportError:
    sys.path.append(os.path.join(parent_dir, 'new_pipeline'))
    from model import Model_map

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Task name: humaneval or mbpp')
args = parser.parse_args()

task = args.task
output_dir = f"/mnt/sharefs/users/haolong.jia/result/{task}"
log_dir = f"{output_dir}/logs"
job_dir = f"/mnt/weka/home/haolong.jia/eval/runs/{task}_jobs"

# Ensure directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(job_dir, exist_ok=True)

# Task-specific settings
if task == "humaneval":
    N_SAMPLES = 64
    MAX_TOKENS = 1024
elif task == "mbpp":
    N_SAMPLES = 64
    MAX_TOKENS = 1024
else:
    print(f"Unknown task: {task}")
    sys.exit(1)

TEMPERATURE = 0.6
TP_SIZE = 1

# Simplified SBATCH template with optimized settings
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task}_{model_name}
#SBATCH --output={log_dir}/{model_name}.out
#SBATCH --error={log_dir}/{model_name}.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --mem=400G

cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate evalplus-eval

# Set task-specific cache directory
export TORCH_COMPILE_CACHE_DIR="/mnt/weka/home/haolong.jia/.cache/vllm/torch_compile_cache_{task}_{model_name}"
mkdir -p "$TORCH_COMPILE_CACHE_DIR"

# Set optimized parallel processing environment variables
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

echo "=========================================="
echo "Processing {model_name} on {task}"
echo "=========================================="

# Step 1: Generate code samples
echo "[Step 1] Generating code samples..."
python new_pipeline/generate_code.py \
  --model_path {model_path} \
  --dataset {task} \
  --n_samples {n_samples} \
  --temperature {temperature} \
  --tensor_parallel_size {tp_size} \
  --output_dir {raw_samples_dir} \
  --max_tokens {max_tokens}
  
if [ $? -ne 0 ]; then
    echo "Error: Code generation failed for {model_name}"
    exit 1
fi

# Step 2: Sanitize the generated samples (optimized)
echo "[Step 2] Sanitizing code samples (12 workers, batch_size=500)..."
python -m evalplus.sanitize_optimized \
  --samples {raw_samples_dir}/samples.jsonl \
  --n-workers 12 --batch-size 500

if [ $? -ne 0 ]; then
    echo "Error: Sanitization failed for {model_name}"
    exit 1
fi

# Determine which samples to evaluate
SAN_SAMPLES_JSONL="{raw_samples_dir}/samples-sanitized.jsonl"
if [ -f "$SAN_SAMPLES_JSONL" ] && [ -s "$SAN_SAMPLES_JSONL" ]; then
    SAMPLES_TO_EVAL="$SAN_SAMPLES_JSONL"
    echo "Using sanitized samples for evaluation"
else
    SAMPLES_TO_EVAL="{raw_samples_dir}/samples.jsonl"
    echo "Using raw samples for evaluation"
fi

# Step 3: Evaluate the samples
echo "[Step 3] Evaluating code samples..."
python -m evalplus.evaluate \
  --dataset {task} \
  --samples "$SAMPLES_TO_EVAL" \
  > {results_txt_path} 2>&1

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "All steps completed successfully for {model_name}"
    echo "=========================================="
else
    echo "Error: Evaluation failed for {model_name}"
    exit 1
fi
"""

def main():
    print(f"=" * 60)
    print(f"Code Evaluation Pipeline for {task}")
    print(f"Mode: All steps on GPU (optimized)")
    print(f"=" * 60)
    
    models_to_run = []
    models_skipped = []
    sbatch_commands = []
    
    # Process each model
    for model_path, model_name in Model_map.items():
        model_out_dir = os.path.join(output_dir, model_name)
        results_txt_path = os.path.join(model_out_dir, "results.txt")
        raw_samples_dir = os.path.join(model_out_dir, "raw_samples")
        
        # Skip if already completed
        if os.path.exists(results_txt_path):
            models_skipped.append(model_name)
            continue
            
        models_to_run.append(model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        os.makedirs(raw_samples_dir, exist_ok=True)
        
        # Create job script
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
            results_txt_path=results_txt_path
        )
        
        with open(job_script, "w") as f:
            f.write(sbatch_script_content)
        
        sbatch_commands.append((f"sbatch {job_script}", model_name))
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Models to run: {len(models_to_run)}")
    print(f"  Models skipped (already done): {len(models_skipped)}")
    
    if models_skipped:
        print(f"\n⏭️ Skipped models (already completed):")
        for model in models_skipped:
            print(f"  - {model}")
    
    if models_to_run:
        print(f"\n🚀 Models to process:")
        for model in models_to_run:
            print(f"  - {model}")
        
        # Submit jobs
        print(f"\n📤 Submitting {len(sbatch_commands)} jobs...")
        submitted_job_ids = []
        
        for cmd, model_name in sbatch_commands:
            try:
                result = subprocess.check_output(cmd, shell=True, text=True)
                match = re.search(r'Submitted batch job (\d+)', result)
                if match:
                    job_id = match.group(1)
                    submitted_job_ids.append(job_id)
                    print(f"  ✅ {model_name}: Job {job_id}")
                else:
                    print(f"  ⚠️ {model_name}: Submitted but couldn't parse job ID")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ {model_name}: Failed to submit - {e}")
            
            time.sleep(0.2)  # Small delay between submissions
        
        if submitted_job_ids:
            print(f"\n✅ Successfully submitted {len(submitted_job_ids)} jobs")
            print(f"Job IDs: {', '.join(submitted_job_ids)}")
            print(f"\nMonitor progress with:")
            print(f"  squeue -u $USER")
            print(f"  tail -f {log_dir}/<model_name>.out")
            
            # Wait for all jobs to complete
            print(f"\n⏳ Waiting for all {len(submitted_job_ids)} Slurm jobs to complete...")
            active_jobs = set(submitted_job_ids)
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
                                if status in ['R', 'PD', 'CG', 'CF']:  # Running, Pending, Completing, Configuring
                                    current_running_or_pending.add(job_id)
                    
                    active_jobs = current_running_or_pending
                    if active_jobs:
                        print(f"  📊 {len(active_jobs)} jobs still active: {sorted(list(active_jobs))}. Checking again in 30 seconds...")
                        time.sleep(30)
                    else:
                        print("  All submitted Slurm jobs have finished processing according to squeue.")
                        break 
                except subprocess.CalledProcessError as e:
                    if "Invalid job id specified" in e.output or "No jobs in partition" in e.output:
                        print("  squeue reported no active jobs (or invalid job ID), assuming completion.")
                        active_jobs.clear()
                    else:
                        print(f"  Error querying squeue: {e.output}. Will retry in 30 seconds...")
                    time.sleep(30)
            
            print("\n✅ All Slurm jobs have completed!")
            
            # Automatically generate pass@k summary after all jobs complete
            print(f"\n{'='*60}")
            # Update all models to ensure consistency, even for skipped ones
            generate_passk_summary(task, models_to_update=None)
        else:
            print("\n✅ All models already completed!")
            # Even if nothing ran, ensure summary is up to date for all models
            generate_passk_summary(task, models_to_update=None)
    else:
        print("\n✅ All models already completed!")
        # Even if nothing ran, ensure summary is up to date for all models
        generate_passk_summary(task, models_to_update=None)
    
    print(f"\n{'='*60}")

def generate_passk_summary(task, models_to_update=None):
    """Generate overall pass@k JSON files for all models
    
    Args:
        task: Task name (humaneval or mbpp)
        models_to_update: Optional list of model names to update. If None, update all models.
                         If provided, only update these models and preserve others.
    """
    output_dir = f"/mnt/sharefs/users/haolong.jia/result/{task}"
    
    print(f"\n📝 Generating overall pass@k JSON files for {task}...")
    
    # Load existing summaries if they exist
    base_path = os.path.join(output_dir, "base_passk.json")
    plus_path = os.path.join(output_dir, "plus_passk.json")
    
    if os.path.exists(base_path):
        with open(base_path, 'r') as f:
            overall_base_summary = json.load(f)
        print(f"  📖 Loaded existing base pass@k from {base_path}")
    else:
        overall_base_summary = {}
    
    if os.path.exists(plus_path):
        with open(plus_path, 'r') as f:
            overall_plus_summary = json.load(f)
        print(f"  📖 Loaded existing plus pass@k from {plus_path}")
    else:
        overall_plus_summary = {}
    
    ALL_PASS_K_VALUES = [1, 8, 16, 32, 64]
    
    # Determine which models to process
    if models_to_update is None:
        models_to_process = [model_name for _, model_name in Model_map.items()]
    else:
        models_to_process = models_to_update
    
    print(f"  🔄 Processing {len(models_to_process)} models...")
    
    for model_name in models_to_process:
        model_out_dir = os.path.join(output_dir, model_name)
        raw_samples_dir = os.path.join(model_out_dir, "raw_samples")
        
        # Find the samples file used for evaluation
        sanitized_samples_path = os.path.join(raw_samples_dir, "samples-sanitized.jsonl")
        raw_samples_path = os.path.join(raw_samples_dir, "samples.jsonl")
        
        if os.path.exists(sanitized_samples_path) and os.path.getsize(sanitized_samples_path) > 0:
            samples_path = sanitized_samples_path
        elif os.path.exists(raw_samples_path) and os.path.getsize(raw_samples_path) > 0:
            samples_path = raw_samples_path
        else:
            print(f"  ⚠️ No samples found for {model_name}, skipping...")
            for k in ALL_PASS_K_VALUES:
                overall_base_summary.setdefault(model_name, {})[f"pass@{k}"] = None
                overall_plus_summary.setdefault(model_name, {})[f"pass@{k}"] = None
            continue
        
        # Find eval results JSON - try multiple naming conventions
        possible_eval_paths = [
            f"{samples_path}.eval_results.json",
            samples_path.replace(".jsonl", ".eval_results.json"),
            samples_path.replace(".jsonl", "_eval_results.json"),
            os.path.join(raw_samples_dir, "samples.eval_results.json"),
            os.path.join(raw_samples_dir, "samples_eval_results.json"),
            os.path.join(raw_samples_dir, "samples-sanitized.eval_results.json"),
        ]
        
        eval_results_path = None
        for path in possible_eval_paths:
            if os.path.exists(path):
                eval_results_path = path
                break
        
        if not eval_results_path:
            print(f"  ⚠️ No eval results for {model_name}, skipping...")
            for k in ALL_PASS_K_VALUES:
                overall_base_summary.setdefault(model_name, {})[f"pass@{k}"] = None
                overall_plus_summary.setdefault(model_name, {})[f"pass@{k}"] = None
            continue
        
        # Load and process results
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
        
        # Calculate pass@k
        total_samples = []
        base_correct = []
        plus_correct = []
        
        if "eval" in results and results["eval"]:
            for _, task_results in results["eval"].items():
                n_samples = len(task_results)
                total_samples.append(n_samples)
                
                base_count = 0
                plus_count = 0
                for r in task_results:
                    # Handle different result formats
                    if isinstance(r, dict):
                        if "base_status" in r:
                            # Format with base_status and plus_status
                            if r["base_status"] == PASS:
                                base_count += 1
                                if r.get("plus_status") == PASS:
                                    plus_count += 1
                        elif "base" in r:
                            # Format with base and plus as lists
                            if isinstance(r["base"], list) and all(r["base"]):
                                base_count += 1
                                if isinstance(r.get("plus"), list) and all(r["plus"]):
                                    plus_count += 1
                
                base_correct.append(base_count)
                plus_correct.append(plus_count)
        
        # Check if we have valid data
        if not total_samples:
            print(f"  ⚠️ No evaluation data found for {model_name}")
            for k in ALL_PASS_K_VALUES:
                overall_base_summary.setdefault(model_name, {})[f"pass@{k}"] = None
                overall_plus_summary.setdefault(model_name, {})[f"pass@{k}"] = None
            continue
        
        # Calculate pass@k metrics
        total_samples = np.array(total_samples)
        base_correct = np.array(base_correct)
        plus_correct = np.array(plus_correct)
        
        model_base_passk = {}
        model_plus_passk = {}
        
        for k in ALL_PASS_K_VALUES:
            if len(total_samples) > 0 and total_samples.min() >= k:
                base_passk = estimate_pass_at_k(total_samples, base_correct, k).mean()
                plus_passk = estimate_pass_at_k(total_samples, plus_correct, k).mean()
                model_base_passk[f"pass@{k}"] = float(base_passk)
                model_plus_passk[f"pass@{k}"] = float(plus_passk)
            else:
                model_base_passk[f"pass@{k}"] = None
                model_plus_passk[f"pass@{k}"] = None
        
        overall_base_summary[model_name] = model_base_passk
        overall_plus_summary[model_name] = model_plus_passk
        pass1_val = model_base_passk.get('pass@1')
        if pass1_val is not None:
            print(f"  ✅ {model_name}: base pass@1={pass1_val:.2%}")
        else:
            print(f"  ✅ {model_name}: base pass@1=N/A")
    
    # Save results
    base_path = os.path.join(output_dir, "base_passk.json")
    plus_path = os.path.join(output_dir, "plus_passk.json")
    
    with open(base_path, "w") as f:
        json.dump(overall_base_summary, f, indent=2)
    print(f"\n✅ Base pass@k saved to: {base_path}")
    
    with open(plus_path, "w") as f:
        json.dump(overall_plus_summary, f, indent=2)
    print(f"✅ Plus pass@k saved to: {plus_path}")
    
    print(f"\n🎉 Pass@k summary complete for {len(overall_base_summary)} models!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-summary":
        # Allow running just the summary generation
        if len(sys.argv) > 2:
            # When run manually, update all models
            generate_passk_summary(sys.argv[2], models_to_update=None)
        else:
            print("Usage: python evaluate_code_simple.py --generate-summary <task>")
    else:
        main()