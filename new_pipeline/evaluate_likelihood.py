import os
import time
import sys
import argparse
import subprocess
import json
import re

# dynamically import Model_map
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model_map

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='drop', help='Task name for evaluation')
args = parser.parse_args()
task = args.task

output_dir = f"/mnt/weka/shrd/k2m/haolong.jia/result/{task}"
job_dir = os.path.join(output_dir, "job_scripts")
log_dir = os.path.join(output_dir, "logs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(job_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# prepare stage: initialize the lists
models_to_run = []
models_skipped = []
sbatch_commands = []
submitted_slurm_job_ids = []

# different task's main metric field
TASK_METRIC = {
    'drop': ('f1,none', 'drop'),
    'arc_easy': ('acc_norm,none', 'arc_easy'),
    'arc_challenge': ('acc_norm,none', 'arc_challenge'),
    'hellaswag': ('acc_norm,none', 'hellaswag'),
    'piqa': ('acc_norm,none', 'piqa'),
    'winogrande': ('acc_norm,none', 'winogrande'),
    'triviaqa': ('exact_match,remove_whitespace', 'triviaqa'),
    'nq_open': ('exact_match,remove_whitespace', 'nq_open'),
    'commonsense_qa': ('acc_norm,none', 'commonsense_qa'),
    "agieval": ("acc_norm,none", "agieval_en"),
    "openbookqa": ("acc_norm,none", "openbookqa"),
    "social_iqa": ("acc_norm,none", "social_iqa"),
    "truthfulqa_mc2": ("acc_norm,none", "truthfulqa_mc2"),
    "leaderboard_gpqa_diamond": ("acc_norm,none", "leaderboard_gpqa_diamond"),
}

# can adjust the resource parameters as needed
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
#SBATCH --mem=400G

cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

{lm_eval_cmd}

# post process logic
echo "⏳ now we post process {model_name} result..."

# find the intermediate subdirectory created by lm_eval 
SUBDIR=$(find {model_out_dir} -mindepth 1 -maxdepth 1 -type d | head -1)

if [ -n "$SUBDIR" ]; then
    # move and rename results_*.json
    RESULT_FILE=$(find "$SUBDIR" -name "results_*.json" | head -1)
    if [ -n "$RESULT_FILE" ]; then
        mv "$RESULT_FILE" {model_out_dir}/result.json
    fi
    
    # handle sample files
    if [ "{task}" = "agieval" ]; then
        # merge all samples_agieval_*.jsonl
        find "$SUBDIR" -name "samples_agieval_*.jsonl" -exec cat {{}} \\; > {model_out_dir}/sample.jsonl
        find "$SUBDIR" -name "samples_agieval_*.jsonl" -delete
    else
        SAMPLE_FILE=$(find "$SUBDIR" -name "samples_*.jsonl" | head -1)
        if [ -n "$SAMPLE_FILE" ]; then
            mv "$SAMPLE_FILE" {model_out_dir}/sample.jsonl
        fi
    fi
    
    # delete the intermediate subdirectory
    rm -rf "$SUBDIR"
fi

echo "evaluation and post-processing of {model_name} completed."
"""

for model_path, model_name in Model_map.items():
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
    if task == "social_iqa":
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,gpu_memory_utilization=0.95 \
  --tasks {task} \
  --output_path {output_dir}/{model_name} \
  --batch_size auto \
  --log_samples \
  --num_fewshot 0 \
  --trust_remote_code"""
    elif task == "arc_challenge":
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,gpu_memory_utilization=0.95 \
  --tasks {task} \
  --output_path {output_dir}/{model_name} \
  --batch_size auto \
  --log_samples \
  --num_fewshot 25 \
  --trust_remote_code"""
    elif task == "winogrande":
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,gpu_memory_utilization=0.95 \
  --tasks {task} \
  --output_path {output_dir}/{model_name} \
  --batch_size auto \
  --log_samples \
  --num_fewshot 5 \
  --trust_remote_code"""
    elif task == "triviaqa":
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,gpu_memory_utilization=0.95 \
  --tasks {task} \
  --output_path {output_dir}/{model_name} \
  --batch_size auto \
  --log_samples \
  --num_fewshot 5 \
  --trust_remote_code"""
    else:
        lm_eval_cmd = f"""lm_eval --model vllm \
  --model_args pretrained={model_path},tensor_parallel_size=1,gpu_memory_utilization=0.95 \
  --tasks {task} \
  --output_path {output_dir}/{model_name} \
  --batch_size auto \
  --log_samples \
  --num_fewshot 0"""
    
    # write the job script
    with open(job_script, "w") as f:
        f.write(SBATCH_TEMPLATE.format(
            model_name=model_name,
            model_path=model_path,
            model_out_dir=model_out_dir,
            output_dir=output_dir,
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
            # Use subprocess to capture the output
            result = subprocess.check_output(cmd, shell=True, text=True)
            # Extract job ID from output like "Submitted batch job 123456"
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
        # Check job status
        job_ids_str = ",".join(submitted_slurm_job_ids)
        try:
            # Query job status
            squeue_output = subprocess.check_output(
                f"squeue -h -j {job_ids_str} -o '%i %t'", 
                shell=True, 
                text=True
            )
            
            # Parse running/pending jobs
            running_jobs = set()
            for line in squeue_output.strip().split('\n'):
                if line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id = parts[0]
                        status = parts[1]
                        # R=running, PD=pending, CG=completing
                        if status in ['R', 'PD', 'CG']:
                            running_jobs.add(job_id)
            
            # Update the list to only include jobs that are still running/pending
            submitted_slurm_job_ids = [job_id for job_id in submitted_slurm_job_ids if job_id in running_jobs]
            
            if submitted_slurm_job_ids:
                print(f"📊 {len(submitted_slurm_job_ids)} jobs still running/pending. Checking again in 30 seconds...")
                time.sleep(30)
            
        except subprocess.CalledProcessError:
            # If squeue fails (e.g., all jobs completed), assume they're done
            print("All jobs appear to have completed.")
            break
    
    print("\n✅ All Slurm jobs have completed!")

# Generate overall result.json
print("\n📝 Generating overall result.json...")
metric_field, result_key = TASK_METRIC.get(task, ('f1,none', task))
overall_summary = {}

for model_path, model_name in Model_map.items():
    result_json_path = os.path.join(output_dir, model_name, "result.json")
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, "r") as f:
                data = json.load(f)
                if task == "agieval":
                    # Extract overall agieval score
                    metric = data["results"]["agieval"][metric_field]
                    overall_summary[model_name] = {metric_field: metric}
                else:
                    metric = data["results"][result_key][metric_field]
                    overall_summary[model_name] = {metric_field: metric}
        except Exception as e:
            print(f"Error reading result for {model_name}: {e}")

# Save overall result.json
overall_result_path = os.path.join(output_dir, "result.json")
with open(overall_result_path, "w") as f:
    json.dump(overall_summary, f, indent=2, ensure_ascii=False)

print(f"✅ Overall result.json has been saved to {overall_result_path}")
print(f"\n🎉 All tasks completed! Processed {len(overall_summary)} models.")  
