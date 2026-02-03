import os
import json
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
import argparse
from model import get_model_map_by_type, ModelQueue
import time
import datetime
import glob
import pandas as pd
import numpy as np
import fcntl

def _update_overall_passk_json_atomically(overall_passk_path, model_name, model_passk_results):
    os.makedirs(os.path.dirname(overall_passk_path), exist_ok=True)
    with open(overall_passk_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = {}
        all_results[model_name] = model_passk_results
        f.seek(0)
        f.truncate()
        json.dump(all_results, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)

def create_job_script(script_path, exp_name, log_path, command_args, time_limit="0:20:00"):
    script_content = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name={exp_name}
#SBATCH --time={time_limit}
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH -o {log_path}%j_%x.out
#SBATCH -e {log_path}%j_%x.err

cd /mnt/weka/home/haolong.jia/eval/RL-eval
# source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate base

export TRITON_CACHE_DIR="/tmp/triton-cache"

{command_args}
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
        f.flush()
        os.fsync(f.fileno())

def create_array_job_script(script_path, exp_name, log_path, tasks_file, num_tasks, time_limit="0:20:00"):
    """Create SLURM array job script for multiple tasks"""
    script_content = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name={exp_name}
#SBATCH --time={time_limit}
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH -o {log_path}%A_%a.out
#SBATCH -e {log_path}%A_%a.err
#SBATCH --array=1-{num_tasks}

cd /mnt/weka/home/haolong.jia/eval/RL-eval
# source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate base

export TRITON_CACHE_DIR="/tmp/triton-cache"

# Read task command from tasks file based on array task ID
TASK_CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {tasks_file})

if [[ -z "$TASK_CMD" ]]; then
    echo "Error: No task found for array task ID ${{SLURM_ARRAY_TASK_ID}}"
    exit 1
fi

echo "Running task ${{SLURM_ARRAY_TASK_ID}}: $TASK_CMD"

# Execute the task
eval "$TASK_CMD"
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
        f.flush()
        os.fsync(f.fileno())

def auto_postprocess_all_models(output_dir,  all_batch_results_exist, concat_csvs, calc_passk):
    for model_name in os.listdir(output_dir):
        model_dir = os.path.join(output_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        result_json_path = os.path.join(model_dir, "result.json")
        if os.path.exists(result_json_path):
            continue
        if all_batch_results_exist(model_dir):
            print(f"[AUTO] Found completed csvs for {model_name}, running post-processing.")
            try:
                concat_csvs(model_dir)
                calc_passk(model_dir, output_dir, model_name)
                with open(result_json_path, "w") as f:
                    f.write("done\n")
                print(f"[AUTO] Post-processing done for {model_name}")
            except Exception as e:
                print(f"[AUTO][ERROR] Post-processing failed for {model_name}: {e}")

def run_mmlu_flan_cot_fewshot_pass16(max_model=10, force=False, model_type="base"):
    if model_type == "sft":
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu_flan_pass16_sft"
    else:
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu_flan_pass16"
    cot_prompts_path = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_cot_prompts.json"
    scripts_dir = f'{output_dir}/job_scripts/'
    logs_dir = f'{output_dir}/logs/'
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_times = {}
    model_times_path = os.path.join(output_dir, 'model_eval_times.json')
    if os.path.exists(model_times_path):
        with open(model_times_path, 'r') as f:
            try:
                model_times = json.load(f)
            except Exception:
                model_times = {}

    with open(cot_prompts_path, "r", encoding="utf-8") as f:
        subjects = list(json.load(f).keys())

    abs_eval_script = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_mmlu_flan_cot_fewshot.py"
    batch_size = 200  

    # optimize: pre-cache n_total and n_batches for each subject
    subject_batches = {}
    for subject in subjects:
        dataset_test_split = load_dataset(
            "hails/mmlu_no_train",
            subject,
            split="test",
            cache_dir="/mnt/weka/shrd/k2m/haolong.jia/eval_data",
            trust_remote_code=True
        )
        n_total = len(dataset_test_split)
        n_batches = (n_total + batch_size - 1) // batch_size
        subject_batches[subject] = (n_total, n_batches)

    def all_batch_results_exist(model_dir):
        for subject in subjects:
            n_total, n_batches = subject_batches[subject]
            for i in range(n_batches):
                idx_start = i * batch_size
                idx_end = min((i + 1) * batch_size, n_total)
                result_csv = os.path.join(model_dir, f"{subject}_{idx_start}-{idx_end}.csv")
                if not os.path.exists(result_csv):
                    return False
        return True

    model_map = get_model_map_by_type(model_type)
    queue = ModelQueue(model_map, output_dir, max_model=max_model, result_filename="result.json")
    # auto post-process all models before main loop
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
    while queue.is_active():
        queue.update_finished()
        # Run post-processing after updating status
        auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
        queue.print_status()
        # submit new model
        while queue.can_submit():
            model_path, model_name = queue.submit_next()
            if model_path is None:
                break
            model_dir = os.path.join(output_dir, str(model_name))
            os.makedirs(model_dir, exist_ok=True)
            # record model start time
            if model_name not in model_times:
                model_times[model_name] = {'start': datetime.datetime.now().isoformat()}
                with open(model_times_path, 'w') as f:
                    json.dump(model_times, f, indent=2)
            # Collect all tasks for this model
            model_tasks = []
            for subject in subjects:
                safe_subject = subject.replace(" ", "_")
                n_total, n_batches = subject_batches[subject]
                for i in range(n_batches):
                    idx_start = i * batch_size
                    idx_end = min((i + 1) * batch_size, n_total)
                    result_csv = os.path.join(model_dir, f"{safe_subject}_{idx_start}-{idx_end}.csv")
                    if os.path.exists(result_csv):
                        continue
                    command_args = (
                        f"python {abs_eval_script} "
                        f"--model {model_path} --subject {subject} --idx_start {idx_start} --idx_end {idx_end} "
                        f"--type {model_type} "
                        f"--results_dir {output_dir} --cot_prompts_path {cot_prompts_path}"
                    )
                    model_tasks.append(command_args)
            
            # Submit as job array if there are tasks
            if model_tasks:
                exp_name = f"mmlu-cot-{model_name}"
                script_path = os.path.join(scripts_dir, f"array_job_{exp_name}.sh")
                tasks_file = os.path.join(scripts_dir, f"tasks_{exp_name}.txt")
                log_path = logs_dir
                
                # Skip if job script exists and force is not set
                if os.path.exists(script_path) and not force:
                    print(f"Array job script {script_path} exists, skipping (use --force to overwrite)")
                    continue
                
                # Write tasks to file
                with open(tasks_file, 'w') as f:
                    for task_cmd in model_tasks:
                        f.write(task_cmd + '\n')
                
                # Create array job script
                create_array_job_script(script_path, exp_name, log_path, tasks_file, len(model_tasks), time_limit="0:10:00")
                
                # Submit array job
                print(f"Submitting array job for {model_name} with {len(model_tasks)} tasks")
                os.system(f"sbatch {script_path}")
        queue.wait_for_slot()
    
    # Post-process all models after main loop ends
    print("[INFO] Main loop completed, running final post-processing...")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)

def run_bbh_pass16(max_model=10, force=False, model_type="base"):
    if model_type == "sft":
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/bbh_pass16_sft"
    else:
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/bbh_pass16"
    cot_prompts_path = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/bbh_cot_prompts.json"
    scripts_dir = f'{output_dir}/job_scripts/'
    logs_dir = f'{output_dir}/logs/'
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_times = {}
    model_times_path = os.path.join(output_dir, 'model_eval_times.json')
    if os.path.exists(model_times_path):
        with open(model_times_path, 'r') as f:
            try:
                model_times = json.load(f)
            except Exception:
                model_times = {}

    with open(cot_prompts_path, "r", encoding="utf-8") as f:
        tasks = list(json.load(f).keys())

    abs_eval_script = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_bbh_pass16.py"
    batch_size = 200  # changeable

    # pre-cache n_total and n_batches for each task
    task_batches = {}
    for task in tasks:
        dataset_test_split = load_dataset(
            "lukaemon/bbh",
            task,
            split="test",
            cache_dir="/mnt/weka/shrd/k2m/haolong.jia/eval_data",
            trust_remote_code=True
        )
        n_total = len(dataset_test_split)
        n_batches = (n_total + batch_size - 1) // batch_size
        task_batches[task] = (n_total, n_batches)

    def all_batch_results_exist(model_dir, max_missing=20):
        missing = 0
        missing_batches = []
        for task in tasks:
            n_total, n_batches = task_batches[task]
            for i in range(n_batches):
                idx_start = i * batch_size
                idx_end = min((i + 1) * batch_size, n_total)
                result_csv = os.path.join(model_dir, f"{task}_{idx_start}-{idx_end}.csv")
                if not os.path.exists(result_csv):
                    missing += 1
                    missing_batches.append((task, idx_start, idx_end))
                    if missing > max_missing:
                        return False
        if missing > 0:
            print(f"[WARN] {model_dir} missing {missing} batch(es): {missing_batches}")
        return True

    model_map = get_model_map_by_type(model_type)
    queue = ModelQueue(model_map, output_dir, max_model=max_model, result_filename="result.json")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
    while queue.is_active():
        queue.update_finished()
        # Run post-processing after updating status
        auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
        queue.print_status()
        while queue.can_submit():
            model_path, model_name = queue.submit_next()
            tp_size = 1  # fix to 1, we use 1 gpu per model
            if model_path is None:
                break
            model_dir = os.path.join(output_dir, str(model_name))
            os.makedirs(model_dir, exist_ok=True)
            if model_name not in model_times:
                model_times[model_name] = {'start': datetime.datetime.now().isoformat()}
                with open(model_times_path, 'w') as f:
                    json.dump(model_times, f, indent=2)
            # Collect all tasks for this model
            model_tasks = []
            for task in tasks:
                n_total, n_batches = task_batches[task]
                for i in range(n_batches):
                    idx_start = i * batch_size
                    idx_end = min((i + 1) * batch_size, n_total)
                    result_csv = os.path.join(model_dir, f"{task}_{idx_start}-{idx_end}.csv")
                    if os.path.exists(result_csv):
                        continue
                    command_args = (
                        f"python {abs_eval_script} "
                        f"--model {model_path} --task {task} --idx_start {idx_start} --idx_end {idx_end} "
                        f"--type {model_type} "
                        f"--cot_prompts_path {cot_prompts_path} --tp_size {tp_size}"
                    )
                    model_tasks.append(command_args)
            
            # Submit as job array if there are tasks
            if model_tasks:
                exp_name = f"bbh-{model_name}"
                script_path = os.path.join(scripts_dir, f"array_job_{exp_name}.sh")
                tasks_file = os.path.join(scripts_dir, f"tasks_{exp_name}.txt")
                log_path = logs_dir
                
                # Skip if job script exists and force is not set
                if os.path.exists(script_path) and not force:
                    print(f"Array job script {script_path} exists, skipping (use --force to overwrite)")
                    continue
                
                # Write tasks to file
                with open(tasks_file, 'w') as f:
                    for task_cmd in model_tasks:
                        f.write(task_cmd + '\n')
                
                # Create array job script
                create_array_job_script(script_path, exp_name, log_path, tasks_file, len(model_tasks), time_limit="0:10:00")
                
                # Submit array job
                print(f"Submitting array job for {model_name} with {len(model_tasks)} tasks")
                os.system(f"sbatch {script_path}")
        queue.wait_for_slot()
    
    # Post-process all models after main loop ends
    print("[INFO] Main loop completed, running final post-processing...")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)

def run_mmlu_pro_pass16(max_model=10, force=False, model_type="base"):
    if model_type == "sft":
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu_pro_pass16_sft"
    else:
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu_pro_pass16"
    prompts_path = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_pro_prompts.json"
    scripts_dir = f'{output_dir}/job_scripts/'
    logs_dir = f'{output_dir}/logs/'
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_times = {}
    model_times_path = os.path.join(output_dir, 'model_eval_times.json')
    if os.path.exists(model_times_path):
        with open(model_times_path, 'r') as f:
            try:
                model_times = json.load(f)
            except Exception:
                model_times = {}

    # get all subjects
    with open(prompts_path, "r", encoding="utf-8") as f:
        subjects = list(json.load(f).keys())

    abs_eval_script = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_mmlu_pro_pass16.py"
    batch_size = 500  # changeable

    # pre-cache n_total and n_batches for each subject
    import pandas as pd
    with open(prompts_path, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    subject_batches = {}
    for subject in subjects:
        n_total = len(all_prompts[subject])
        n_batches = (n_total + batch_size - 1) // batch_size
        subject_batches[subject] = (n_total, n_batches)

    def all_batch_results_exist(model_dir):
        for subject in subjects:
            safe_subject = subject.replace(" ", "_")
            n_total, n_batches = subject_batches[subject]
            for i in range(n_batches):
                idx_start = i * batch_size
                idx_end = min((i + 1) * batch_size, n_total)
                result_csv = os.path.join(model_dir, f"{safe_subject}_{idx_start}-{idx_end}.csv")
                legacy_result_csv = os.path.join(model_dir, f"{subject}_{idx_start}-{idx_end}.csv")
                if not (os.path.exists(result_csv) or os.path.exists(legacy_result_csv)):
                    return False
        return True

    model_map = get_model_map_by_type(model_type)
    queue = ModelQueue(model_map, output_dir, max_model=max_model, result_filename="result.json")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
    while queue.is_active():
        queue.update_finished()
        # Run post-processing after updating status
        auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
        queue.print_status()
        while queue.can_submit():
            model_path, model_name = queue.submit_next()
            if model_path is None:
                break
            model_dir = os.path.join(output_dir, str(model_name))
            os.makedirs(model_dir, exist_ok=True)
            if model_name not in model_times:
                model_times[model_name] = {'start': datetime.datetime.now().isoformat()}
                with open(model_times_path, 'w') as f:
                    json.dump(model_times, f, indent=2)
            # Collect all tasks for this model
            model_tasks = []
            for subject in subjects:
                safe_subject = subject.replace(" ", "_")
                n_total, n_batches = subject_batches[subject]
                for i in range(n_batches):
                    idx_start = i * batch_size
                    idx_end = min((i + 1) * batch_size, n_total)
                    result_csv = os.path.join(model_dir, f"{safe_subject}_{idx_start}-{idx_end}.csv")
                    legacy_result_csv = os.path.join(model_dir, f"{subject}_{idx_start}-{idx_end}.csv")
                    if os.path.exists(result_csv) or os.path.exists(legacy_result_csv):
                        continue
                    command_args = (
                        f"python {abs_eval_script} "
                        f"--model {model_path} --subject '{subject}' --idx_start {idx_start} --idx_end {idx_end} "
                        f"--type {model_type} "
                        f"--prompts_path {prompts_path}"
                    )
                    model_tasks.append(command_args)
            
            # Submit as job array if there are tasks
            if model_tasks:
                exp_name = f"mmlupro-{model_name}"
                script_path = os.path.join(scripts_dir, f"array_job_{exp_name}.sh")
                tasks_file = os.path.join(scripts_dir, f"tasks_{exp_name}.txt")
                log_path = logs_dir
                
                # Skip if job script exists and force is not set
                if os.path.exists(script_path) and not force:
                    print(f"Array job script {script_path} exists, skipping (use --force to overwrite)")
                    continue
                
                # Write tasks to file
                with open(tasks_file, 'w') as f:
                    for task_cmd in model_tasks:
                        f.write(task_cmd + '\n')
                
                # Create array job script
                create_array_job_script(script_path, exp_name, log_path, tasks_file, len(model_tasks), time_limit="0:20:00")
                
                # Submit array job
                print(f"Submitting array job for {model_name} with {len(model_tasks)} tasks")
                os.system(f"sbatch {script_path}")
        queue.wait_for_slot()
    
    # Post-process all models after main loop ends
    print("[INFO] Main loop completed, running final post-processing...")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)

def run_mmlu(max_model=10, force=False, model_type="base"):
    if model_type == "sft":
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu_sft"
    else:
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu"
    prompts_path = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_prompts.json"
    scripts_dir = f'{output_dir}/job_scripts/'
    logs_dir = f'{output_dir}/logs/'
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_times = {}
    model_times_path = os.path.join(output_dir, 'model_eval_times.json')
    if os.path.exists(model_times_path):
        with open(model_times_path, 'r') as f:
            try:
                model_times = json.load(f)
            except Exception:
                model_times = {}

    with open(prompts_path, "r", encoding="utf-8") as f:
        subjects = list(json.load(f).keys())

    abs_eval_script = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_mmlu.py"
    batch_size = 200

    subject_batches = {}
    for subject in subjects:
        dataset_test_split = load_dataset(
            "hails/mmlu_no_train",
            subject,
            split="test",
            cache_dir="/mnt/weka/shrd/k2m/haolong.jia/eval_data",
            trust_remote_code=True
        )
        n_total = len(dataset_test_split)
        n_batches = (n_total + batch_size - 1) // batch_size
        subject_batches[subject] = (n_total, n_batches)

    def all_batch_results_exist(model_dir):
        for subject in subjects:
            n_total, n_batches = subject_batches[subject]
            for i in range(n_batches):
                idx_start = i * batch_size
                idx_end = min((i + 1) * batch_size, n_total)
                result_csv = os.path.join(model_dir, f"{subject}_{idx_start}-{idx_end}.csv")
                if not os.path.exists(result_csv):
                    return False
        return True

    model_map = get_model_map_by_type(model_type)
    queue = ModelQueue(model_map, output_dir, max_model=max_model, result_filename="result.json")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
    while queue.is_active():
        queue.update_finished()
        # Run post-processing after updating status
        auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
        queue.print_status()
        while queue.can_submit():
            model_path, model_name = queue.submit_next()
            if model_path is None:
                break
            model_dir = os.path.join(output_dir, str(model_name))
            os.makedirs(model_dir, exist_ok=True)
            if model_name not in model_times:
                model_times[model_name] = {'start': datetime.datetime.now().isoformat()}
                with open(model_times_path, 'w') as f:
                    json.dump(model_times, f, indent=2)
            # Collect all tasks for this model
            model_tasks = []
            for subject in subjects:
                n_total, n_batches = subject_batches[subject]
                for i in range(n_batches):
                    idx_start = i * batch_size
                    idx_end = min((i + 1) * batch_size, n_total)
                    result_csv = os.path.join(model_dir, f"{subject}_{idx_start}-{idx_end}.csv")
                    if os.path.exists(result_csv):
                        continue
                    command_args = (
                        f"python {abs_eval_script} "
                        f"--model {model_path} --subject '{subject}' --idx_start {idx_start} --idx_end {idx_end} "
                        f"--type {model_type} "
                        f"--results_dir {output_dir} --prompts_path {prompts_path}"
                    )
                    model_tasks.append(command_args)
            
            # Submit as job array if there are tasks
            if model_tasks:
                exp_name = f"mmlu-{model_name}"
                script_path = os.path.join(scripts_dir, f"array_job_{exp_name}.sh")
                tasks_file = os.path.join(scripts_dir, f"tasks_{exp_name}.txt")
                log_path = logs_dir
                
                # Skip if job script exists and force is not set
                if os.path.exists(script_path) and not force:
                    print(f"Array job script {script_path} exists, skipping (use --force to overwrite)")
                    continue
                
                # Write tasks to file
                with open(tasks_file, 'w') as f:
                    for task_cmd in model_tasks:
                        f.write(task_cmd + '\n')
                
                # Create array job script
                create_array_job_script(script_path, exp_name, log_path, tasks_file, len(model_tasks), time_limit="0:10:00")
                
                # Submit array job
                print(f"Submitting array job for {model_name} with {len(model_tasks)} tasks")
                os.system(f"sbatch {script_path}")
        queue.wait_for_slot()
    
    # Post-process all models after main loop ends
    print("[INFO] Main loop completed, running final post-processing...")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)

def concat_csvs(model_dir):
    """
    concat all csv files in model_dir to result.csv
    """
    csv_files = sorted([
        f for f in glob.glob(os.path.join(model_dir, "*.csv"))
        if not f.endswith("result.csv")
    ])
    if not csv_files:
        print(f"[WARN] No csv files found in {model_dir}")
        return
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to read {f}: {e}")
    if not dfs:
        print(f"[WARN] No valid csv files to concat in {model_dir}")
        return
    result_df = pd.concat(dfs, ignore_index=True)
    result_path = os.path.join(model_dir, "result.csv")
    result_df.to_csv(result_path, index=False)
    print(f"[INFO] Saved merged result.csv to {result_path}")

def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def calc_passk(model_dir, output_root_dir, model_name, n_sampling=None):
    result_path = os.path.join(model_dir, "result.csv")
    if not os.path.exists(result_path):
        print(f"[WARN] result.csv not found in {model_dir}")
        return
    df = pd.read_csv(result_path)
    if n_sampling is None:
        sample_cols = [col for col in df.columns if col.isdigit() or col.startswith("em_")]
        n_sampling = len(sample_cols)
        if n_sampling == 0:
            print(f"[ERROR] Cannot infer n_sampling from columns: {df.columns}")
            return
    ks = [k for k in [1, 2, 4, 8, 16, 32] if k <= n_sampling]
    total_pass_at_k = {k: 0.0 for k in ks}
    for _, row in df.iterrows():
        if all(col.isdigit() for col in df.columns[:n_sampling]):
            correct_count = sum(int(row[str(i)]) for i in range(n_sampling))
        else:
            correct_count = sum(int(row[f'em_{i+1}']) for i in range(n_sampling))
        for k in ks:
            total_pass_at_k[k] += pass_at_k(n_sampling, correct_count, k)
    n_samples = len(df)
    passk_dict = {f'pass@{k}': (total_pass_at_k[k] / n_samples if n_samples > 0 else 0.0) for k in ks}
    overall_passk_path = os.path.join(output_root_dir, "passk.json")
    _update_overall_passk_json_atomically(overall_passk_path, model_name, passk_dict)
    print(f"[INFO] Saved pass@k metrics to {overall_passk_path} for {model_name}")

TASKS = {
    "mmlu_flan_cot_fewshot_pass16": run_mmlu_flan_cot_fewshot_pass16,
    "bbh_pass16": run_bbh_pass16,
    "mmlu_pro_pass16": run_mmlu_pro_pass16,
    "mmlu": run_mmlu,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=TASKS.keys())
    parser.add_argument("--max_model", type=int, default=None, help="Max number of models to evaluate at once")
    parser.add_argument("--force", action="store_true", help="Force resubmit jobs even if .sh exists")
    parser.add_argument("--type", type=str, default="base", choices=["base", "sft"], help="Model type: base or sft")
    args = parser.parse_args()

    # set default max_model for each task
    if args.max_model is None:
        if args.task == "mmlu_flan_cot_fewshot_pass16":
            args.max_model = 20
        elif args.task == "mmlu":
            args.max_model = 20
        elif args.task == "bbh_pass16":
            args.max_model = 20
        elif args.task == "mmlu_pro_pass16":
            args.max_model = 20
        else:
            args.max_model = 8

    TASKS[args.task](max_model=args.max_model, force=args.force, model_type=args.type)
