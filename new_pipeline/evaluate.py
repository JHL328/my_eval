import os
import json
from datasets import load_dataset
import argparse
from model import Model_map, ModelQueue
import time
from result import concat_csvs, calc_passk
import datetime

def create_job_script(script_path, exp_name, log_path, command_args, time_limit="0:20:00"):
    script_content = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --job-name={exp_name}
#SBATCH --time={time_limit}
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH -o {log_path}%j_%x.out
#SBATCH -e {log_path}%j_%x.err

cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

export TRITON_CACHE_DIR="/tmp/triton-cache"

{command_args}
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
        f.flush()
        os.fsync(f.fileno())

def auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk):
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
                calc_passk(model_dir)
                with open(result_json_path, "w") as f:
                    f.write("done\n")
                print(f"[AUTO] Post-processing done for {model_name}")
            except Exception as e:
                print(f"[AUTO][ERROR] Post-processing failed for {model_name}: {e}")

def run_mmlu_flan_cot_fewshot_pass16(max_model=10, force=False):
    output_dir = "/mnt/sharefs/users/haolong.jia/result/mmlu_flan_pass16"
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
        dataset = load_dataset(
            "hails/mmlu_no_train",
            subject,
            cache_dir="/mnt/sharefs/users/haolong.jia/eval_data",
            trust_remote_code=True
        )
        n_total = len(dataset['test'])
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

    queue = ModelQueue(Model_map, output_dir, max_model=max_model, result_filename="result.json")
    # auto post-process all models before main loop
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
    last_status_time = time.time()
    status_interval = 60  # print status every 1 minute
    while queue.is_active():
        queue.update_finished()
        queue.print_status()
        auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)  # call every turn to ensure progress
        # print status every 1 minute
        if time.time() - last_status_time > status_interval:
            print(f"[MONITOR] {datetime.datetime.now().isoformat()} Current queue status:")
            queue.print_status()
            last_status_time = time.time()
        # submit new model
        while queue.can_submit():
            model_path, model_name = queue.submit_next()
            if model_path is None:
                break
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            # record model start time
            if model_name not in model_times:
                model_times[model_name] = {'start': datetime.datetime.now().isoformat()}
                with open(model_times_path, 'w') as f:
                    json.dump(model_times, f, indent=2)
            # submit job for each subject
            for subject in subjects:
                safe_subject = subject.replace(" ", "_")
                n_total, n_batches = subject_batches[subject]
                for i in range(n_batches):
                    idx_start = i * batch_size
                    idx_end = min((i + 1) * batch_size, n_total)
                    result_csv = os.path.join(model_dir, f"{safe_subject}_{idx_start}-{idx_end}.csv")
                    exp_name = f"mmlu-{model_name}-{safe_subject}-{idx_start}_{idx_end}"
                    script_path = os.path.join(scripts_dir, f"job_{exp_name}.sh")
                    log_path = logs_dir
                    # skip if result csv exists
                    if os.path.exists(result_csv):
                        continue
                    # skip if job script exists and force is not set
                    if os.path.exists(script_path) and not force:
                        print(f"Job script {script_path} exists, skipping (use --force to overwrite)")
                        continue
                    command_args = (
                        f"python {abs_eval_script} "
                        f"--model {model_path} --subject {subject} --idx_start {idx_start} --idx_end {idx_end} "
                        f"--results_dir {output_dir} --cot_prompts_path {cot_prompts_path}"
                    )
                    create_job_script(script_path, exp_name, log_path, command_args, time_limit="0:10:00")
                    # 等待脚本真正写入磁盘
                    for _ in range(10):
                        if os.path.exists(script_path):
                            break
                        time.sleep(0.1)
                    else:
                        print(f"[ERROR] Job script {script_path} not found after write!")
                        continue
                    # 新增：再次检查脚本是否存在
                    if not os.path.exists(script_path):
                        print(f"[ERROR] Script {script_path} disappeared before sbatch!")
                        continue
                    os.system(f"sbatch {script_path} for model {model_name}")
                    # print status every 1 minute
                    if time.time() - last_status_time > status_interval:
                        print(f"[MONITOR] {datetime.datetime.now().isoformat()} Current queue status:")
                        queue.print_status()
                        last_status_time = time.time()
        # check if all batch results are generated for running models
        for model_name in list(queue.running_models):
            model_dir = os.path.join(output_dir, model_name)
            try:
                if all_batch_results_exist(model_dir):
                    print(f"[INFO] All batch results exist for {model_name}. Proceeding, regardless of slurm job status.")
                    print(f"[INFO] All batch results exist for {model_name}, marking as complete and starting post-processing in background.")
                    # mark as complete immediately
                    queue.running_models.remove(model_name)
                    queue.completed.add(model_name)
                    queue.print_status()  # print status immediately
                    # post-process in background
                    try:
                        print(f"[INFO] All batch results exist for {model_name}, starting concat")
                        concat_csvs(model_dir)
                        print(f"[INFO] Concat done for {model_name}")
                        print(f"[INFO] All batch results exist for {model_name}, starting pass@k")
                        calc_passk(model_dir)
                        print(f"[INFO] Pass@k done for {model_name}")
                        # mark result.json as done
                        with open(os.path.join(model_dir, "result.json"), "w") as f:
                            f.write("done\n")
                        print(f"[INFO] result.json marked as done for {model_name}")
                        # record model end time
                        model_times[model_name]['end'] = datetime.datetime.now().isoformat()
                        with open(model_times_path, 'w') as f:
                            json.dump(model_times, f, indent=2)
                        queue.print_status()  # print status immediately
                    except Exception as e:
                        print(f"[ERROR] Post-processing failed for {model_name}: {e}")
            except Exception as e:
                print(f"[ERROR] Checking batch results for {model_name} failed: {e}")
                queue.running_models.remove(model_name)
                queue.fail.add(model_name)
                queue.print_status()
        queue.wait_for_slot()

def run_bbh_pass16(max_model=10, force=False):
    output_dir = "/mnt/sharefs/users/haolong.jia/result/bbh_pass16"
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
        dataset = load_dataset(
            "lukaemon/bbh",
            task,
            cache_dir="/mnt/sharefs/users/haolong.jia/eval_data",
            trust_remote_code=True
        )
        n_total = len(dataset['test'])
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

    queue = ModelQueue(Model_map, output_dir, max_model=max_model, result_filename="result.json")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
    last_status_time = time.time()
    status_interval = 60
    while queue.is_active():
        queue.update_finished()
        queue.print_status()
        auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)  # call every turn to ensure progress
        # print status every 1 minute
        if time.time() - last_status_time > status_interval:
            print(f"[MONITOR] {datetime.datetime.now().isoformat()} Current queue status:")
            queue.print_status()
            last_status_time = time.time()
        while queue.can_submit():
            model_path, model_name = queue.submit_next()
            if model_path is None:
                break
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            if model_name not in model_times:
                model_times[model_name] = {'start': datetime.datetime.now().isoformat()}
                with open(model_times_path, 'w') as f:
                    json.dump(model_times, f, indent=2)
            for task in tasks:
                n_total, n_batches = task_batches[task]
                for i in range(n_batches):
                    idx_start = i * batch_size
                    idx_end = min((i + 1) * batch_size, n_total)
                    result_csv = os.path.join(model_dir, f"{task}_{idx_start}-{idx_end}.csv")
                    exp_name = f"bbh-{model_name}-{task}-{idx_start}_{idx_end}"
                    script_path = os.path.join(scripts_dir, f"job_{exp_name}.sh")
                    log_path = logs_dir
                    if os.path.exists(result_csv):
                        continue
                    if os.path.exists(script_path) and not force:
                        print(f"Job script {script_path} exists, skipping (use --force to overwrite)")
                        continue
                    command_args = (
                        f"python {abs_eval_script} "
                        f"--model {model_path} --task {task} --idx_start {idx_start} --idx_end {idx_end} "
                        f"--cot_prompts_path {cot_prompts_path}"
                    )
                    create_job_script(script_path, exp_name, log_path, command_args, time_limit="0:10:00")
                    # 等待脚本真正写入磁盘
                    for _ in range(10):
                        if os.path.exists(script_path):
                            break
                        time.sleep(0.1)
                    else:
                        print(f"[ERROR] Job script {script_path} not found after write!")
                        continue
                    # 新增：再次检查脚本是否存在
                    if not os.path.exists(script_path):
                        print(f"[ERROR] Script {script_path} disappeared before sbatch!")
                        continue
                    os.system(f"sbatch {script_path} for model {model_name}")
                    # print status every 1 minute
                    if time.time() - last_status_time > status_interval:
                        print(f"[MONITOR] {datetime.datetime.now().isoformat()} Current queue status:")
                        queue.print_status()
                        last_status_time = time.time()
        for model_name in list(queue.running_models):
            model_dir = os.path.join(output_dir, model_name)
            try:
                if all_batch_results_exist(model_dir):
                    print(f"[INFO] All batch results exist for {model_name}. Proceeding, regardless of slurm job status.")
                    print(f"[INFO] All batch results exist for {model_name}, marking as complete and starting post-processing in background.")
                    queue.running_models.remove(model_name)
                    queue.completed.add(model_name)
                    queue.print_status()
                    try:
                        print(f"[INFO] All batch results exist for {model_name}, starting concat")
                        concat_csvs(model_dir)
                        print(f"[INFO] Concat done for {model_name}")
                        print(f"[INFO] All batch results exist for {model_name}, starting pass@k")
                        calc_passk(model_dir)
                        print(f"[INFO] Pass@k done for {model_name}")
                        with open(os.path.join(model_dir, "result.json"), "w") as f:
                            f.write("done\n")
                        print(f"[INFO] result.json marked as done for {model_name}")
                        model_times[model_name]['end'] = datetime.datetime.now().isoformat()
                        with open(model_times_path, 'w') as f:
                            json.dump(model_times, f, indent=2)
                        queue.print_status()
                    except Exception as e:
                        print(f"[ERROR] Post-processing failed for {model_name}: {e}")
            except Exception as e:
                print(f"[ERROR] Checking batch results for {model_name} failed: {e}")
                queue.running_models.remove(model_name)
                queue.fail.add(model_name)
                queue.print_status()
        queue.wait_for_slot()

def run_mmlu_pro_pass16(max_model=10, force=False):
    output_dir = "/mnt/sharefs/users/haolong.jia/result/mmlu_pro_pass16_newpipeline"
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

    # 获取所有subject
    with open(prompts_path, "r", encoding="utf-8") as f:
        subjects = list(json.load(f).keys())

    abs_eval_script = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/evaluate_mmlu_pro_pass16.py"
    batch_size = 500  # 可调整

    # 预缓存每个subject的样本数和batch数
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

    queue = ModelQueue(Model_map, output_dir, max_model=max_model, result_filename="result.json")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
    last_status_time = time.time()
    status_interval = 60
    while queue.is_active():
        queue.update_finished()
        queue.print_status()
        auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
        if time.time() - last_status_time > status_interval:
            print(f"[MONITOR] {datetime.datetime.now().isoformat()} Current queue status:")
            queue.print_status()
            last_status_time = time.time()
        while queue.can_submit():
            model_path, model_name = queue.submit_next()
            if model_path is None:
                break
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            if model_name not in model_times:
                model_times[model_name] = {'start': datetime.datetime.now().isoformat()}
                with open(model_times_path, 'w') as f:
                    json.dump(model_times, f, indent=2)
            for subject in subjects:
                safe_subject = subject.replace(" ", "_")
                n_total, n_batches = subject_batches[subject]
                for i in range(n_batches):
                    idx_start = i * batch_size
                    idx_end = min((i + 1) * batch_size, n_total)
                    result_csv = os.path.join(model_dir, f"{safe_subject}_{idx_start}-{idx_end}.csv")
                    legacy_result_csv = os.path.join(model_dir, f"{subject}_{idx_start}-{idx_end}.csv")
                    exp_name = f"mmlupro-{model_name}-{safe_subject}-{idx_start}_{idx_end}"
                    script_path = os.path.join(scripts_dir, f"job_{exp_name}.sh")
                    log_path = logs_dir
                    if os.path.exists(result_csv) or os.path.exists(legacy_result_csv):
                        continue
                    if os.path.exists(script_path) and not force:
                        print(f"Job script {script_path} exists, skipping (use --force to overwrite)")
                        continue
                    command_args = (
                        f"python {abs_eval_script} "
                        f"--model {model_path} --subject '{subject}' --idx_start {idx_start} --idx_end {idx_end} "
                        f"--prompts_path {prompts_path}"
                    )
                    create_job_script(script_path, exp_name, log_path, command_args, time_limit="0:20:00")
                    # 等待脚本真正写入磁盘
                    for _ in range(10):
                        if os.path.exists(script_path):
                            break
                        time.sleep(0.1)
                    else:
                        print(f"[ERROR] Job script {script_path} not found after write!")
                        continue
                    # 新增：再次检查脚本是否存在
                    if not os.path.exists(script_path):
                        print(f"[ERROR] Script {script_path} disappeared before sbatch!")
                        continue
                    os.system(f"sbatch {script_path} for model {model_name}")
                    if time.time() - last_status_time > status_interval:
                        print(f"[MONITOR] {datetime.datetime.now().isoformat()} Current queue status:")
                        queue.print_status()
                        last_status_time = time.time()
        for model_name in list(queue.running_models):
            model_dir = os.path.join(output_dir, model_name)
            try:
                if all_batch_results_exist(model_dir):
                    print(f"[INFO] All batch results exist for {model_name}. Proceeding, regardless of slurm job status.")
                    print(f"[INFO] All batch results exist for {model_name}, marking as complete and starting post-processing in background.")
                    queue.running_models.remove(model_name)
                    queue.completed.add(model_name)
                    queue.print_status()
                    try:
                        print(f"[INFO] All batch results exist for {model_name}, starting concat")
                        concat_csvs(model_dir)
                        print(f"[INFO] Concat done for {model_name}")
                        print(f"[INFO] All batch results exist for {model_name}, starting pass@k")
                        calc_passk(model_dir)
                        print(f"[INFO] Pass@k done for {model_name}")
                        with open(os.path.join(model_dir, "result.json"), "w") as f:
                            f.write("done\n")
                        print(f"[INFO] result.json marked as done for {model_name}")
                        model_times[model_name]['end'] = datetime.datetime.now().isoformat()
                        with open(model_times_path, 'w') as f:
                            json.dump(model_times, f, indent=2)
                        queue.print_status()
                    except Exception as e:
                        print(f"[ERROR] Post-processing failed for {model_name}: {e}")
            except Exception as e:
                print(f"[ERROR] Checking batch results for {model_name} failed: {e}")
                queue.running_models.remove(model_name)
                queue.fail.add(model_name)
                queue.print_status()
        queue.wait_for_slot()

def run_mmlu():
    print("[TODO]")
    pass

def run_mmlu(max_model=10, force=False):
    output_dir = "/mnt/sharefs/users/haolong.jia/result/mmlu"
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
        dataset = load_dataset(
            "hails/mmlu_no_train",
            subject,
            cache_dir="/mnt/sharefs/users/haolong.jia/eval_data",
            trust_remote_code=True
        )
        n_total = len(dataset['test'])
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

    queue = ModelQueue(Model_map, output_dir, max_model=max_model, result_filename="result.json")
    auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
    last_status_time = time.time()
    status_interval = 60
    while queue.is_active():
        queue.update_finished()
        queue.print_status()
        auto_postprocess_all_models(output_dir, all_batch_results_exist, concat_csvs, calc_passk)
        if time.time() - last_status_time > status_interval:
            print(f"[MONITOR] {datetime.datetime.now().isoformat()} Current queue status:")
            queue.print_status()
            last_status_time = time.time()
        while queue.can_submit():
            model_path, model_name = queue.submit_next()
            if model_path is None:
                break
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            if model_name not in model_times:
                model_times[model_name] = {'start': datetime.datetime.now().isoformat()}
                with open(model_times_path, 'w') as f:
                    json.dump(model_times, f, indent=2)
            for subject in subjects:
                n_total, n_batches = subject_batches[subject]
                for i in range(n_batches):
                    idx_start = i * batch_size
                    idx_end = min((i + 1) * batch_size, n_total)
                    result_csv = os.path.join(model_dir, f"{subject}_{idx_start}-{idx_end}.csv")
                    exp_name = f"mmlu-{model_name}-{subject}-{idx_start}_{idx_end}"
                    script_path = os.path.join(scripts_dir, f"job_{exp_name}.sh")
                    log_path = logs_dir
                    if os.path.exists(result_csv):
                        continue
                    if os.path.exists(script_path) and not force:
                        print(f"Job script {script_path} exists, skipping (use --force to overwrite)")
                        continue
                    command_args = (
                        f"python {abs_eval_script} "
                        f"--model {model_path} --subject '{subject}' --idx_start {idx_start} --idx_end {idx_end} "
                        f"--results_dir {output_dir} --prompts_path {prompts_path}"
                    )
                    create_job_script(script_path, exp_name, log_path, command_args, time_limit="0:10:00")
                    for _ in range(10):
                        if os.path.exists(script_path):
                            break
                        time.sleep(0.1)
                    else:
                        print(f"[ERROR] Job script {script_path} not found after write!")
                        continue
                    if not os.path.exists(script_path):
                        print(f"[ERROR] Script {script_path} disappeared before sbatch!")
                        continue
                    os.system(f"sbatch {script_path} for model {model_name}")
                    if time.time() - last_status_time > status_interval:
                        print(f"[MONITOR] {datetime.datetime.now().isoformat()} Current queue status:")
                        queue.print_status()
                        last_status_time = time.time()
        for model_name in list(queue.running_models):
            model_dir = os.path.join(output_dir, model_name)
            try:
                if all_batch_results_exist(model_dir):
                    print(f"[INFO] All batch results exist for {model_name}. Proceeding, regardless of slurm job status.")
                    print(f"[INFO] All batch results exist for {model_name}, marking as complete and starting post-processing in background.")
                    queue.running_models.remove(model_name)
                    queue.completed.add(model_name)
                    queue.print_status()
                    try:
                        print(f"[INFO] All batch results exist for {model_name}, starting concat")
                        concat_csvs(model_dir)
                        print(f"[INFO] Concat done for {model_name}")
                        print(f"[INFO] All batch results exist for {model_name}, starting pass@k")
                        calc_passk(model_dir)
                        print(f"[INFO] Pass@k done for {model_name}")
                        with open(os.path.join(model_dir, "result.json"), "w") as f:
                            f.write("done\n")
                        print(f"[INFO] result.json marked as done for {model_name}")
                        model_times[model_name]['end'] = datetime.datetime.now().isoformat()
                        with open(model_times_path, 'w') as f:
                            json.dump(model_times, f, indent=2)
                        queue.print_status()
                    except Exception as e:
                        print(f"[ERROR] Post-processing failed for {model_name}: {e}")
            except Exception as e:
                print(f"[ERROR] Checking batch results for {model_name} failed: {e}")
                queue.running_models.remove(model_name)
                queue.fail.add(model_name)
                queue.print_status()
        queue.wait_for_slot()

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
    args = parser.parse_args()

    # set default max_model for each task
    if args.max_model is None:
        if args.task == "mmlu_flan_cot_fewshot_pass16":
            args.max_model = 6
        elif args.task == "mmlu":
            args.max_model = 4
        elif args.task == "bbh_pass16":
            args.max_model = 16
        elif args.task == "mmlu_pro_pass16":
            args.max_model = 48
        else:
            args.max_model = 8

    TASKS[args.task](max_model=args.max_model, force=args.force)
