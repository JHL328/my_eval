import argparse
import csv
import json
import os
import random
import re
import subprocess
import time

import numpy as np
from vllm import SamplingParams

from aime_verifier import parse_assistant_output, score_answer
from hf_utils import get_llm, get_tokenizer
from model import model_map
THINK_TAG = "think"
PREFIX = (
    "You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. "
    "The reasoning process and answer are enclosed within <{THINK_TAG}> </{THINK_TAG}> and <answer> </answer> tags, respectively, i.e., <{THINK_TAG}> reasoning process here </{THINK_TAG}><answer> answer here </answer>. "
    "Provide a single number as the answer, for example, <answer> 47 </answer>. Now the user asks you to solve a math problem.\n\nUser: {quiz}\nAssistant:\n<{THINK_TAG}>\n"
)

# "Solve the following math problem step by step. $Answer (without quotes) where $Answer is the answer to the problem.\n\nUser: {quiz}\nAssistant:\n"

# "You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. "
# "The answer are enclosed within <answer> </answer> tags, respectively, i.e., <answer> answer here </answer>. "
# "Provide a single number as the answer, for example, <answer> 47 </answer>. Now the user asks you to solve a math problem, let's think step by step.\n\nUser: {quiz}\nAssistant:\n"

CHAT_PREFIX = (
    "{quiz}\nProvide a single number as the answer enclosed within the <answer> </answer> tags. For example, <answer> 47 </answer>."
)

TASK_CONFIGS = {
    "aime24": {
        "DATA_PATH": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/data/aime24/test.jsonl",
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/node/aime24_avg32",
    },
    "aime25": {
        "DATA_PATH": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/data/aime25/test.jsonl",
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/node/aime25_avg32",
    },
}

K_LIST = [1, 2, 4, 8, 16, 32]
N_SAMPLES = 32
MAX_MODEL_LEN = 16384
MAX_GEN_TOKENS = 15260
TEMPERATURE = 0.6
TOP_P = 0.95

SLURM_CONFIG = {
    "NODES": 1,
    "NTASKS": 1,
    "CPUS_PER_TASK": 96,
    "GPUS": 8,
    "MEM": "0",
    "TIME_LIMIT": "24:00:00",
    "PARTITION": "main",
    "CONDA_ENV": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval",
    "CD_PATH": os.path.dirname(os.path.abspath(__file__)),
}

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
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def parse_args():
    parser = argparse.ArgumentParser(description="AIME24/25 evaluation (Avg@32)")
    parser.add_argument("--task", type=str, default="aime24", choices=sorted(TASK_CONFIGS.keys()))
    parser.add_argument("--n_gpu", type=int, default=8)
    parser.add_argument("--apply_chat", action="store_true", default=True)
    parser.add_argument("--reforce", action="store_true",
                        help="If set, rerun evaluation even if result.csv exists")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional: run a single model path or model name from model_map")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Optional: override base output directory")
    return parser.parse_args()


def load_local_dataset(data_path):
    ids = []
    problems = []
    answers = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            problem = (obj.get("problem") or obj.get("question") or "").strip()
            answer = str(obj.get("answer", "")).strip()
            if not problem or not answer:
                continue
            ids.append(obj.get("id", len(ids)))
            problems.append(problem)
            answers.append(answer)
    return ids, problems, answers


def build_prompts(problems, tokenizer, apply_chat):
    if apply_chat:
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": CHAT_PREFIX.format(quiz=p.strip())}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in problems
        ]
    return [PREFIX.format(quiz=p.strip(), THINK_TAG=THINK_TAG) for p in problems]


def should_run_model(args, model_path, model_name):
    if args.model is None:
        return True
    return args.model == model_path or args.model == model_name


def is_job_done(model_out_dir):
    result_csv = os.path.join(model_out_dir, "result.csv")
    return os.path.exists(result_csv)


def submit_jobs_for_all_models(args, task_config, base_out):
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
        if not should_run_model(args, model_path, model_name):
            continue
        model_out_dir = os.path.join(base_out, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        if not args.reforce and is_job_done(model_out_dir):
            models_skipped.append(model_name)
            continue

        models_to_run.append(model_name)
        job_name = f"{args.task}_{model_name}"
        job_script = os.path.join(job_dir, f"job_{model_name}.sh")

        extra_flags = []
        if args.apply_chat:
            extra_flags.append("--apply_chat")
        if args.reforce:
            extra_flags.append("--reforce")
        extra_flags_str = " ".join(extra_flags)
        eval_cmd = f"""python3 -u {SLURM_CONFIG['CD_PATH']}/evaluate_aime_2.py \\
    --task {args.task} \\
    --n_gpu {args.n_gpu} \\
    --model "{model_path}" \\
    --results_dir "{base_out}" {extra_flags_str}"""

        with open(job_script, "w") as f:
            f.write(SBATCH_TEMPLATE.format(
                job_name=job_name,
                log_dir=log_dir,
                model_name=model_name,
                nodes=SLURM_CONFIG["NODES"],
                ntasks=SLURM_CONFIG["NTASKS"],
                cpus_per_task=SLURM_CONFIG["CPUS_PER_TASK"],
                gpus=args.n_gpu,
                mem=SLURM_CONFIG["MEM"],
                time_limit=SLURM_CONFIG["TIME_LIMIT"],
                partition=SLURM_CONFIG["PARTITION"],
                cd_path=SLURM_CONFIG["CD_PATH"],
                conda_env=SLURM_CONFIG["CONDA_ENV"],
                eval_cmd=eval_cmd,
            ))

        sbatch_commands.append((f"sbatch {job_script}", model_name))

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

    if sbatch_commands:
        print(f"\nSubmitting {len(sbatch_commands)} jobs...")
        for cmd, model_name in sbatch_commands:
            try:
                result = subprocess.check_output(cmd, shell=True, text=True)
                match = re.search(r"Submitted batch job (\d+)", result)
                if match:
                    job_id = match.group(1)
                    submitted_slurm_job_ids.append(job_id)
                    print(f"Successfully submitted job: {model_name} (Job ID: {job_id})")
                else:
                    print(f"Submitted job but could not parse job ID: {model_name}")
            except subprocess.CalledProcessError:
                print(f"Failed to submit job: {model_name}")
            time.sleep(0.2)

    print(
        f"\nAll model evaluation tasks have been submitted "
        f"(submitted {len(models_to_run)} models, skipped {len(models_skipped)} models)."
    )
    return submitted_slurm_job_ids


def wait_for_jobs_completion(submitted_slurm_job_ids):
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

            submitted_slurm_job_ids = [
                job_id for job_id in submitted_slurm_job_ids if job_id in running_jobs
            ]

            if submitted_slurm_job_ids:
                print(
                    f"📊 {len(submitted_slurm_job_ids)} jobs still running/pending. "
                    "Checking again in 60 seconds..."
                )
                time.sleep(60)
        except subprocess.CalledProcessError:
            print("All jobs appear to have completed.")
            break

    print("\n✅ All Slurm jobs have completed!")


def evaluate_model(model_path, model_name, task_config, args, ids, problems, answers, base_out):
    model_out_dir = os.path.join(base_out, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    result_csv = os.path.join(model_out_dir, "result.csv")
    if os.path.exists(result_csv) and not args.reforce:
        print(f"[skip] {model_name} already has result.csv")
        return

    tokenizer = get_tokenizer(model_path)
    prompts = build_prompts(problems, tokenizer, args.apply_chat)

    llm = get_llm(model_path, args.n_gpu, max_model_len_override=MAX_MODEL_LEN)
    sampling_params = SamplingParams(
        max_tokens=MAX_GEN_TOKENS,
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    if args.apply_chat:
        sampling_params.stop = (sampling_params.stop or []) + ["<|im_end|>"]

    print(f"Evaluating {model_name} with n_samples={N_SAMPLES}, max_tokens={MAX_GEN_TOKENS}")

    gens = llm.generate(prompts, sampling_params, use_tqdm=True)

    n_prompts = len(prompts)
    scores_matrix = np.zeros((n_prompts, N_SAMPLES), dtype=int)
    response_token_counts = np.zeros((n_prompts, N_SAMPLES), dtype=int)
    sample_records = []

    for i, (output, ground_truth, problem) in enumerate(zip(gens, answers, problems)):
        responses = []
        preds = []
        scores = []
        token_counts = []
        for j, single_output in enumerate(output.outputs):
            response_text = single_output.text
            _, pred = parse_assistant_output("<" + THINK_TAG + ">" + response_text, THINK_TAG=THINK_TAG)
            score = score_answer(pred, ground_truth)
            score_val = 1 if score == 1 else 0
            if j < N_SAMPLES:
                scores_matrix[i, j] = score_val
            responses.append(response_text)
            preds.append(pred)
            scores.append(score_val)
            n_tokens = len(tokenizer.encode(response_text, add_special_tokens=False))
            if j < N_SAMPLES:
                response_token_counts[i, j] = n_tokens
            token_counts.append(n_tokens)
            do_print = random.randint(1, N_SAMPLES) == 1
            if do_print:
                print("--------------------------------")
                print(f"PROMPT: {prompts[i]}")
                print(f"RESPONSE: {response_text}")
                print(f"PARSED ANSWER: {pred}")
                print(f"GROUND TRUTH: {ground_truth}")
                print(f"SCORE: {score_val}")
                print(f"RESPONSE TOKEN COUNT: {n_tokens}\n\n")

        sample_records.append({
            "id": ids[i],
            "problem": problem,
            "answer": ground_truth,
            "prompt": prompts[i],
            "responses": responses,
            "preds": preds,
            "scores": scores,
            "response_token_counts": token_counts,
        })

    print(f"Average response token count: {np.mean(response_token_counts)}")
    print(
        f"Fraction of responses with token count > {MAX_GEN_TOKENS - 5} "
        f"(max tokens is {MAX_GEN_TOKENS}): {np.mean(response_token_counts > MAX_GEN_TOKENS - 5)}"
    )

    with open(result_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(scores_matrix.tolist())

    sample_path = os.path.join(model_out_dir, "sample.jsonl")
    with open(sample_path, "w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_results(base_out):
    overall_summary = {}
    for model_path, model_name in model_map.items():
        model_dir = os.path.join(base_out, model_name)
        csv_path = os.path.join(model_dir, "result.csv")
        if not os.path.exists(csv_path):
            continue

        data = np.loadtxt(csv_path, delimiter=",")
        data = np.atleast_2d(data)
        total_correct = float(np.sum(data))
        total_attempts = float(data.size)
        avg_score = total_correct / total_attempts if total_attempts > 0 else 0.0

        passk_dict = {}
        for k in K_LIST:
            if k > data.shape[1]:
                continue
            pass_at_k_scores = []
            for row in data:
                n = len(row)
                c = int(np.sum(row))
                pass_at_k_scores.append(pass_at_k(n, c, k))
            passk_dict[f"pass@{k}"] = float(np.mean(pass_at_k_scores)) if pass_at_k_scores else 0.0

        metrics_path = os.path.join(model_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"exact_match: {avg_score:.4f}\n")
            f.write(f"avg@{N_SAMPLES}: {avg_score:.4f}\n")
            for k in K_LIST:
                if f"pass@{k}" in passk_dict:
                    f.write(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}\n")

        overall_summary[model_name] = {
            f"avg@{N_SAMPLES}": float(avg_score),
            "exact_match": float(avg_score),
            "total_questions": int(data.shape[0]),
            "total_samples": int(data.size),
            **passk_dict,
        }

    if not overall_summary:
        print("⚠️ No metrics were computed; result.json will not be written.")
        return

    result_json = os.path.join(base_out, "result.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved aggregated metrics to {result_json}")


def main():
    args = parse_args()
    task_config = TASK_CONFIGS[args.task]
    base_out = args.results_dir or task_config["BASE_OUT"]
    os.makedirs(base_out, exist_ok=True)

    if args.model is None:
        submitted_ids = submit_jobs_for_all_models(args, task_config, base_out)
        wait_for_jobs_completion(submitted_ids)
        summarize_results(base_out)
        return

    ids, problems, answers = load_local_dataset(task_config["DATA_PATH"])
    print(f"Loaded {len(problems)} problems from {task_config['DATA_PATH']}")

    for model_path, model_name in model_map.items():
        if not should_run_model(args, model_path, model_name):
            continue
        evaluate_model(model_path, model_name, task_config, args, ids, problems, answers, base_out)

    summarize_results(base_out)


if __name__ == "__main__":
    main()
