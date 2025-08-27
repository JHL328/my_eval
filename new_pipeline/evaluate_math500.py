#!/usr/bin/env python3
"""
Math500 Evaluation Script with math_verify Integration
This script evaluates models on Math500 dataset using math_verify for answer comparison.
It maintains the same generation settings as the original math_eval.py.
"""

import os
import json
import re
import sys
import csv
import fcntl
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import subprocess
from math_verify import parse, verify
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Model_map

# =====================
# Math500 Examples (4-shot)
# =====================
MATH500_EXAMPLES = [
    {
        "question": "What is $\\frac{2^2 \\cdot 2^{-3}}{2^3 \\cdot 2^{-2}}$?",
        "answer": "We compute that \\[\\frac{2^2 \\cdot 2^{-3}}{2^3 \\cdot 2^{-2}} = \\frac{2^{2 - 3}}{2^{3 - 2}} = \\frac{2^{-1}}{2^1} = 2^{-1 - 1} = 2^{-2} = \\frac{1}{2^2} = \\boxed{\\frac{1}{4}}.\\] The Answer is \\frac{1}{4}"
    },
    {
        "question": "What is the value of $\\dfrac{3 \\times 4}{6}?$",
        "answer": "Calculating the numerator first, $\\dfrac{3 \\times 4}{6} = \\dfrac{12}{6} = \\boxed{2}$. The Answer is 2"
    },
    {
        "question": "How many positive integers less than $101$ are multiples of either $5$ or $7$, but not both at once?",
        "answer": "There are $20$ positive multiples of $5$ less than $101$. There are $14$ positive multiples of $7$ less than $101$. However, the least common multiple of $5$ and $7$ is $35$, and there are $2$ positive multiples of $35$ less than $101$. This means there are $20 - 2 = 18$ multiples of $5$ that aren't multiples of $7$, and $14 - 2 = 12$ multiples of 7 that aren't multiples of $5$, for a total of $18 + 12 = \\boxed{30}$. The Answer is 30"
    },
    {
        "question": "How many digits does the smallest repeating block in the decimal expansion of $\\frac{5}{7}$ contain?",
        "answer": "We use long division to find that the decimal representation of $\\frac{5}{7}$ is $0.\\overline{714285}$, which is a repeating block of $\\boxed{6}$ digits. The Answer is 6"
    }
]

# =====================
# Task Configuration
# =====================
TASK_CONFIG = {
    "BASE_OUT": "/mnt/sharefs/users/haolong.jia/result-rewrite/math500_pass64",
    "DATA_PATH": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/data/math500/test.jsonl",
    "DATA_NAME": "math500",
    "K_LIST": [1, 2, 4, 8, 16, 32, 64],
    "N_SAMPLING": 64,
    "NUM_SHOTS": 4,
    "PROMPT_TYPE": "cot",
    "GPUS_PER_TASK": 1,
    "TIME_LIMIT": "12:00:00",
    "PARTITION": "lowprio",
    "QOS": "lowprio",
    "MEM": "400G",
    "CONDA_ACTIVATE_PATH": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval",
    "CD_PATH_IN_JOB_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline",
}

SAMPLING_PARAMS = dict(
    temperature=0.6,
    top_p=0.95,
    n=64,
    max_tokens=4096,
    stop=["Q:", "</s>", "<|im_end|>", "\n\nQ:", "\n\nHuman:", "\n\nAssistant:", "Human:", "Assistant:"],
    seed=42,
)

# =====================
# Utility Functions
# =====================
def generate_fewshot_prompt(fewshot_examples=None):
    """Generate few-shot prompt from examples."""
    if not fewshot_examples:
        fewshot_examples = MATH500_EXAMPLES
    prompt = ""
    for ex in fewshot_examples:
        prompt += f"Question: {ex['question']}\nSolution: {ex['answer']}\n\n"
    return prompt

def parse_answer_with_verify(text):
    """Parse answer using math_verify library with fallback to regex patterns."""
    if text is None:
        return None
    
    try:
        # First try to parse the entire text with math_verify
        parsed = parse(text)
        if parsed is not None:
            return parsed
    except Exception:
        pass
    
    # Look for boxed answers (common in MATH dataset)
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    boxed_matches = re.findall(boxed_pattern, text)
    if boxed_matches:
        # Take the last boxed answer
        ans = boxed_matches[-1]
        try:
            return parse(ans)
        except:
            return ans
    
    # Look for "The answer is" patterns
    answer_patterns = [
        r"The answer is:?\s*\$?([^\$\n]+?)(?:\$|\.|\n|$)",
        r"The Answer is:?\s*\$?([^\$\n]+?)(?:\$|\.|\n|$)",
        r"answer is:?\s*\$?([^\$\n]+?)(?:\$|\.|\n|$)",
        r"Therefore,?\s*(?:the answer is:?)?\s*\$?([^\$\n]+?)(?:\$|\.|\n|$)",
        r"Thus,?\s*(?:the answer is:?)?\s*\$?([^\$\n]+?)(?:\$|\.|\n|$)",
        r"Hence,?\s*(?:the answer is:?)?\s*\$?([^\$\n]+?)(?:\$|\.|\n|$)",
        r"So,?\s*(?:the answer is:?)?\s*\$?([^\$\n]+?)(?:\$|\.|\n|$)",
    ]
    
    for pat in answer_patterns:
        matches = re.findall(pat, text, re.IGNORECASE | re.DOTALL)
        if matches:
            # Take the last match
            ans = matches[-1].strip()
            try:
                return parse(ans)
            except:
                return ans
    
    # Look for final mathematical expressions
    math_expr_pattern = r"\$([^\$]+)\$"
    math_matches = re.findall(math_expr_pattern, text)
    if math_matches:
        # Take the last mathematical expression
        ans = math_matches[-1]
        try:
            return parse(ans)
        except:
            return ans
    
    # Last resort: try to parse the entire text after cleanup
    try:
        # Remove common prefixes and suffixes
        cleaned_text = text.strip()
        for prefix in ["The answer is", "The Answer is", "answer is", "Therefore", "Thus", "Hence", "So"]:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].strip()
                if cleaned_text.startswith(":"):
                    cleaned_text = cleaned_text[1:].strip()
                break
        
        # Remove trailing periods
        cleaned_text = cleaned_text.rstrip(".")
        
        if cleaned_text:
            parsed = parse(cleaned_text)
            if parsed is not None:
                return parsed
    except:
        pass
    
    return None

def compare_answers(gold, pred):
    """Compare two answers using math_verify with fallback to string comparison."""
    try:
        # If both are parsed objects from math_verify, use verify
        if gold is not None and pred is not None:
            return verify(gold, pred)
    except Exception:
        pass
    
    # Fallback to string comparison if math_verify fails
    if gold is None or pred is None:
        return False
    
    # Convert to string for comparison if needed
    gold_str = str(gold) if gold is not None else ""
    pred_str = str(pred) if pred is not None else ""
    
    # Clean up for comparison
    gold_str = gold_str.replace(" ", "").replace(",", "").strip()
    pred_str = pred_str.replace(" ", "").replace(",", "").strip()
    
    # Handle fraction comparison
    if "/" in gold_str or "/" in pred_str:
        try:
            # Try to evaluate as fractions
            from fractions import Fraction
            gold_frac = Fraction(gold_str.replace("\\frac", "").replace("{", "").replace("}", "/"))
            pred_frac = Fraction(pred_str.replace("\\frac", "").replace("{", "").replace("}", "/"))
            return gold_frac == pred_frac
        except:
            pass
    
    return gold_str == pred_str

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

def is_job_running_or_done(model_out_dir):
    """Check if job is already running or completed."""
    result_csv = os.path.join(model_out_dir, "result.csv")
    return os.path.exists(result_csv)

def update_passk_json(passk_path, model_name, passk_result, overwrite=False):
    """Update pass@k results JSON file."""
    with open(passk_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        if overwrite:
            all_results = {}
        else:
            try:
                all_results = json.load(f)
            except Exception:
                all_results = {}
        all_results[model_name] = passk_result
        f.seek(0)
        f.truncate()
        json.dump(all_results, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)

# =====================
# Main Evaluation Function
# =====================
def run_single_model_evaluation(data_path, output_dir, n_sampling, model_path, model_name, base_out, overwrite, task_config):
    """Run evaluation for a single model on Math500 dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, dtype="auto", tensor_parallel_size=1)
    print("Model loaded successfully!")
    
    # Load data
    print(f"Loading Math500 data from: {data_path}")
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Prepare prompts
    print("Preparing prompts...")
    fewshot_prompt = generate_fewshot_prompt()
    prompts, golds, questions = [], [], []
    
    for item in tqdm(data, desc="Preparing data"):
        q = item.get("problem", item.get("question", ""))
        gold = item.get("solution", item.get("answer", ""))
        
        # Parse gold answer
        gold_parsed = parse_answer_with_verify(gold)
        
        # Create prompt
        prompt = fewshot_prompt + f"Question: {q}\nSolution: Let's think step by step."
        
        prompts.append(prompt)
        golds.append(gold_parsed)
        questions.append(q)
    
    # Print first prompt for debugging
    if len(prompts) > 0:
        print("\n" + "="*80)
        print("🔍 FIRST PROMPT FOR DEBUGGING:")
        print("="*80)
        print(prompts[0])
        print("="*80)
        print(f"First prompt length: {len(prompts[0])} characters")
        print("="*80 + "\n")
    
    # Run inference
    print(f"Running inference on {len(prompts)} prompts with {n_sampling} samples each...")
    params = SamplingParams(**{**SAMPLING_PARAMS, "n": n_sampling})
    outputs = llm.generate(prompts, params)
    
    # Process results
    print("Processing results...")
    results = []
    all_scores = []
    sample_jsonl = []
    
    for idx, (q, gold, output) in enumerate(tqdm(zip(questions, golds, outputs), total=len(questions), desc="Processing results")):
        generations = [output.outputs[i].text for i in range(n_sampling)]
        parsed = [parse_answer_with_verify(gen) for gen in generations]
        scores = [compare_answers(gold, p) for p in parsed]
        
        # Convert parsed results to strings for JSON serialization
        parsed_str = []
        for p in parsed:
            if p is None:
                parsed_str.append("")
            elif isinstance(p, (list, tuple)):
                # If it's a list or tuple from math_verify, convert each element
                parsed_str.append(str(p[0]) if len(p) > 0 else "")
            else:
                parsed_str.append(str(p))
        
        # Convert gold to string as well
        gold_str = ""
        if gold is not None:
            if isinstance(gold, (list, tuple)):
                gold_str = str(gold[0]) if len(gold) > 0 else ""
            else:
                gold_str = str(gold)
        
        # Store results
        results.append({
            "question": q,
            "gold": gold_str,
            "generations": generations,
            "parsed": parsed_str,
            "scores": scores,
            "pass@k": any(scores)
        })
        
        # For sample.jsonl format
        sample_jsonl.append({
            "idx": idx,
            "problem": q,
            "solution": gold_str,
            "pred": generations,
            "score": scores
        })
        
        all_scores.append(scores)
        
        # Debug info for the first sample
        if idx == 0:
            print("\n==== First Sample Debug Info ====")
            print(f"Question: {q[:100]}...")
            print(f"Gold: {gold_str}")
            print(f"First generation: {generations[0][:200]}...")
            print(f"Parsed: {parsed_str[0]}")
            print(f"Score: {scores[0]}")
            print("===============================\n")
    
    # Save sample.jsonl
    sample_jsonl_path = os.path.join(output_dir, "sample.jsonl")
    with open(sample_jsonl_path, 'w') as f:
        for item in sample_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save result.csv (binary score matrix)
    csv_path = os.path.join(output_dir, "result.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for scores in all_scores:
            # Convert boolean to int
            row = [1 if s else 0 for s in scores]
            writer.writerow(row)
    
    # Calculate metrics
    passk_dict = {}
    for k in task_config["K_LIST"]:
        pass_at_k_scores = []
        for scores in all_scores:
            n = len(scores)
            c = sum(scores)
            sample_pass_k = pass_at_k(n, c, k)
            pass_at_k_scores.append(sample_pass_k)
        passk_dict[f"pass@{k}"] = float(np.mean(pass_at_k_scores))
    
    # Calculate exact match rate
    total_correct = sum(sum(scores) for scores in all_scores)
    total_attempts = sum(len(scores) for scores in all_scores)
    exact_match = total_correct / total_attempts if total_attempts > 0 else 0.0
    
    # Save metrics.txt
    metrics_txt_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        f.write(f"exact_match: {exact_match:.4f}\n")
        for k in task_config["K_LIST"]:
            f.write(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}\n")
    
    # Save detailed results
    results_json_path = os.path.join(output_dir, "math500_eval_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Update global pass@k JSON
    if model_name is not None and base_out is not None:
        passk_path = os.path.join(base_out, "passk.json")
        update_passk_json(passk_path, model_name, passk_dict, overwrite=overwrite)
    
    print(f"\n✅ Evaluation completed for {model_name}")
    print(f"Exact match: {exact_match:.4f}")
    for k in [1, 4, 16, 64]:
        if f"pass@{k}" in passk_dict:
            print(f"Pass@{k}: {passk_dict[f'pass@{k}']:.4f}")

# =====================
# Submit jobs for all models
# =====================
def submit_jobs_for_all_models(args, task_config):
    """Submit SLURM jobs for all models."""
    os.makedirs(task_config["BASE_OUT"], exist_ok=True)
    models_to_run = []
    models_skipped = []
    sbatch_commands = []
    submitted_slurm_job_ids = []
    
    for model_path, model_name in Model_map.items():
        model_out_dir = os.path.join(task_config["BASE_OUT"], model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        if not args.reforce and is_job_running_or_done(model_out_dir):
            models_skipped.append(model_name)
            continue
        
        models_to_run.append(model_name)
        job_name = f"math500_{model_name}"
        job_script = os.path.join(model_out_dir, f"{job_name}.sh")
        
        with open(job_script, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={model_out_dir}/slurm.out
#SBATCH --error={model_out_dir}/slurm.err
#SBATCH --gres=gpu:{task_config['GPUS_PER_TASK']}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time={task_config['TIME_LIMIT']}
#SBATCH --partition={task_config['PARTITION']}
#SBATCH --qos={task_config['QOS']}
#SBATCH --mem={task_config['MEM']}

cd {task_config['CD_PATH_IN_JOB_SCRIPT']}
{task_config['CONDA_ACTIVATE_PATH']}
which python
export TOKENIZERS_PARALLELISM=false
python3 -u {os.path.abspath(__file__)} \
    --data_path {args.data_path} \
    --output_dir {model_out_dir} \
    --n_sampling {task_config['N_SAMPLING']} \
    --model_path {model_path} \
    --model_name {model_name}
""")
        sbatch_commands.append((f"sbatch {job_script}", model_name))
    
    print("\n--- 📝 Summary of Math500 evaluation plan ---")
    if models_to_run:
        print(f"\n🚀 Models to run (total {len(models_to_run)} models):")
        for model in models_to_run:
            print(f"  - {model}")
    
    if models_skipped:
        print(f"\n🚫 Skipped models (total {len(models_skipped)} models, result.csv exists):")
        for model in models_skipped:
            print(f"  - {model}")
    
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
                    print(f"Successfully submitted: {model_name} (Job ID: {job_id})")
                else:
                    print(f"Submitted but could not parse job ID: {model_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit: {model_name}")
            time.sleep(0.2)
    
    print(f"\n🚀 All Math500 evaluation tasks submitted ({len(models_to_run)} submitted, {len(models_skipped)} skipped)")
    return submitted_slurm_job_ids, models_to_run, models_skipped

# =====================
# Wait for jobs completion
# =====================
def wait_for_jobs_completion(submitted_slurm_job_ids):
    """Wait for all submitted SLURM jobs to complete."""
    if not submitted_slurm_job_ids:
        return
    
    print(f"\n🔄 Waiting for {len(submitted_slurm_job_ids)} SLURM jobs to complete...")
    
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
                print(f"🔄 {len(submitted_slurm_job_ids)} jobs still running/pending. Checking again in 30 seconds...")
                time.sleep(30)
        except subprocess.CalledProcessError:
            print("All jobs appear to have completed.")
            break
    
    print("\n✅ All SLURM jobs have completed!")

# =====================
# Summarize results
# =====================
def summarize_passk_for_all_models(task_config):
    """Summarize pass@k results for all models."""
    passk_json = os.path.join(task_config["BASE_OUT"], "passk.json")
    
    if os.path.exists(passk_json):
        try:
            with open(passk_json, 'r') as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
    else:
        all_results = {}
    
    # Update pass@k for all models
    for model_path, model_name in Model_map.items():
        model_dir = os.path.join(task_config["BASE_OUT"], model_name)
        csv_path = os.path.join(model_dir, "result.csv")
        
        if not os.path.exists(csv_path):
            continue
        
        data = pd.read_csv(csv_path, header=None).values
        all_samples = data.tolist()
        
        results = {}
        for k in task_config["K_LIST"]:
            pass_at_k_scores = []
            for sample_attempts in all_samples:
                n = len(sample_attempts)
                c = sum(sample_attempts)
                sample_pass_k = pass_at_k(n, c, k)
                pass_at_k_scores.append(sample_pass_k)
            results[f"pass@{k}"] = float(np.mean(pass_at_k_scores))
        
        all_results[model_name] = results
    
    with open(passk_json, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ All models pass@k results saved to: {passk_json}")
    print(f"🎉 Math500 evaluation completed!")

# =====================
# Main Entry
# =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Math500 Evaluation with math_verify")
    parser.add_argument("--data_path", type=str, default=TASK_CONFIG["DATA_PATH"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_sampling", type=int, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--submit_jobs", action="store_true", help="Submit SLURM jobs for all models")
    parser.add_argument("--reforce", action="store_true", help="Rerun even if result.csv exists")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.submit_jobs:
        # Submit jobs for all models
        submitted_ids, models_run, models_skipped = submit_jobs_for_all_models(args, TASK_CONFIG)
        wait_for_jobs_completion(submitted_ids)
        summarize_passk_for_all_models(TASK_CONFIG)
    else:
        # Run single model evaluation
        assert args.model_path is not None and args.output_dir is not None
        n_sampling = args.n_sampling if args.n_sampling is not None else TASK_CONFIG["N_SAMPLING"]
        run_single_model_evaluation(
            args.data_path,
            args.output_dir,
            n_sampling,
            args.model_path,
            args.model_name,
            TASK_CONFIG["BASE_OUT"],
            overwrite=args.reforce,
            task_config=TASK_CONFIG
        )