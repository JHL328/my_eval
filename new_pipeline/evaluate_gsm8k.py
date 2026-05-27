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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Model_map, get_model_map_by_type

# =====================
# Task Configurations
# =====================
TASK_CONFIGS = {
    "gsm8k": {
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/gsm8k_pass16",
        "EVAL_SCRIPT": None,
        "DATA_NAME": "gsm8k",
        "K_LIST": [1, 2, 4, 8, 16],
        "N_SAMPLING": 16,
        "NUM_SHOTS": 8,
        "PROMPT_TYPE": "cot",
        "GPUS_PER_TASK": 1,
        "TIME_LIMIT": "12:00:00",
        "PARTITION": "main",
        # "QOS": "lowprio",
        "MEM": "800G",
        "FEWSHOT_EXAMPLES": [
            {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "target": "Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."},
            {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "target": "Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."},
            {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "target": "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."},
            {"question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "target": "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."},
            {"question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "target": "Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."},
            {"question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "target": "Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."},
            {"question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "target": "Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."},
            {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "target": "Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."},
        ],
        "CONDA_ACTIVATE_PATH": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval",
        # "CONDA_ACTIVATE_PATH": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate base",
        "CD_PATH_IN_JOB_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline",
    },
    "math500": {
        "BASE_OUT": "/mnt/weka/shrd/k2m/haolong.jia/result/math500_pass64",
        "EVAL_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/math_eval.py",
        "DATA_NAME": "math500",
        "K_LIST": [1, 2, 4, 8, 16, 32, 64],
        "N_SAMPLING": 64,
        "NUM_SHOTS": 4,
        "PROMPT_TYPE": "cot",
        "GPUS_PER_TASK": 1,
        "TIME_LIMIT": "12:00:00",
        # "PARTITION": "lowprio",
        # "QOS": "lowprio",
        "PARTITION": "main",
        "MEM": "800G",
        "FEWSHOT_EXAMPLES": None,
        "CONDA_ACTIVATE_PATH": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval",
        "CD_PATH_IN_JOB_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation",
    }
}

SAMPLING_PARAMS = dict(
    temperature=0.6,
    top_p=0.95,
    n=16,
    max_tokens=4096,
    stop=["</s>", "<|im_end|>"],
    seed=42,
)

# =====================
# Utility Functions
# =====================
def generate_fewshot_prompt(fewshot_examples):
    if not fewshot_examples:
        return ""
    prompt = ""
    for ex in fewshot_examples:
        prompt += f"Q: {ex['question']}\nA: {ex['target']}\n\n"
    return prompt

def parse_answer_with_verify(text):
    """Parse answer using math_verify library with fallback to regex patterns."""
    try:
        # First try to parse with math_verify
        parsed = parse(text)
        if parsed is not None:
            return parsed
    except Exception:
        pass
    
    # Fallback to regex patterns for extraction
    answer_patterns = [
        r"The answer is:?\s*\$?([\-0-9\.,]+)",
        r"#### ?\$?([\-0-9\.,]+)",
        r"Therefore,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"So,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Thus,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Hence,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Final answer:?\s*\$?([\-0-9\.,]+)",
        r"The final answer is:?\s*\$?([\-0-9\.,]+)",
        r"The answer is:?\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?",
        r"=\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?\.?\s*(?:The answer|$)",
    ]
    for pat in answer_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            # take the last match (usually the final answer)
            ans = matches[-1].replace(",", "").strip().rstrip(".")
            if ans:
                try:
                    return parse(ans)
                except:
                    return ans
    
    sentence_end_pattern = r"(?:is|are|equals?|makes?|has|have|gets?|arrives?|covers?|travels?)\s+\$?([\-0-9\.,]+)(?:\s*(?:miles?|minutes?|hours?|dollars?|GB))?\.?\s*$"
    m = re.search(sentence_end_pattern, text, re.MULTILINE | re.IGNORECASE)
    if m:
        ans = m.group(1).replace(",", "").strip().rstrip(".")
        if ans:
            try:
                return parse(ans)
            except:
                return ans
    
    # last fallback: find the last number in the last complete sentence
    sentences = text.split('.')
    for sent in reversed(sentences):
        # skip sentences containing Human/Assistant (possibly irrelevant content)
        if 'Human:' in sent or 'Assistant:' in sent:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sent)
        if numbers:
            num = numbers[-1].lstrip('0') or '0'
            try:
                return parse(num)
            except:
                return num
    return None

def compare_answers(gold, pred):
    """Compare two answers using math_verify with fallback to string comparison."""
    try:
        # If both are already parsed objects from math_verify
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
    gold_str = gold_str.replace(",", "").strip().rstrip(".")
    pred_str = pred_str.replace(",", "").strip().rstrip(".")
    
    return gold_str == pred_str

def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def is_job_running_or_done(model_out_dir):
    result_csv = os.path.join(model_out_dir, "result.csv")
    return os.path.exists(result_csv)

def update_passk_json(passk_path, model_name, passk_result, overwrite=False):
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
# Math500 Postprocess
# =====================
def postprocess_math_results(model_out_dir, model_name, task_config):
    # 1. move nested directories
    base_name = os.path.basename(model_name)
    math500_dir = os.path.join(model_out_dir, "math500")
    
    if not os.path.isdir(math500_dir):
        return
    
    # Look for checkpoint directories (e.g., checkpoint-5472)
    import glob, shutil
    checkpoint_dirs = glob.glob(os.path.join(math500_dir, "checkpoint-*"))
    
    if checkpoint_dirs:
        # Use the first checkpoint directory found
        target_dir = checkpoint_dirs[0]
    else:
        # Fallbacks for other directory layouts
        candidate = os.path.join(math500_dir, base_name)
        if os.path.isdir(candidate):
            target_dir = candidate
        else:
            # Some runs write into a numeric step directory (e.g., math500/19122)
            subdirs = [
                os.path.join(math500_dir, d)
                for d in os.listdir(math500_dir)
                if os.path.isdir(os.path.join(math500_dir, d))
            ]
            subdirs.sort()
            target_dir = subdirs[0] if subdirs else None
    
    if target_dir and os.path.isdir(target_dir):
        # process jsonl
        jsonl_files = glob.glob(os.path.join(target_dir, "*.jsonl"))
        if jsonl_files:
            dest_jsonl = os.path.join(model_out_dir, "sample.jsonl")
            if not os.path.exists(dest_jsonl):
                shutil.move(jsonl_files[0], dest_jsonl)
            else:
                # destination already exists; keep newest file by replacing
                shutil.move(jsonl_files[0], dest_jsonl)
        # process json
        json_files = glob.glob(os.path.join(target_dir, "*.json"))
        if json_files:
            dest_json = os.path.join(model_out_dir, "result.json")
            if not os.path.exists(dest_json):
                shutil.move(json_files[0], dest_json)
            else:
                shutil.move(json_files[0], dest_json)
        # remove the nested directory if it is now empty to keep the folder tidy
        if not os.listdir(target_dir):
            shutil.rmtree(target_dir)
    # 2. generate result.csv
    sample_jsonl = os.path.join(model_out_dir, "sample.jsonl")
    if not os.path.exists(sample_jsonl):
        return
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
    # 3. calculate pass@k and exact_match, write to metrics.txt
    if all_scores:
        n_sampling = task_config["N_SAMPLING"]
        passk_dict = {}
        for k in task_config["K_LIST"]:
            pass_at_k_scores = []
            for sample_attempts in all_scores:
                n = len(sample_attempts)
                c = sum(sample_attempts)
                sample_pass_k = pass_at_k(n, c, k)
                pass_at_k_scores.append(sample_pass_k)
            passk_dict[f"pass@{k}"] = float(np.mean(pass_at_k_scores))
        total_correct = sum(sum(sample_attempts) for sample_attempts in all_scores)
        total_attempts = sum(len(sample_attempts) for sample_attempts in all_scores)
        exact_match = total_correct / total_attempts if total_attempts > 0 else 0.0
        metrics_txt_path = os.path.join(model_out_dir, "metrics.txt")
        with open(metrics_txt_path, "w") as f:
            f.write(f"exact_match: {exact_match:.4f}\n")
            for k in task_config["K_LIST"]:
                f.write(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}\n")

# =====================
# Main Evaluation Function (GSM8K only)
# =====================
def run_single_model_evaluation(task_name, gsm8k_path, output_dir, n_sampling, model_path, model_name, base_out, overwrite, task_config, model_type="base"):
    if task_name != "gsm8k":
        return  # only run for gsm8k
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, dtype="auto", tensor_parallel_size=1, trust_remote_code=True)
    print("Model loaded successfully!")
    
    # Load tokenizer for SFT chat template
    tokenizer = None
    if model_type == "sft":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Tokenizer loaded for SFT chat template.")

    results = []
    csv_rows = []
    with open(gsm8k_path, 'r') as f:
        data = [json.loads(line) for line in f]
    # prepare all prompts
    print("Preparing prompts...")
    fewshot_prompt = generate_fewshot_prompt(task_config["FEWSHOT_EXAMPLES"])
    prompts, golds, questions = [], [], []
    
    for item in tqdm(data, desc="Preparing data"):
        q = item["question"]
        gold = parse_answer_with_verify(item["answer"])
        
        # Base prompt construction
        raw_prompt = fewshot_prompt + f"Q: {q}\nA: Let's think step by step."
        
        if model_type == "sft":
            # Apply chat template
            # We treat the entire constructed few-shot prompt as the user input
            messages = [{"role": "user", "content": raw_prompt}]
            final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            final_prompt = raw_prompt
            
        prompts.append(final_prompt)
        golds.append(gold)
        questions.append(q)
    
    # run inference in batch, let vLLM handle batching
    print(f"Running inference on {len(prompts)} prompts...")
    
    # Update stop tokens for SFT
    current_stop = SAMPLING_PARAMS["stop"]
    if model_type == "sft" and "<|im_end|>" not in current_stop:
        current_stop = current_stop + ["<|im_end|>"]
    elif model_type != "sft":
        # Base models are prompted with 8-shot "Q: ... A: ..." pairs. Without a
        # stop at the next "Q:", the model keeps emitting fabricated Q/A pairs and
        # the answer extractor (which takes the LAST "The answer is" match) grades
        # a hallucinated question instead of the real one -> spuriously low scores.
        current_stop = current_stop + ["\n\nQ:"]

    params = SamplingParams(**{**SAMPLING_PARAMS, "n": n_sampling, "stop": current_stop})
    outputs = llm.generate(prompts, params)
    
    # process results
    print("Processing results...")
    for idx, (q, gold, output) in enumerate(tqdm(zip(questions, golds, outputs), total=len(questions), desc="Processing results")):
        generations = [output.outputs[i].text for i in range(n_sampling)]
        parsed = [parse_answer_with_verify(gen) for gen in generations]
        em = [compare_answers(gold, p) for p in parsed]
        passk = any(em)
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
        
        results.append({
            "question": q,
            "gold": gold_str,
            "generations": generations,
            "parsed": parsed_str,
            "exact_match": em,
            "pass@16": passk
        })
        
        # debug info for the first sample
        if idx == 0:
            print("\n==== First Sample Debug Info ====\nPrompt:\n{}\n\nGenerations:".format(prompts[idx]))
            for i, g in enumerate(generations):
                print(f"[{i+1}] {g}")
            print(f"\nGT: {gold_str}")
            print(f"Parsed: {parsed_str}")
            print(f"EM: {em}")
            print("===============================\n")
        row = {"question": q, "gt": gold_str}
        for i in range(n_sampling):
            row[f"gen_{i+1}"] = generations[i]
            row[f"parse_{i+1}"] = parsed_str[i]
            row[f"em_{i+1}"] = int(em[i])
        csv_rows.append(row)
    
    passk_rate = sum(r["pass@16"] for r in results) / len(results)
    em_total = sum(sum(r["exact_match"]) for r in results) / (len(results) * n_sampling)
    passk_dict = {}
    for k in task_config["K_LIST"]:
        count = 0
        for r in results:
            c = sum(r["exact_match"])
            count += pass_at_k(n_sampling, c, k)
        passk_dict[f"pass@{k}"] = count / len(results)
    with open(os.path.join(output_dir, "gsm8k_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"pass@16: {passk_rate:.4f}\nexact_match: {em_total:.4f}\n")
        for k in task_config["K_LIST"]:
            f.write(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}\n")
    csv_path = os.path.join(output_dir, "result.csv")
    with open(csv_path, "w", newline='') as csvfile:
        fieldnames = [f"em_{i+1}" for i in range(n_sampling)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            # only extract em_1 to em_16
            em_row = {f"em_{i+1}": row[f"em_{i+1}"] for i in range(n_sampling)}
            writer.writerow(em_row)
    # write to base_out/passk.json
    if model_name is not None and base_out is not None:
        passk_path = os.path.join(base_out, "passk.json")
        update_passk_json(passk_path, model_name, passk_dict, overwrite=overwrite)

# =====================
# Submit jobs for all models (with job queue and skip queue)
# =====================
def submit_jobs_for_all_models(args, task_config):
    os.makedirs(task_config["BASE_OUT"], exist_ok=True)
    
    # choose model map according to type
    if hasattr(args, 'type') and args.type == 'sft':
        model_map = get_model_map_by_type('sft')
    else:
        model_map = Model_map
    
    models_to_run = []
    models_skipped = []
    sbatch_commands = []
    submitted_slurm_job_ids = []
    for model_path, model_name in model_map.items():
        model_out_dir = os.path.join(task_config["BASE_OUT"], model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        if not args.reforce and is_job_running_or_done(model_out_dir):
            models_skipped.append(model_name)
            continue
        models_to_run.append(model_name)
        job_name = f"{args.task}_{model_name}"
        job_script = os.path.join(model_out_dir, f"{job_name}.sh")
        with open(job_script, 'w') as f:
            if args.task == "gsm8k":
                f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={model_out_dir}/slurm.out
#SBATCH --error={model_out_dir}/slurm.err
#SBATCH --gres=gpu:{task_config['GPUS_PER_TASK']}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time={task_config['TIME_LIMIT']}
#SBATCH --partition={task_config['PARTITION']}
#SBATCH --mem={task_config['MEM']}

cd {task_config['CD_PATH_IN_JOB_SCRIPT']}
{task_config['CONDA_ACTIVATE_PATH']}
which python
    export TOKENIZERS_PARALLELISM=false
    python3 -u {os.path.abspath(__file__)} \
    --task {args.task} \
    --gsm8k_path {args.gsm8k_path} \
    --output_dir {model_out_dir} \
    --n_sampling {task_config['N_SAMPLING']} \
    --model_path {model_path} \
    --model_name {model_name} \
    --type {args.type}
""")
            elif args.task == "math500":
                # apply chat template according to type
                apply_chat_template_flag = "--apply_chat_template" if hasattr(args, 'type') and args.type == 'sft' else ""
                
                f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={model_out_dir}/slurm.out
#SBATCH --error={model_out_dir}/slurm.err
#SBATCH --gres=gpu:{task_config['GPUS_PER_TASK']}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time={task_config['TIME_LIMIT'    ]}
#SBATCH --partition={task_config['PARTITION']}

#SBATCH --mem={task_config['MEM']}

cd {task_config['CD_PATH_IN_JOB_SCRIPT']}
{task_config['CONDA_ACTIVATE_PATH']}
which python
export TOKENIZERS_PARALLELISM=false
python3 -u {task_config['EVAL_SCRIPT']} \
    --model_name_or_path {model_path} \
    --data_names {task_config['DATA_NAME']} \
    --output_dir {model_out_dir} \
    --split test \
    --prompt_type {task_config['PROMPT_TYPE']} \
    --num_test_sample -1 \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling {task_config['N_SAMPLING']} \
    --top_p 0.95 \
    --max_tokens_per_call 4096 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --num_shots {task_config['NUM_SHOTS']} \
    {apply_chat_template_flag}
""")
        sbatch_commands.append((f"sbatch {job_script}", model_name))
    print("\n--- 📝 summary of the model evaluation plan ---")
    if models_to_run:
        print(f"\n📊models to run (total {len(models_to_run)} models):")
        for model in models_to_run:
            print(f"  - {model}")
    if models_skipped:
        print(f"\n📌 skipped models (total {len(models_skipped)} models, because result.csv already exists):")
        for model in models_skipped:
            print(f"  - {model}")
    if not models_to_run and not models_skipped:
        print("\nno models to run.")
    # Submit jobs and capture job IDs
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
    print(f"\nall model evaluation tasks have been submitted (submitted {len(models_to_run)} models, skipped {len(models_skipped)} models).")
    return submitted_slurm_job_ids, models_to_run, models_skipped

# =====================
# Wait for all jobs to complete
# =====================
def wait_for_jobs_completion(submitted_slurm_job_ids):
    if not submitted_slurm_job_ids:
        return
    print(f"\n⏳ waiting for all {len(submitted_slurm_job_ids)} Slurm jobs to complete...")
    
    user = os.environ.get('USER')
    
    while submitted_slurm_job_ids:
        try:
            # Use -u USER if available to avoid errors when specific jobs finish
            if user:
                cmd = f"squeue -h -u {user} -o '%i'"
            else:
                job_ids_str = ",".join(submitted_slurm_job_ids)
                cmd = f"squeue -h -j {job_ids_str} -o '%i'"
            
            # Use subprocess.run to capture errors without throwing exception immediately
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                err_msg = result.stderr.strip()
                # If failure is likely due to connection/slurm issues, retry
                if "slurm_load_jobs error" in err_msg or "Socket timed out" in err_msg:
                    print(f"⚠️ Slurm communication error: {err_msg}. Retrying...")
                    time.sleep(30)
                    continue
                
                # If using -j and error is Invalid job id, it implies jobs are done.
                if not user and "Invalid job id" in err_msg:
                    print("Jobs not found in queue (completed).")
                    break
                
                # Unknown error, print and retry
                print(f"⚠️ squeue failed with code {result.returncode}: {err_msg}. Retrying...")
                time.sleep(30)
                continue

            running_ids = set(result.stdout.strip().split())
            submitted_slurm_job_ids = [jid for jid in submitted_slurm_job_ids if jid in running_ids]
            
            if submitted_slurm_job_ids:
                print(f"📊 {len(submitted_slurm_job_ids)} jobs still running/pending. Checking again in 30 seconds...")
                time.sleep(30)
        
        except Exception as e:
            print(f"⚠️ Exception in monitoring loop: {e}. Retrying...")
            time.sleep(30)

    print("\n✅ All Slurm jobs have completed!")

# =====================
# Summarize pass@k for all models
# =====================
def summarize_passk_for_all_models(task_name, task_config, model_map=None):
    if model_map is None:
        model_map = Model_map
    
    passk_json = os.path.join(task_config["BASE_OUT"], "passk.json")
    if os.path.exists(passk_json):
        try:
            with open(passk_json, 'r') as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
    else:
        all_results = {}
    # 2. update/add pass@k for all models
    for model_path, model_name in model_map.items():
        model_dir = os.path.join(task_config["BASE_OUT"], model_name)
        csv_path = os.path.join(model_dir, "result.csv")
        if not os.path.exists(csv_path):
            continue
        data = pd.read_csv(csv_path).values
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
    print(f"\n✅ all models pass@k results saved to: {passk_json}")
    print(f"🎉 evaluation completed!")

# =====================
# Main Entry
# =====================
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "math500"], help="Task name: gsm8k or math500")
    parser.add_argument("--gsm8k_path", type=str, default="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/gsm8k_test.jsonl")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_sampling", type=int, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--submit_jobs", action="store_true", help="If set, submit slurm jobs for all models.")
    parser.add_argument("--reforce", action="store_true", help="If set, rerun evaluation even if result.csv already exists.")
    parser.add_argument("--type", type=str, default="base", choices=["base", "sft"], help="Model type: base or sft")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    task_config = TASK_CONFIGS[args.task].copy()  # Create a copy to avoid modifying the original
    
    # SFT model special configuration
    if args.task == "math500" and hasattr(args, 'type') and args.type == "sft":
        task_config["BASE_OUT"] = "/mnt/weka/shrd/k2m/haolong.jia/result/math500_pass64_sft"
    
    if args.submit_jobs:
        submitted_ids, models_run, models_skipped_list = submit_jobs_for_all_models(args, task_config)
        wait_for_jobs_completion(submitted_ids)
        # math500: postprocess after all jobs
        if args.task == "math500":
            # use the correct model_map
            model_map = get_model_map_by_type(args.type) if hasattr(args, 'type') and args.type == 'sft' else Model_map
            for model_path, model_name in model_map.items():
                model_out_dir = os.path.join(task_config["BASE_OUT"], model_name)
                postprocess_math_results(model_out_dir, model_name, task_config)
        # Pass the correct model_map to summarize function
        used_model_map = get_model_map_by_type(args.type) if hasattr(args, 'type') and args.type == 'sft' else Model_map
        summarize_passk_for_all_models(args.task, task_config, used_model_map)
    else:
        assert args.model_path is not None and args.output_dir is not None
        n_sampling = args.n_sampling if args.n_sampling is not None else task_config["N_SAMPLING"]
        run_single_model_evaluation(args.task, args.gsm8k_path, args.output_dir, n_sampling, args.model_path, args.model_name, task_config["BASE_OUT"], overwrite=args.reforce, task_config=task_config, model_type=args.type)
