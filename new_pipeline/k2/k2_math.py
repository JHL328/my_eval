import os
import sys
import json
import re
import csv
import numpy as np
import pandas as pd
import subprocess
import shutil
import glob
from optparse import OptionParser
from tqdm import tqdm
import gc
import fcntl

# --- Helper Functions (from evaluate_gsm8k.py) ---

def generate_fewshot_prompt(fewshot_examples):
    if not fewshot_examples: return ""
    return "".join(f"Q: {ex['question']}\nA: {ex['target']}\n\n" for ex in fewshot_examples)

def parse_answer(text):
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
                return ans
    sentence_end_pattern = r"(?:is|are|equals?|makes?|has|have|gets?|arrives?|covers?|travels?)\s+\$?([\-0-9\.,]+)(?:\s*(?:miles?|minutes?|hours?|dollars?|GB))?\.?\s*$"
    m = re.search(sentence_end_pattern, text, re.MULTILINE | re.IGNORECASE)
    if m:
        ans = m.group(1).replace(",", "").strip().rstrip(".")
        if ans:
            return ans
    # last fallback: find the last number in the last complete sentence
    sentences = text.split('.')
    for sent in reversed(sentences):
        # skip sentences containing Human/Assistant (possibly irrelevant content)
        if 'Human:' in sent or 'Assistant:' in sent:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sent)
        if numbers:
            return numbers[-1].lstrip('0') or '0'
    return ""

def pass_at_k(n, c, k):
    if c == 0 or n - c < k:
        return 1.0 if n - c < k else 0.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def update_summary_json(summary_path, model_name, results):
    """Atomically updates the summary JSON file with results for a model."""
    with open(summary_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = {}
        all_results[model_name] = results
        f.seek(0)
        f.truncate()
        json.dump(all_results, f, indent=4)
        fcntl.flock(f, fcntl.LOCK_UN)

# --- Task-specific Configurations (internal to this script) ---

TASK_CONFIGS = {
    "gsm8k": {
        "k_list": [1, 2, 4, 8, 16],
        "fewshot_examples": [
            {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "target": "Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."},
            {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "target": "Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."},
            {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "target": "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."},
            {"question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "target": "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."},
            {"question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "target": "Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."},
            {"question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "target": "Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."},
            {"question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "target": "Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."},
            {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "target": "Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."},
        ]
    },
    "math500": {
        "k_list": [1, 4, 16],
        "eval_script": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation/math_eval.py",
        "cd_path": "/mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation",
        "num_shots": 4
    }
}

# --- Evaluation Functions ---

def run_gsm8k_evaluation(options, task_config):
    from vllm import LLM, SamplingParams
    
    print("--- Running GSM8K evaluation with direct VLLM inference ---")
    llm = LLM(model=options.model_path, tensor_parallel_size=options.tp_size, gpu_memory_utilization=0.9, dtype="float32")
    
    sampling_params = SamplingParams(
        n=options.n_sampling, temperature=0.6, top_p=0.95, max_tokens=2048,
        stop=["Q:", "</s>", "<|im_end|>", "\n\nQ:", "\n\nHuman:", "\n\nAssistant:"]
    )
    
    with open(options.data_path, 'r') as f: dataset = [json.loads(line) for line in f]
    fewshot_prompt = generate_fewshot_prompt(task_config["fewshot_examples"])
    prompts = [fewshot_prompt + f"Q: {item['question']}\nA: Let's think step by step." for item in dataset]
    golds = [parse_answer(item['answer']) for item in dataset]

    print(f"Loaded {len(prompts)} samples. Starting inference...")
    gens = llm.generate(prompts, sampling_params, use_tqdm=True)

    results_data, all_pass_at_k_scores = [], {k: [] for k in task_config["k_list"]}
    for i, (output, gold) in enumerate(tqdm(zip(gens, golds), total=len(golds), desc="Processing results")):
        generations = [out.text for out in output.outputs]
        parsed_answers = [parse_answer(gen) for gen in generations]
        correct_matches = [p == gold for p in parsed_answers]
        for k in task_config["k_list"]:
            all_pass_at_k_scores[k].append(pass_at_k(options.n_sampling, sum(correct_matches), k))
        results_data.append({"question": dataset[i]['question'], "gold": gold, "parsed_answers": parsed_answers})

    final_pass_k_results = {f"pass@{k}": np.mean(all_pass_at_k_scores[k]) for k in task_config["k_list"]}
    model_dir = os.path.join(options.output_base_dir, options.model_name)
    pd.DataFrame(results_data).to_csv(os.path.join(model_dir, "gsm8k_results.csv"), index=False)
    
    del llm, gens
    gc.collect()
    return final_pass_k_results

def run_math500_evaluation_external(options, task_config):
    print("--- Running MATH500 evaluation by calling external script ---")
    model_out_dir = os.path.join(options.output_base_dir, options.model_name)
    command = (
        f"cd {task_config['cd_path']} && "
        f"python3 -u {task_config['eval_script']} "
        f"--model_name_or_path {options.model_path} "
        f"--data_names math500 "
        f"--output_dir {model_out_dir} "
        f"--prompt_type cot --num_test_sample -1 --seed 0 --temperature 0.6 "
        f"--n_sampling {options.n_sampling} --top_p 0.95 --max_tokens_per_call 4096 "
        f"--use_vllm --save_outputs --overwrite --num_shots {task_config['num_shots']}"
    )
    try:
        subprocess.run(command, shell=True, check=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"❌ External script execution failed: {e}", file=sys.stderr)
        return None

    # --- Post-process results: Flatten directory structure ---
    print(f"\n--- Post-processing results for {options.model_name} ---")
    nested_dir_name = os.path.basename(options.model_path)
    nested_dir_path = os.path.join(model_out_dir, 'math500', nested_dir_name)

    if os.path.isdir(nested_dir_path):
        print(f"Found nested results directory: {nested_dir_path}")
        jsonl_files = glob.glob(os.path.join(nested_dir_path, "*.jsonl"))
        if jsonl_files:
            shutil.move(jsonl_files[0], os.path.join(model_out_dir, "sample.jsonl"))
            print(f"  - Moved {os.path.basename(jsonl_files[0])} to {model_out_dir}/sample.jsonl")

        # Move the metrics json as a backup result file
        metrics_json_files = glob.glob(os.path.join(nested_dir_path, "*_metrics.json"))
        if metrics_json_files:
            shutil.move(metrics_json_files[0], os.path.join(model_out_dir, "result.json"))
            print(f"  - Moved {os.path.basename(metrics_json_files[0])} to {model_out_dir}/result.json")
        
        # Clean up the processed nested directory
        shutil.rmtree(os.path.join(model_out_dir, 'math500'))
        print(f"  - Cleaned up nested 'math500' directory.")
    else:
        print(f"⚠️  Nested result directory not found at {nested_dir_path}, assuming files are already processed.")

    # --- Calculate pass@k from sample.jsonl and write to metrics.txt ---
    sample_jsonl_path = os.path.join(model_out_dir, "sample.jsonl")
    if not os.path.exists(sample_jsonl_path):
        print(f"❌ Post-processing error: Cannot find sample.jsonl at {sample_jsonl_path}. Aborting.")
        return None

    all_scores = []
    try:
        with open(sample_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                scores = obj.get("score", [])
                # Ensure scores are in a list format
                row = [1 if s is True or s == 1 else 0 for s in (scores if isinstance(scores, list) else [scores])]
                all_scores.append(row)
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error reading or parsing {sample_jsonl_path}: {e}")
        return None

    if not all_scores:
        print(f"⚠️ Post-processing warning: No scores found in {sample_jsonl_path}, cannot calculate metrics.")
        return None

    final_pass_k_results = {}
    for k in task_config["k_list"]:
        pass_k_scores = [pass_at_k(len(sample), sum(sample), k) for sample in all_scores if sample]
        if pass_k_scores:
            final_pass_k_results[f"pass@{k}"] = np.mean(pass_k_scores)

    # --- Calculate exact_match and write all metrics to metrics.txt ---
    total_correct = sum(sum(s) for s in all_scores)
    total_attempts = sum(len(s) for s in all_scores)
    exact_match = total_correct / total_attempts if total_attempts > 0 else 0.0
    
    metrics_txt_path = os.path.join(model_out_dir, "metrics.txt")
    try:
        with open(metrics_txt_path, "w") as f:
            f.write(f"exact_match: {exact_match:.4f}\n")
            for k, v in final_pass_k_results.items():
                f.write(f"{k}: {v:.4f}\n")
        print(f"✅ Metrics calculated and saved to {metrics_txt_path}")
    except IOError as e:
        print(f"❌ Error writing metrics.txt: {e}")
        return None # Can't proceed if we can't write metrics
    
    return final_pass_k_results

# --- Main Dispatcher ---

def parse_args():
    parser = OptionParser()
    parser.add_option("--model_path", dest="model_path", help="Path to the HuggingFace model.")
    parser.add_option("--model_name", dest="model_name", help="A friendly name for the model.")
    parser.add_option("--output_base_dir", dest="output_base_dir", help="The base directory for evaluation results.")
    parser.add_option("--task_name", dest="task_name", choices=["gsm8k", "math500"], help="The math task to run.")
    parser.add_option("--data_path", dest="data_path", help="Path to the data file.")
    parser.add_option("--tp_size", dest="tp_size", type="int", help="Tensor parallel size for VLLM.")
    parser.add_option("--n_sampling", dest="n_sampling", type="int", help="Number of samples to generate (n for pass@k).")
    (options, args) = parser.parse_args()
    if not all(getattr(options, attr) for attr in ['model_path', 'model_name', 'output_base_dir', 'task_name', 'data_path', 'tp_size', 'n_sampling']):
        parser.error("All arguments are required.")
    return options

def main():
    options = parse_args()
    
    
    model_dir = os.path.join(options.output_base_dir, options.model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    result_json_path = os.path.join(model_dir, "result.json")
    if os.path.exists(result_json_path):
        print(f"⏩ Final result.json already exists for model '{options.model_name}' on task '{options.task_name}', skipping.")
        return

    task_config = TASK_CONFIGS[options.task_name]
    final_results = None
    if options.task_name == "gsm8k":
        final_results = run_gsm8k_evaluation(options, task_config)
    elif options.task_name == "math500":
        final_results = run_math500_evaluation_external(options, task_config)

    if final_results:
        with open(result_json_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\n--- 📊 Final Summary for {options.model_name} on {options.task_name.upper()} ---")
        for k, v in final_results.items():
            print(f"{k}: {v:.4f}")
        print(f"✅ Final model summary saved to {result_json_path}")
        
        # Contribute to the global summary json
        summary_json_path = os.path.join(options.output_base_dir, "result.json")
        update_summary_json(summary_json_path, options.model_name, final_results)
        print(f"✅ Contributed to global summary: {summary_json_path}")
    else:
        print(f"--- ⚠️ Evaluation failed for {options.model_name} on {options.task_name}, no result generated. ---")

if __name__ == "__main__":
    main()
