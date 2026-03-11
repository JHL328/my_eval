import numpy as np
import pandas as pd
import json
import re
from optparse import OptionParser
import time
import random
from datasets import load_dataset
from vllm import LLM, SamplingParams
import os
import sys

# eval result output dir will be set based on model type

from model import get_model_map_by_type

def parse_args():
    parser = OptionParser()
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    parser.add_option("--model", type="str", dest="model")
    parser.add_option("--task", type="str", dest="task")
    parser.add_option("--cot_prompts_path", type="str", dest="cot_prompts_path", default="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/bbh_cot_prompts.json")
    parser.add_option("--tp_size", type="int", dest="tp_size", default=1, help="Tensor parallel size for VLLM")
    parser.add_option("--type", type="str", dest="type", default="base", help="Model type: base or sft")
    (options, args) = parser.parse_args()
    return options

def load_fewshot_from_json(task, cot_prompts_path):
    with open(cot_prompts_path, 'r', encoding='utf-8') as f:
        cot_prompts = json.load(f)
    if task not in cot_prompts:
        raise ValueError(f"Task {task} not found in cot_prompts json!")
    return cot_prompts[task]

def build_target(example):
    # BBH target is free-form, but always ends with "So the answer is ..."
    # We use the regex extraction below
    return example['target']

def extract_answer(text):
    # Try to extract answer after "the answer is "
    match = re.search(r"the answer is ([^\n\.]*)", text)
    if match:
        return match.group(1).strip()
    # fallback: last word
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if "answer" in line:
            return line.split()[-1].strip('.').strip()
    return ""

def build_prompt(fewshot, example, model_type="base"):
    if model_type == "sft":
        """
        SFT model uses 0-shot, return the input
        """
        return example['input']
    else:
        """
        Base model keeps the original fewshot format
        """
        return fewshot + '\n\nQ: ' + example['input'] + '\nA: Let\'s think step by step.'

def main():
    options = parse_args()
    idx_start = options.idx_start
    idx_end = options.idx_end
    model_path = options.model
    task = options.task
    cot_prompts_path = options.cot_prompts_path
    model_type = options.type

    # Set output dir based on model type
    if model_type == "sft":
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/bbh_pass16_sft"
    else:
        output_dir = "/mnt/weka/shrd/k2m/haolong.jia/result/bbh_pass16"
    os.makedirs(output_dir, exist_ok=True)

    model_map = get_model_map_by_type(model_type)
    model_name = model_map[model_path]
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    fewshot = load_fewshot_from_json(task, cot_prompts_path)

    dataset = load_dataset(
        "lukaemon/bbh",
        task,
        cache_dir="/mnt/weka/shrd/k2m/haolong.jia/eval_data"
    )
    data = dataset['test'].select(range(idx_start, idx_end))
    print(type(data), data[:2])
    
    prompts = [build_prompt(fewshot, ex, model_type) for ex in data]
    targets = [build_target(ex) for ex in data]

    # For SFT model, use chat template
    if model_type == "sft":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # System prompt to guide the model to solve logical reasoning problems
        system_prompt = """You are a helpful assistant that solves logical reasoning problems step by step. When given a problem: 1. Think through the solution systematically 2. Show your reasoning process clearly 3. remember, must end with a clear final answer using the format: "So the answer is [your answer]". Note: there maybe some options provided in the problem, for example, (A) (B) (C) (D) OR Yes/No OR valid/invalid, You should keep the same format as the options, format like this: "So the answer is (A)" to help parse the answer. Remember to be precise and logical in your reasoning.
<think>"""
        
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        prompts = formatted_prompts

    print("==== First Prompt Example ====")
    print(prompts[0])
    print("=============================")

    # load model
    max_retries = 10
    for attempt in range(max_retries):
        try:
            llm = LLM(model=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=options.tp_size, enable_prefix_caching=True, dtype="auto", trust_remote_code=True)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} to load model failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.randint(2, 15))
            else:
                raise

    sampling_params = SamplingParams(
        max_tokens=1024,
        n=16,
        temperature=0.7,
        # stop=["</s>"]  # stop=["</s>"] will cause the model to stop generating when it sees </s>
    )

    gens = llm.generate(prompts, sampling_params, use_tqdm=True)
    print(f"Generated {len(gens)} responses")
    print("preserve in the csv file")

    n_prompts = len(prompts)
    n_samples = 16
    scores_matrix = np.zeros((n_prompts, n_samples))

    results_for_jsonl = []  # for saving input/target/outputs
    separator = "\n---\n"  # for separating different samples in jsonl

    print("---------print the first sample---------")
    for i, (output, ground_truth) in enumerate(zip(gens, targets)):
        input_text = data[i]['input']
        gt = ground_truth.strip()  # use original target
        outputs = []
        for j, single_output in enumerate(output.outputs):
            response_text = single_output.text
            pred = extract_answer(response_text)
            scores_matrix[i, j] = int(pred == gt)
            outputs.append(response_text)
            if i == 0 and j == 0:
                print(f"--------------------------------")
                print(f"PROMPT: {prompts[i]}")
                print(f"RESPONSE: {response_text}")
                print(f"PARSED ANSWER: {pred}")
                print(f"GROUND TRUTH: {gt}")
                print(f"SCORE: {int(pred == gt)}\n\n")
        # save input/target/outputs
        results_for_jsonl.append({
            "input": input_text,
            "target": gt,
            "outputs": separator.join(outputs)
        })

    # save csv
    scores_df = pd.DataFrame(scores_matrix)
    scores_df.to_csv(f'{model_dir}/{task}_{idx_start}-{idx_end}.csv', index=False)
    print("Finished generating csv file")

    # save jsonl
    jsonl_path = f'{model_dir}/{task}_{idx_start}-{idx_end}.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in results_for_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved input/outputs to {jsonl_path}")

    del llm
    import gc; gc.collect()
    print("EVAL SCRIPT FINISHED")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"EVAL SCRIPT FAILED: {e}")
        import traceback; traceback.print_exc()
        exit(1)
