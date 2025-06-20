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

output_dir = "/mnt/sharefs/users/haolong.jia/result/mmlu"
os.makedirs(output_dir, exist_ok=True)

# load Model_map
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Model_map

def load_fewshot_from_json(subject, prompts_path, num_fewshot=5):
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    if subject not in prompts:
        raise ValueError(f"Subject {subject} not found in prompts json!")
    content = prompts[subject]
    blocks = content.split('\nQ: ')
    prefix = blocks[0]
    qa_blocks = blocks[1:num_fewshot+1]
    fewshot = prefix
    for qa in qa_blocks:
        fewshot += '\nQ: ' + qa
    return fewshot

def build_prompt(fewshot, example):
    fewshot = fewshot.rstrip()
    q = example['question'].strip()
    choices = example['choices']
    prompt = f"Q: {q}\n(A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}\nA:"
    return fewshot + '\n\n' + prompt

def build_target(example):
    idx = example['answer']
    return f"({chr(ord('A') + idx)})"

def extract_answer(text):
    match = re.search(r"\(([A-D])\)", text)
    if match:
        return f"({match.group(1)})"
    return ""

def parse_args():
    parser = OptionParser()
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    parser.add_option("--results_dir", type="str", dest="results_dir")
    parser.add_option("--model", type="str", dest="model")
    parser.add_option("--subject", type="str", dest="subject")
    parser.add_option("--prompts_path", type="str", dest="prompts_path", default="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_prompts.json")
    (options, args) = parser.parse_args()
    return options

def main():
    options = parse_args()
    idx_start = options.idx_start
    idx_end = options.idx_end
    results_dir = options.results_dir
    model_path = options.model  
    subject = options.subject
    prompts_path = options.prompts_path

    model_name = Model_map[model_path]
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    fewshot = load_fewshot_from_json(subject, prompts_path, num_fewshot=5)

    dataset = load_dataset(
        "hails/mmlu_no_train",
        subject,
        cache_dir="/mnt/sharefs/users/haolong.jia/eval_data",
        trust_remote_code=True
    )
    data = dataset['test'].select(range(idx_start, idx_end))

    prompts = [build_prompt(fewshot, ex) for ex in data]
    targets = [build_target(ex) for ex in data]

    print("==== First Prompt Example ====")
    print(prompts[0])
    print("=============================")

    max_retries = 10
    for attempt in range(max_retries):
        try:
            llm = LLM(model=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=1, enable_prefix_caching=True)
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
        # stop=["</s>"]
    )

    gens = llm.generate(prompts, sampling_params, use_tqdm=True)
    print(f"Generated {len(gens)} responses")

    n_prompts = len(prompts)
    n_samples = 16
    scores_matrix = np.zeros((n_prompts, n_samples))

    for i, (output, ground_truth) in enumerate(zip(gens, targets)):
        for j, single_output in enumerate(output.outputs):
            response_text = single_output.text
            pred = extract_answer(response_text)
            scores_matrix[i, j] = int(pred == ground_truth)
            if i == 0 and j == 0:
                print(f"PROMPT: {prompts[i]}")
                print(f"RESPONSE: {response_text}")
                print(f"PARSED ANSWER: {pred}")
                print(f"GROUND TRUTH: {ground_truth}")
                print(f"SCORE: {int(pred == ground_truth)}\n\n")

    scores_df = pd.DataFrame(scores_matrix)
    scores_df.to_csv(f'{model_dir}/{subject}_{idx_start}-{idx_end}.csv', index=False)
    print("Finished generating csv file")
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
