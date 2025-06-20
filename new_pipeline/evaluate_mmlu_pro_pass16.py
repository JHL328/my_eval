import numpy as np
import pandas as pd
import json
import re
from optparse import OptionParser
import time
import random
import os
import sys
from vllm import LLM, SamplingParams

# Output directory for results
output_dir = "/mnt/sharefs/users/haolong.jia/result/mmlu_pro_pass16"
os.makedirs(output_dir, exist_ok=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Model_map

def parse_args():
    parser = OptionParser()
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    parser.add_option("--model", type="str", dest="model")
    parser.add_option("--subject", type="str", dest="subject")
    parser.add_option("--prompts_path", type="str", dest="prompts_path", default="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_pro_prompts.json")
    (options, args) = parser.parse_args()
    return options

def load_prompts_from_json(subject, prompts_path, idx_start, idx_end):
    with open(prompts_path, 'r', encoding='utf-8') as f:
        all_prompts = json.load(f)
    if subject not in all_prompts:
        raise ValueError(f"Subject {subject} not found in prompts json!")
    subject_prompts = all_prompts[subject][idx_start:idx_end]
    return subject_prompts

def build_target(example):
    # The answer is a single uppercase letter (A-P)
    return example['answer'].strip().upper()

def extract_answer(text):
    # Try to extract answer using the template's regex: 'answer is \(?([ABCDEFGHIJ])\)?'
    match = re.search(r"answer is \(?([ABCDEFGHIJ])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    # fallback: match 'the answer is (X)' or 'the answer is X' for A-P
    match2 = re.search(r"[Tt]he answer is[\s\(]*([A-P])[\)\s\.]*", text)
    if match2:
        return match2.group(1).strip().upper()
    # fallback: look for a single capital letter (A-P) in the last line
    lines = text.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r"([A-P])", line)
        if m:
            return m.group(1)
    return ""

def main():
    options = parse_args()
    idx_start = options.idx_start
    idx_end = options.idx_end
    model_path = options.model
    subject = options.subject
    prompts_path = options.prompts_path

    model_name = Model_map[model_path]
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Load prompts and targets
    subject_prompts = load_prompts_from_json(subject, prompts_path, idx_start, idx_end)
    prompts = [ex['prompt'] for ex in subject_prompts]
    targets = [build_target(ex) for ex in subject_prompts]

    print("==== First Prompt Example ====")
    print(prompts[0])
    print("=============================")

    # Load model
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
        stop=["</s>"]
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
        input_text = prompts[i]
        gt = ground_truth
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
    scores_df.to_csv(f'{model_dir}/{subject}_{idx_start}-{idx_end}.csv', index=False)
    print("Finished generating csv file")

    # save jsonl
    jsonl_path = f'{model_dir}/{subject}_{idx_start}-{idx_end}.jsonl'
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
