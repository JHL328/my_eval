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

# output_dir will be set based on model type

# load Model_map
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import get_model_map_by_type

# load fewshot from json
def load_fewshot_from_json(subject, cot_prompts_path, num_fewshot=4):
    with open(cot_prompts_path, 'r', encoding='utf-8') as f:
        cot_prompts = json.load(f)
    if subject not in cot_prompts:
        raise ValueError(f"Subject {subject} not found in cot_prompts json!")
    content = cot_prompts[subject]
    blocks = content.split('\nQ: ')
    prefix = blocks[0]
    qa_blocks = blocks[1:num_fewshot+1]
    fewshot = prefix
    for qa in qa_blocks:
        fewshot += '\nQ: ' + qa
    return fewshot

# construct single prompt, refer to harness template
# doc_to_text: Q: {{question.strip()}}\n(A) {{choices[0]}} (B) {{choices[1]}} (C) {{choices[2]}} (D) {{choices[3]}}\nA: Let's think step by step.
def build_prompt(fewshot, example):
    q = example['question'].strip()
    choices = example['choices']
    prompt = f"Q: {q}\n(A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}\nA: Let's think step by step."
    return fewshot + '\n\n' + prompt

# target: {{['(A)', '(B)', '(C)', '(D)'][answer] if answer is defined else target}}
def build_target(example):
    idx = example['answer']
    return f"({chr(ord('A') + idx)})"

# use regex to extract answer is (X)
def extract_answer(text):
    match = re.search(r"answer is ([A-D])", text)
    if match:
        return f"({match.group(1)})"
    # fallback: only find (X)
    match2 = re.search(r"\(([A-D])\)", text)
    if match2:
        return f"({match2.group(1)})"
    return ""

def parse_args():
    parser = OptionParser()
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    parser.add_option("--results_dir", type="str", dest="results_dir")
    parser.add_option("--model", type="str", dest="model")
    parser.add_option("--subject", type="str", dest="subject")
    parser.add_option("--cot_prompts_path", type="str", dest="cot_prompts_path", default="/mnt/weka/home/haolong.jia/eval/RL-eval/lm-evaluation-harness/lm_eval/tasks/mmlu/flan_cot_fewshot/_cot_prompts.json")
    parser.add_option("--type", type="str", dest="type", default="base", help="Model type: base or sft")
    (options, args) = parser.parse_args()
    return options

def main():
    options = parse_args()
    idx_start = options.idx_start
    idx_end = options.idx_end
    results_dir = options.results_dir
    model_path = options.model  
    subject = options.subject
    cot_prompts_path = options.cot_prompts_path
    model_type = options.type

    # Set output dir based on model type
    if model_type == "sft":
        output_dir = "/mnt/sharefs/users/haolong.jia/result/mmlu_flan_pass16_sft"
    else:
        output_dir = "/mnt/sharefs/users/haolong.jia/result/mmlu_flan_pass16"
    os.makedirs(output_dir, exist_ok=True)

    # get model_name from Model_map
    model_map = get_model_map_by_type(model_type)
    model_name = model_map[model_path]
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # load fewshot from json
    fewshot = load_fewshot_from_json(subject, cot_prompts_path, num_fewshot=4)

    # load data
    dataset = load_dataset(
        "hails/mmlu_no_train",
        subject,
        cache_dir="/mnt/sharefs/users/haolong.jia/eval_data",
        trust_remote_code=True
    )
    data = dataset['test'].select(range(idx_start, idx_end))

    print(type(data), data[:2])

    prompts = [build_prompt(fewshot, ex) for ex in data]
    targets = [build_target(ex) for ex in data]

    # 为SFT模型使用chat template
    if model_type == "sft":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        prompts = formatted_prompts

    # print first prompt content, for manual inspection of fewshot and cot prompt concatenation
    print("==== First Prompt Example ====")
    print(prompts[0])
    print("=============================")

    # load model
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

    # generate
    gens = llm.generate(prompts, sampling_params, use_tqdm=True)
    print(f"Generated {len(gens)} responses")
    print("preserve in the csv file")

    n_prompts = len(prompts)
    n_samples = 16
    scores_matrix = np.zeros((n_prompts, n_samples))

    print("---------print the first sample---------")
    for i, (output, ground_truth) in enumerate(zip(gens, targets)):
        for j, single_output in enumerate(output.outputs):
            response_text = single_output.text
            pred = extract_answer(response_text)
            scores_matrix[i, j] = int(pred == ground_truth)
            # print first sample
            if i == 0 and j == 0:
                print(f"--------------------------------")
                print(f"PROMPT: {prompts[i]}")
                print(f"RESPONSE: {response_text}")
                print(f"PARSED ANSWER: {pred}")
                print(f"GROUND TRUTH: {ground_truth}")
                print(f"SCORE: {int(pred == ground_truth)}\n\n")

    # save
    scores_df = pd.DataFrame(scores_matrix)
    scores_df.to_csv(f'{model_dir}/{subject}_{idx_start}-{idx_end}.csv', index=False)
    print("Finished generating csv file")
    del llm  # free memory
    import gc; gc.collect()
    print("EVAL SCRIPT FINISHED")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"EVAL SCRIPT FAILED: {e}")
        import traceback; traceback.print_exc()
        exit(1)
