import numpy as np
import pandas as pd
import json
from optparse import OptionParser
import time

from vllm import LLM, SamplingParams
from sum_verifier import parse_assistant_output, score_answer
import random
from hf_utils import find_local_hf_model

# Prompt prefix as in sum_debug.ipynb
PREFIX = (
    "The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. "
    "In the answer mention each unknown and its solution, for example, <answer> x=5 y=10 </answer>. Now the user asks you to solve a math reasoning problem.\n\nUser:{quiz}\nAssistant: <think>"
)

def parse_args():
    parser = OptionParser()
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    parser.add_option("--results_dir", type="str", dest="results_dir")
    parser.add_option("--model", type="str", dest="model")
    (options, args) = parser.parse_args()
    return options

def load_prompts_and_targets(jsonl_file):
    prompts = []
    targets = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj["messages"]
            user_msg = messages[0]["content"]
            assistant_msg = messages[1]["content"]
            prompt = PREFIX.format(quiz=user_msg)
            _, target = parse_assistant_output(assistant_msg)
            prompts.append(prompt)
            targets.append(target)
    return prompts, targets

def main():
    options = parse_args()
    print(options)
    idx_start = options.idx_start
    idx_end = options.idx_end
    results_dir = options.results_dir
    model_name = options.model
    prompts, targets = load_prompts_and_targets("./sum_data/sum_test.jsonl")
    prompts = prompts[idx_start:idx_end]
    targets = targets[idx_start:idx_end]

    model_name = find_local_hf_model(model_name)
    print('Using model:', model_name)

    max_retries = 10
    for attempt in range(max_retries):
        try:
            if model_name == '/lustrefs/users/haolong.jia/train/nanotron_checkpoints/test':
                llm = LLM(model=model_name, gpu_memory_utilization=0.95, tokenizer="gpt2", tensor_parallel_size=1, enable_prefix_caching=True)
            else:
                llm = LLM(model=model_name, gpu_memory_utilization=0.95, tensor_parallel_size=1, enable_prefix_caching=True)
            break  # Success!
        except Exception as e:
            print(f"Attempt {attempt + 1} to load model failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.randint(2, 15))
            else:
                raise

    sampling_params = SamplingParams(
        max_tokens=2000,
        n=256,  # 256 samples per prompt
        temperature=0.7
    )

    # Generate responses
    gens = llm.generate(prompts, sampling_params, use_tqdm=True)

    n_prompts = len(prompts)
    n_samples = 256
    scores_matrix = np.zeros((n_prompts, n_samples))

    for i, (output, ground_truth) in enumerate(zip(gens, targets)):
        for j, single_output in enumerate(output.outputs):
            response_text = single_output.text
            _, pred = parse_assistant_output('<think>' + response_text)
            score = score_answer(pred, ground_truth)
            scores_matrix[i, j] = score if score == 1 else 0
            do_print = random.randint(1, 64) == 1
            if do_print:
                print(f"--------------------------------")
                print(f"PROMPT: {prompts[i]}")
                print(f"RESPONSE: {response_text}")
                print(f"PARSED ANSWER: {pred}")
                print(f"GROUND TRUTH: {ground_truth}")
                print(f"SCORE: {score}\n\n")

    # Save the scores matrix
    scores_df = pd.DataFrame(scores_matrix)
    scores_df.to_csv(f'{results_dir}{idx_start}-{idx_end}.csv', index=False)

if __name__ == "__main__":
    main()
