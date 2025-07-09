import os
import sys
import json
import re
import numpy as np
import pandas as pd
from optparse import OptionParser
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- Helper Functions ---

def load_fewshot_from_json(task, cot_prompts_path):
    """Load the few-shot CoT prompts for the specified task"""
    with open(cot_prompts_path, 'r', encoding='utf-8') as f:
        cot_prompts = json.load(f)
    if task not in cot_prompts:
        raise ValueError(f"Task {task} not found in cot_prompts json at {cot_prompts_path}!")
    return cot_prompts[task]

def build_target(example):
    """Build the target answer for the BBH task"""
    return example['target']

def extract_answer(text):
    """Extract the answer from the model-generated text"""
    match = re.search(r"the answer is ([^\n\.]*)", text)
    if match:
        return match.group(1).strip().strip('.')
    
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if "answer" in line.lower():
            return line.split()[-1].strip('.').strip()
    return ""

def build_prompt(fewshot, example):
    """Build the complete few-shot CoT prompt"""
    return fewshot + '\n\nQ: ' + example['input'] + '\nA: Let\'s think step by step.'

# --- Main Logic ---

def parse_args():
    """Parse the command line arguments"""
    parser = OptionParser()
    parser.add_option("--model_path", dest="model_path", type="str", help="Path to the HuggingFace model.")
    parser.add_option("--model_name", dest="model_name", type="str", help="A friendly name for the model.")
    parser.add_option("--output_base_dir", dest="output_base_dir", type="str", help="The base directory for evaluation results.")
    parser.add_option("--cot_prompts_path", dest="cot_prompts_path", type="str", help="Path to the BBH CoT prompts JSON file.")
    parser.add_option("--tp_size", dest="tp_size", type="int", help="Tensor parallel size for VLLM.")
    parser.add_option("--n_sampling", dest="n_sampling", type="int", help="Number of samples to generate (n for pass@k).")
    
    (options, args) = parser.parse_args()
    if not all(getattr(options, attr) for attr in ['model_path', 'model_name', 'output_base_dir', 'cot_prompts_path', 'tp_size', 'n_sampling']):
        parser.error("All arguments are required.")
    return options

def main():
    """Evaluate all BBH sub-tasks for a single model, and aggregate the results"""
    options = parse_args()
    
    model_dir = os.path.join(options.output_base_dir, options.model_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"--- 🚀 Starting full BBH evaluation for model: {options.model_name} ---")
    print(f"Model Path: {options.model_path}")

    # --- 1. load the VLLM model ---
    try:
        llm = LLM(
            model=options.model_path,
            tensor_parallel_size=options.tp_size,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            dtype="float32"
        )
    except Exception as e:
        print(f"❌ Error initializing VLLM model: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. get all BBH sub-tasks ---
    try:
        with open(options.cot_prompts_path, "r", encoding="utf-8") as f:
            bbh_sub_tasks = list(json.load(f).keys())
        print(f"Found {len(bbh_sub_tasks)} BBH sub-tasks to evaluate.")
    except FileNotFoundError:
        print(f"❌ Error: COT prompts file not found at {options.cot_prompts_path}", file=sys.stderr)
        sys.exit(1)
        
    sampling_params = SamplingParams(
        n=options.n_sampling,
        temperature=0.7,
        max_tokens=512,
    )

    pass1_scores = []
    total_correct_globally = 0
    total_samples_globally = 0
    
    # --- 3. loop through and evaluate each sub-task ---
    for i, task in enumerate(bbh_sub_tasks):
        print(f"\n--- [{i+1}/{len(bbh_sub_tasks)}] Evaluating task: {task} ---")
        
        result_csv_path = os.path.join(model_dir, f"{task}_results.csv")
        if os.path.exists(result_csv_path):
            print(f"⏩ Result file already exists, skipping task: {result_csv_path}")
            # if the file exists, we still need its score for the final aggregation
            summary_json_path = os.path.join(model_dir, f"{task}_summary.json")
            if os.path.exists(summary_json_path):
                with open(summary_json_path, 'r') as f:
                    summary_data = json.load(f)
                    pass1_scores.append(summary_data.get("pass@1", 0))
            continue

        fewshot = load_fewshot_from_json(task, options.cot_prompts_path)
        dataset = load_dataset("lukaemon/bbh", task, split="test", cache_dir="/mnt/sharefs/users/haolong.jia/eval_data", trust_remote_code=True)
        
        prompts = [build_prompt(fewshot, ex) for ex in dataset]
        targets = [build_target(ex) for ex in dataset]
        
        print(f"Loaded {len(prompts)} samples.")
        
        # execute the inference
        gens = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        # process and save the results for the current task
        results_data = []
        correct_count = 0
        for j, (output, ground_truth) in enumerate(zip(gens, targets)):
            response_text = output.outputs[0].text
            pred_answer = extract_answer(response_text)
            gt_answer = ground_truth.strip()
            is_correct = int(pred_answer == gt_answer)
            
            correct_count += is_correct
            results_data.append({
                "prompt": prompts[j], "response": response_text,
                "predicted_answer": pred_answer, "ground_truth": gt_answer,
                "is_correct": is_correct
            })
            
        task_pass_at_1 = correct_count / len(prompts) if prompts else 0
        pass1_scores.append(task_pass_at_1) # still keep the score for each task, for possible analysis
        
        # accumulate the global correct count and sample count
        total_correct_globally += correct_count
        total_samples_globally += len(prompts)
        
        # save the detailed CSV
        pd.DataFrame(results_data).to_csv(result_csv_path, index=False, encoding='utf-8')
        print(f"✅ Results saved to {result_csv_path}")

        # save the summary_json for the task
        summary_json_path = os.path.join(model_dir, f"{task}_summary.json")
        with open(summary_json_path, 'w') as f:
            json.dump({
                "pass@1": task_pass_at_1,
                "total_correct": correct_count,
                "total_samples": len(prompts)
            }, f, indent=2)
        print(f"✅ Summary saved to {summary_json_path}")


    # --- 4. after all tasks are completed, do the self-aggregation ---
    if total_samples_globally > 0:
        # use the weighted average (Micro Average) to calculate the final score
        weighted_average_pass1 = total_correct_globally / total_samples_globally
        
        model_summary = {
            "pass@1": weighted_average_pass1,
            "total_correct": total_correct_globally,
            "total_samples": total_samples_globally
        }
        
        model_result_path = os.path.join(model_dir, "result.json")
        with open(model_result_path, 'w') as f:
            json.dump(model_summary, f, indent=2)
            
        print(f"\n--- 📊 Final Summary for {options.model_name} ---")
        print(f"Weighted Average Pass@1 across {len(pass1_scores)} BBH tasks: {weighted_average_pass1:.4f}")
        print(f"Total Correct: {total_correct_globally}, Total Samples: {total_samples_globally}")
        print(f"✅ Final model summary saved to {model_result_path}")
    else:
        print("\n--- ⚠️ No tasks were evaluated, skipping final summary generation. ---")

    # --- 5. clean up the model ---
    del llm
    import gc; gc.collect()
    print(f"\n--- 🎉 Finished full BBH evaluation for model: {options.model_name} ---")


if __name__ == "__main__":
    main()
