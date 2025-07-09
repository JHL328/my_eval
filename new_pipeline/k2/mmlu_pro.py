import os
import sys
import json
import re
import numpy as np
import pandas as pd
from optparse import OptionParser
from vllm import LLM, SamplingParams
import gc

# --- Helper Functions (adapted from evaluate_mmlu_pro_pass16.py) ---

def load_prompts_from_json(subject, prompts_path):
    """Load all prompts for a given subject from the MMLU-Pro JSON file."""
    with open(prompts_path, 'r', encoding='utf-8') as f:
        all_prompts = json.load(f)
    if subject not in all_prompts:
        raise ValueError(f"Subject {subject} not found in prompts json at {prompts_path}!")
    return all_prompts[subject]

def build_target(example):
    """Build the target answer string (a single uppercase letter)."""
    return example['answer'].strip().upper()

def extract_answer(text):
    """Extract the answer choice, using fallbacks for MMLU-Pro."""
    # Priority: match 'the answer is (X)' or 'the answer is X' for A-P
    match = re.search(r"[Tt]he answer is[\s\(]*([A-P])[\)\s\.]*", text)
    if match:
        return match.group(1).strip().upper()
    
    # Fallback: look for a single capital letter (A-P) in the last line
    lines = text.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r"^([A-P])$", line.strip()) # Exact match for a single letter line
        if m:
            return m.group(1)
            
    return ""

# --- Main Logic ---

def parse_args():
    """Parse command-line arguments."""
    parser = OptionParser()
    parser.add_option("--model_path", dest="model_path", type="str", help="Path to the HuggingFace model.")
    parser.add_option("--model_name", dest="model_name", type="str", help="A friendly name for the model.")
    parser.add_option("--output_base_dir", dest="output_base_dir", type="str", help="The base directory for evaluation results.")
    parser.add_option("--prompts_path", dest="prompts_path", type="str", help="Path to the MMLU-Pro prompts JSON file.")
    parser.add_option("--tp_size", dest="tp_size", type="int", help="Tensor parallel size for VLLM.")
    parser.add_option("--n_sampling", dest="n_sampling", type="int", help="Number of samples to generate (n for pass@k).")
    
    (options, args) = parser.parse_args()
    if not all(getattr(options, attr) for attr in ['model_path', 'model_name', 'output_base_dir', 'prompts_path', 'tp_size', 'n_sampling']):
        parser.error("All arguments are required.")
    return options

def main():
    """Evaluate all MMLU-Pro subjects for a single model and aggregate the results."""
    options = parse_args()
    
    model_dir = os.path.join(options.output_base_dir, options.model_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"--- 🚀 Starting full MMLU-Pro evaluation for model: {options.model_name} ---")
    print(f"Model Path: {options.model_path}")

    # --- 1. Load VLLM model once ---
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

    # --- 2. Get all MMLU-Pro subjects ---
    try:
        with open(options.prompts_path, "r", encoding="utf-8") as f:
            mmlu_pro_subjects = list(json.load(f).keys())
        print(f"Found {len(mmlu_pro_subjects)} MMLU-Pro subjects to evaluate.")
    except FileNotFoundError:
        print(f"❌ Error: MMLU-Pro prompts file not found at {options.prompts_path}", file=sys.stderr)
        sys.exit(1)
        
    sampling_params = SamplingParams(
        n=options.n_sampling,
        temperature=0.7,
        max_tokens=1024,
        stop=["</s>"] # Important for MMLU-Pro
    )

    total_correct_globally = 0
    total_samples_globally = 0
    
    # --- 3. Loop through and evaluate each subject ---
    for i, subject in enumerate(mmlu_pro_subjects):
        print(f"\n--- [{i+1}/{len(mmlu_pro_subjects)}] Evaluating subject: {subject} ---")
        
        result_csv_path = os.path.join(model_dir, f"{subject}_results.csv")
        summary_json_path = os.path.join(model_dir, f"{subject}_summary.json")

        if os.path.exists(result_csv_path):
            print(f"⏩ Result file already exists, skipping task: {result_csv_path}")
            if os.path.exists(summary_json_path):
                with open(summary_json_path, 'r') as f:
                    summary_data = json.load(f)
                    total_correct_globally += summary_data.get("total_correct", 0)
                    total_samples_globally += summary_data.get("total_samples", 0)
            continue

        subject_prompts_data = load_prompts_from_json(subject, options.prompts_path)
        prompts = [ex['prompt'] for ex in subject_prompts_data]
        targets = [build_target(ex) for ex in subject_prompts_data]
        
        print(f"Loaded {len(prompts)} samples.")
        
        gens = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        results_data = []
        correct_count = 0
        for j, (output, ground_truth) in enumerate(zip(gens, targets)):
            response_text = output.outputs[0].text
            pred_answer = extract_answer(response_text)
            is_correct = int(pred_answer == ground_truth)
            
            correct_count += is_correct
            results_data.append({
                "prompt": prompts[j], "response": response_text,
                "predicted_answer": pred_answer, "ground_truth": ground_truth,
                "is_correct": is_correct
            })
            
        task_pass_at_1 = correct_count / len(prompts) if prompts else 0
        total_correct_globally += correct_count
        total_samples_globally += len(prompts)
        
        pd.DataFrame(results_data).to_csv(result_csv_path, index=False, encoding='utf-8')
        print(f"✅ Results saved to {result_csv_path}")

        with open(summary_json_path, 'w') as f:
            json.dump({
                "pass@1": task_pass_at_1,
                "total_correct": correct_count,
                "total_samples": len(prompts)
            }, f, indent=2)
        print(f"✅ Summary saved to {summary_json_path}")

    # --- 4. Self-aggregation after all subjects ---
    if total_samples_globally > 0:
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
        print(f"Weighted Average Pass@1 across {len(mmlu_pro_subjects)} MMLU-Pro subjects: {weighted_average_pass1:.4f}")
        print(f"Total Correct: {total_correct_globally}, Total Samples: {total_samples_globally}")
        print(f"✅ Final model summary saved to {model_result_path}")
    else:
        print("\n--- ⚠️ No subjects were evaluated, skipping final summary generation. ---")

    # --- 5. Clean up model ---
    del llm
    gc.collect()
    print(f"\n--- 🎉 Finished full MMLU-Pro evaluation for model: {options.model_name} ---")

if __name__ == "__main__":
    main()
