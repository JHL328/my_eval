#!/usr/bin/env python3
"""
Reprocess existing GSM8K evaluation results using math_verify library.
This script reads existing gsm8k_eval_results.json files and reprocesses them
with math_verify for more robust mathematical answer verification.
"""

import json
import os
from tqdm import tqdm
from math_verify import parse, verify
from typing import Dict, Any, Optional

# Directory containing the GSM8K results
SOURCE_DIR = "/mnt/sharefs/users/haolong.jia/result/gsm8k_pass16"

def parse_answer_with_verify(text: Optional[str]) -> Optional[Any]:
    """Extract answer from text using math_verify's parse function."""
    if text is None:
        return None
    
    # Convert to string if not already
    text = str(text)
    
    # Directly use math_verify's parse function to extract answer
    # math_verify.parse can handle full text and extract the answer automatically
    try:
        return parse(text)
    except:
        # If math_verify fails, return None
        return None

def compare_answers(gold, pred) -> bool:
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

def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k metric."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def convert_to_serializable(obj):
    """Convert math_verify parsed objects to JSON-serializable format."""
    if obj is None:
        return ""
    elif isinstance(obj, (list, tuple)):
        # math_verify returns [value, string_representation]
        return str(obj[0]) if len(obj) > 0 else ""
    else:
        return str(obj)

def process_single_model(model_dir: str, model_name: str) -> Optional[Dict[str, float]]:
    """Process a single model's results and return pass@k metrics."""
    
    # Check if original results exist
    original_file = os.path.join(model_dir, "gsm8k_eval_results.json")
    if not os.path.exists(original_file):
        return None
    
    # Check if math-verify results already exist
    output_file = os.path.join(model_dir, "math-verify.json")
    if os.path.exists(output_file):
        print(f" ⚠️  Skipping {model_name} (math-verify.json already exists)")
        # Load existing results to calculate pass@k
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        # Calculate pass@k metrics
        total_samples = len(results)
        correct_samples = sum(1 for item in results if item.get("pass@16", False))
        
        passk_metrics = {}
        for k in [1, 2, 4, 8, 16]:
            if k <= 16:  # We have 16 generations
                # Count correct in first k generations
                correct_in_k = 0
                for item in results:
                    if any(item["exact_match"][:k]):
                        correct_in_k += 1
                passk_metrics[f"pass@{k}"] = correct_in_k / total_samples if total_samples > 0 else 0.0
        
        return passk_metrics
    
    # Load original results
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    
    # Process each sample with progress bar
    processed_results = []
    sample_pbar = tqdm(original_data, desc=f"  Processing samples", leave=False)
    
    for item in sample_pbar:
        # Extract ground truth answer using math_verify
        gold_parsed = parse_answer_with_verify(item["gold"])
        
        # Process each generation
        parsed_answers = []
        exact_matches = []
        
        for generation in item["generations"]:
            pred_parsed = parse_answer_with_verify(generation)
            parsed_answers.append(convert_to_serializable(pred_parsed))
            exact_matches.append(compare_answers(gold_parsed, pred_parsed))
        
        # Check if any generation is correct (for pass@16)
        pass_at_16 = any(exact_matches)
        
        # Create new result item with math_verify processing
        result_item = {
            "question": item["question"],
            "gold": item["gold"],
            "generations": item["generations"],
            "parsed": parsed_answers,
            "exact_match": exact_matches,
            "pass@16": pass_at_16
        }
        
        processed_results.append(result_item)
    
    sample_pbar.close()
    
    # Save processed results
    with open(output_file, 'w') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    
    # Calculate pass@k metrics
    total_samples = len(processed_results)
    correct_samples = sum(1 for item in processed_results if item["pass@16"])
    
    passk_metrics = {}
    for k in [1, 2, 4, 8, 16]:
        if k <= 16:  # We have 16 generations
            # Count correct in first k generations
            correct_in_k = 0
            for item in processed_results:
                if any(item["exact_match"][:k]):
                    correct_in_k += 1
            passk_metrics[f"pass@{k}"] = correct_in_k / total_samples if total_samples > 0 else 0.0
    
    print(f"  ✅ Processed {model_name}: {correct_samples}/{total_samples} pass@16 ({passk_metrics['pass@16']:.2%})")
    
    return passk_metrics

def main():
    """Main function to process all models."""
    print("=" * 80)
    print("GSM8K Math_Verify Reprocessing Tool")
    print("=" * 80)
    print(f"\nSource directory: {SOURCE_DIR}")
    print("Output files: math-verify.json (per model), math-verify-passk.json (global)")
    print("-" * 80)
    
    # Find all model directories
    model_dirs = []
    for item in os.listdir(SOURCE_DIR):
        item_path = os.path.join(SOURCE_DIR, item)
        if os.path.isdir(item_path):
            # Check if it has gsm8k_eval_results.json
            if os.path.exists(os.path.join(item_path, "gsm8k_eval_results.json")):
                model_dirs.append((item_path, item))
    
    if not model_dirs:
        print("🚫 No model directories with gsm8k_eval_results.json found!")
        return
    
    print(f"\n🚀 Found {len(model_dirs)} models to process")
    
    # Process each model
    all_passk_results = {}
    
    model_pbar = tqdm(model_dirs, desc="Processing models")
    for model_dir, model_name in model_pbar:
        model_pbar.set_description(f"Processing {model_name}")
        
        # Process the model
        passk_metrics = process_single_model(model_dir, model_name)
        
        if passk_metrics:
            all_passk_results[model_name] = passk_metrics
    
    model_pbar.close()
    
    # Save global pass@k results
    passk_output_file = os.path.join(SOURCE_DIR, "math-verify-passk.json")
    with open(passk_output_file, 'w') as f:
        json.dump(all_passk_results, f, indent=2)
    
    print("-" * 80)
    print(f"✅ Processing complete!")
    print(f"✅ Pass@k results saved to: {passk_output_file}")
    print(f"✅ Processed {len(all_passk_results)} models successfully")
    print("=" * 80)

if __name__ == "__main__":
    main()