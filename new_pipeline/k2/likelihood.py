#!/usr/bin/env python3
"""
Likelihood evaluation script for K2+ 70B models using lm-evaluation-harness.
This script evaluates a single model on a specified likelihood task.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Task configurations with their metrics and few-shot settings
TASK_CONFIGS = {
    'drop': {
        'metric_key': 'f1,none',
        'result_key': 'drop',
        'num_fewshot': 0
    },
    'arc_easy': {
        'metric_key': 'acc_norm,none',
        'result_key': 'arc_easy',
        'num_fewshot': 0
    },
    'arc_challenge': {
        'metric_key': 'acc_norm,none',
        'result_key': 'arc_challenge',
        'num_fewshot': 25
    },
    'hellaswag': {
        'metric_key': 'acc_norm,none',
        'result_key': 'hellaswag',
        'num_fewshot': 0
    },
    'piqa': {
        'metric_key': 'acc_norm,none',
        'result_key': 'piqa',
        'num_fewshot': 0
    },
    'winogrande': {
        'metric_key': 'acc_norm,none',
        'result_key': 'winogrande',
        'num_fewshot': 5
    },
    'triviaqa': {
        'metric_key': 'exact_match,remove_whitespace',
        'result_key': 'triviaqa',
        'num_fewshot': 5
    },
    'nq_open': {
        'metric_key': 'exact_match,remove_whitespace',
        'result_key': 'nq_open',
        'num_fewshot': 0
    },
    'commonsense_qa': {
        'metric_key': 'acc_norm,none',
        'result_key': 'commonsense_qa',
        'num_fewshot': 0
    },
    'agieval': {
        'metric_key': 'acc_norm,none',
        'result_key': 'agieval_en',
        'num_fewshot': 0
    },
    'openbookqa': {
        'metric_key': 'acc_norm,none',
        'result_key': 'openbookqa',
        'num_fewshot': 0
    },
    'social_iqa': {
        'metric_key': 'acc_norm,none',
        'result_key': 'social_iqa',
        'num_fewshot': 0
    },
    'truthfulqa_mc2': {
        'metric_key': 'acc_norm,none',
        'result_key': 'truthfulqa_mc2',
        'num_fewshot': 0
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single K2+ model on likelihood tasks")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--output_base_dir', type=str, required=True, help='Base output directory')
    parser.add_argument('--tp_size', type=int, default=8, help='Tensor parallel size')
    parser.add_argument('--task_name', type=str, required=True, help='Task name to evaluate')
    parser.add_argument('--n_sampling', type=int, default=1, help='Number of samples (not used for likelihood tasks)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate task
    if args.task_name not in TASK_CONFIGS:
        print(f"Error: Unknown task '{args.task_name}'")
        print(f"Supported tasks: {', '.join(TASK_CONFIGS.keys())}")
        sys.exit(1)
    
    task_config = TASK_CONFIGS[args.task_name]
    
    # Create output directory
    model_output_dir = os.path.join(args.output_base_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"🐎 Evaluating model: {args.model_name}")
    print(f"🎯 Task: {args.task_name}")
    print(f"💾 Output directory: {model_output_dir}")
    
    # Build lm_eval command
    lm_eval_cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={args.model_path},tensor_parallel_size={args.tp_size},gpu_memory_utilization=0.95",
        "--tasks", args.task_name,
        "--output_path", model_output_dir,
        "--batch_size", "auto",
        "--log_samples",
        "--num_fewshot", str(task_config['num_fewshot'])
    ]
    
    # Add trust_remote_code for specific tasks
    if args.task_name in ['social_iqa', 'arc_challenge', 'winogrande', 'triviaqa']:
        lm_eval_cmd.append("--trust_remote_code")
    
    # Run evaluation
    print(f"🚀 Running command: {' '.join(lm_eval_cmd)}")
    try:
        subprocess.run(lm_eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running lm_eval: {e}")
        sys.exit(1)
    
    # Post-process results
    print(f"🚀 Post-processing results for {args.model_name}...")
    
    # Find the intermediate subdirectory created by lm_eval
    subdirs = [d for d in os.listdir(model_output_dir) if os.path.isdir(os.path.join(model_output_dir, d))]
    if not subdirs:
        print("Error: No subdirectory found from lm_eval output")
        sys.exit(1)
    
    subdir = os.path.join(model_output_dir, subdirs[0])
    
    # Move and rename results_*.json
    result_files = [f for f in os.listdir(subdir) if f.startswith('results_') and f.endswith('.json')]
    if result_files:
        src_path = os.path.join(subdir, result_files[0])
        dst_path = os.path.join(model_output_dir, 'result.json')
        os.rename(src_path, dst_path)
        
        # Verify the results are readable and contain expected data
        with open(dst_path, 'r') as f:
            full_results = json.load(f)
        
        # Extract metric for logging
        metric_key = task_config['metric_key']
        result_key = task_config['result_key']
        
        if args.task_name == 'agieval':
            metric_value = full_results['results']['agieval'][metric_key]
        else:
            metric_value = full_results['results'][result_key][metric_key]
        
        print(f"Result: {metric_key} = {metric_value}")
    
    # Handle sample files
    if args.task_name == 'agieval':
        # Merge all samples_agieval_*.jsonl
        sample_files = [f for f in os.listdir(subdir) if f.startswith('samples_agieval_') and f.endswith('.jsonl')]
        if sample_files:
            merged_path = os.path.join(model_output_dir, 'sample.jsonl')
            with open(merged_path, 'w') as outfile:
                for sample_file in sample_files:
                    with open(os.path.join(subdir, sample_file), 'r') as infile:
                        outfile.write(infile.read())
    else:
        sample_files = [f for f in os.listdir(subdir) if f.startswith('samples_') and f.endswith('.jsonl')]
        if sample_files:
            src_path = os.path.join(subdir, sample_files[0])
            dst_path = os.path.join(model_output_dir, 'sample.jsonl')
            os.rename(src_path, dst_path)
    
    # Clean up intermediate directory
    import shutil
    shutil.rmtree(subdir)
    
    print(f"🎉 Evaluation completed for {args.model_name}")

if __name__ == "__main__":
    main()
