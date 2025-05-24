#!/usr/bin/env python3
"""
Pass@K Result Generation Tool

This tool generates summary CSV files for models and calculates model-level pass@k metrics.
Supports both BBH (BigBench Hard) and MMLU-Pro evaluation tasks.

USAGE:
    # List all available models (auto-detect task type)
    python pass_k_result.py --benchmark /path/to/benchmark/results --list
    
    # Process all models (auto-detect task type)
    python pass_k_result.py --benchmark /path/to/benchmark/results
    
    # Process specific model (auto-detect task type)
    python pass_k_result.py --benchmark /path/to/benchmark/results --model model_name
    
    # Explicitly specify task type (optional)
    python pass_k_result.py --benchmark /path/to/benchmark/results --task bbh --model model_name

EXAMPLES:
    # Process all models (auto-detect from path containing bbh_cot_fewshot_pass16)
    python pass_k_result.py --benchmark ./results/bbh_cot_fewshot_pass16
    
    # Process all models (auto-detect from path containing mmlu_pro_pass16)  
    python pass_k_result.py --benchmark ./results/mmlu_pro_pass16
    
    # Process a specific model
    python pass_k_result.py --benchmark ./results/bbh_cot_fewshot_pass16 --model awesome_7343

OUTPUT FILES:
    For each model, the tool generates:
    - summary_all_tasks.csv: Contains all task results in one file
    - pass_k_metrics.json: Contains pass@k metrics (pass@1, pass@2, pass@4, pass@8, pass@16)

CSV FORMAT:
    task_name,sample_id,attempt_1,attempt_2,...,attempt_16
    Each row represents one sample with binary results (1=correct, 0=incorrect)
"""

import json
import csv
import os
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

# Task configurations
TASK_CONFIGS = {
    'bbh': {
        'dir_pattern': 'bbh_cot_fewshot_pass16',
        'file_prefix': 'samples_bbh_cot_fewshot_',
        'file_suffix': '.jsonl'
    },
    'mmlu_pro': {
        'dir_pattern': 'mmlu_pro_pass16',
        'file_prefix': 'samples_mmlu_pro_',
        'file_suffix': '.jsonl'
    },
    'mmlu_flan': {
        'dir_pattern': 'mmlu_flan_cot_fewshot_pass16',
        'file_prefix': 'samples_mmlu_flan_cot_fewshot_',
        'file_suffix': '.jsonl'
    }
}

def detect_task_type(benchmark_dir):
    """Auto-detect task type from benchmark directory structure"""
    benchmark_path = Path(benchmark_dir)
    
    # Method 1: Check if benchmark path contains known patterns
    benchmark_str = str(benchmark_path).lower()
    for task_type, config in TASK_CONFIGS.items():
        if config['dir_pattern'].lower() in benchmark_str:
            print(f"  Auto-detected task type: {task_type} (from path)")
            return task_type
    
    # Method 2: Scan for models and check their subdirectories
    if benchmark_path.exists():
        for item in benchmark_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                for task_type, config in TASK_CONFIGS.items():
                    task_dir = item / config['dir_pattern']
                    if task_dir.exists():
                        print(f"  Auto-detected task type: {task_type} (from directory scan)")
                        return task_type
    
    # Method 3: Check if benchmark_dir itself is a task directory
    for task_type, config in TASK_CONFIGS.items():
        if benchmark_path.name == config['dir_pattern']:
            parent_dir = benchmark_path.parent
            print(f"  Auto-detected task type: {task_type} (benchmark dir is task dir)")
            return task_type, parent_dir
    
    return None

def extract_task_name(filename, task_type):
    """Extract task name from filename based on task type"""
    config = TASK_CONFIGS.get(task_type)
    if not config:
        return None
        
    if filename.startswith(config['file_prefix']) and filename.endswith(config['file_suffix']):
        task_part = filename[len(config['file_prefix']):-len(config['file_suffix'])]
        parts = task_part.split("_")
        # Find timestamp part (starts with year)
        for i, part in enumerate(parts):
            if len(part) >= 4 and part[:4].isdigit():
                return "_".join(parts[:i])
        return task_part
    return None

def process_jsonl_file(file_path):
    """
    Process a single JSONL file and return a 2D list
    Each row represents a sample, each column represents an attempt (up to 16 attempts)
    """
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                target = str(data.get('target', '')).strip()
                filtered_resps = data.get('filtered_resps', [[]])[0]  # Take first response list
                
                # Create binary array: match=1, no match=0
                binary_row = [line_num - 1]  # Add sample ID as first column
                for resp in filtered_resps:
                    if str(resp).strip() == target:
                        binary_row.append(1)
                    else:
                        binary_row.append(0)
                
                # Pad or truncate to 16 columns (17 total with ID column)
                while len(binary_row) < 17:  # sample_id + 16 attempts
                    binary_row.append(0)
                if len(binary_row) > 17:
                    binary_row = binary_row[:17]
                    
                results.append(binary_row)
                
            except Exception as e:
                print(f"    Warning: Error processing line {line_num}: {e}")
                continue
    
    return results

def pass_at_k(n, c, k):
    """
    Numerically stable script for calculating an unbiased estimate of pass@k.
    
    :param n: total number of samples (attempts)
    :param c: number of correct samples
    :param k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def calculate_model_pass_k(all_tasks_data, k_values=[1, 2, 4, 8, 16]):
    """
    Calculate model-level pass@k metrics using unbiased estimator
    all_tasks_data: dict {task_name: matrix_data}
    """
    # Merge all task data
    all_samples = []
    for task_name, task_data in all_tasks_data.items():
        for row in task_data:
            # Extract attempt results (skip sample_id column)
            attempts = row[1:]
            all_samples.append(attempts)
    
    if not all_samples:
        return {}
    
    total_samples = len(all_samples)
    results = {}
    
    for k in k_values:
        if k <= 16:  # Maximum 16 attempts
            pass_at_k_scores = []
            
            for sample_attempts in all_samples:
                # Count total attempts and correct attempts for this sample
                n = len(sample_attempts)  # Should be 16
                c = sum(sample_attempts)  # Number of correct attempts
                
                # Calculate pass@k for this individual sample
                sample_pass_k = pass_at_k(n, c, k)
                pass_at_k_scores.append(sample_pass_k)
            
            # Average across all samples
            pass_rate = np.mean(pass_at_k_scores)
            results[f"pass@{k}"] = pass_rate
    
    return results

def process_model(benchmark_dir, model_name, task_type):
    """Process all tasks for a single model"""
    print(f"\nProcessing model: {model_name} (task: {task_type})")
    
    config = TASK_CONFIGS.get(task_type)
    if not config:
        print(f"  Error: Unsupported task type '{task_type}'")
        return False
    
    # Find model's checkpoint path
    model_path = Path(benchmark_dir) / model_name / config['dir_pattern']
    
    checkpoint_path = None
    if model_path.exists():
        for item in model_path.iterdir():
            if item.is_dir():
                checkpoint_path = item
                break
    
    if not checkpoint_path:
        print(f"  No checkpoint found for {model_name}")
        return False
    
    print(f"  Checkpoint path: {checkpoint_path}")
    
    # Collect all task data
    all_tasks_data = {}
    task_files = []
    
    # Find all task files
    for file_path in checkpoint_path.iterdir():
        if (file_path.is_file() and 
            file_path.name.startswith(config['file_prefix']) and 
            file_path.suffix == config['file_suffix']):
            task_files.append(file_path)
    
    if not task_files:
        print(f"  No task files found for {model_name}")
        return False
    
    print(f"  Found {len(task_files)} task files")
    
    # Process each task file
    for file_path in sorted(task_files):
        task_name = extract_task_name(file_path.name, task_type)
        if task_name:
            print(f"  Processing task: {task_name}")
            
            try:
                # Process JSONL file
                task_data = process_jsonl_file(file_path)
                all_tasks_data[task_name] = task_data
                
                # Only save summary CSV, no individual task CSVs
                print(f"    Processed task: {task_name} ({len(task_data)} samples)")
                
            except Exception as e:
                print(f"    Error processing {task_name}: {e}")
    
    if not all_tasks_data:
        print(f"  No valid tasks processed for {model_name}")
        return False
    
    # Generate summary CSV file
    summary_csv_file = checkpoint_path / "summary_all_tasks.csv"
    create_summary_csv(all_tasks_data, summary_csv_file)
    
    # Calculate model-level pass@k metrics
    pass_k_metrics = calculate_model_pass_k(all_tasks_data)
    
    # Save pass@k metrics
    metrics_file = checkpoint_path / "pass_k_metrics.json"
    metrics_data = {
        'model_name': model_name,
        'task_type': task_type,
        'total_tasks': len(all_tasks_data),
        'total_samples': sum(len(data) for data in all_tasks_data.values()),
        'tasks': list(all_tasks_data.keys()),
        'pass_k_metrics': pass_k_metrics
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"  Model-level pass@k metrics:")
    for metric, value in pass_k_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    print(f"  Results saved to: {checkpoint_path}")
    print(f"    - Summary CSV: {summary_csv_file} (contains all {len(all_tasks_data)} tasks)")
    print(f"    - Metrics JSON: {metrics_file}")
    
    return True

def create_summary_csv(all_tasks_data, output_file):
    """Create summary CSV file containing all task data"""
    headers = ['task_name', 'sample_id'] + [f'attempt_{i+1}' for i in range(16)]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for task_name, task_data in all_tasks_data.items():
            for row in task_data:
                # Add task_name as first column
                summary_row = [task_name] + row
                writer.writerow(summary_row)
    
    total_samples = sum(len(data) for data in all_tasks_data.values())
    print(f"    Saved summary CSV: {output_file} ({total_samples} total samples)")

def find_all_models(benchmark_dir, task_type):
    """Find all models in benchmark directory for specific task type"""
    benchmark_path = Path(benchmark_dir)
    models = []
    
    config = TASK_CONFIGS.get(task_type)
    if not config:
        return []
    
    for item in benchmark_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            task_dir = item / config['dir_pattern']
            if task_dir.exists():
                # Check if there's a checkpoint directory
                has_checkpoint = any(subitem.is_dir() for subitem in task_dir.iterdir())
                if has_checkpoint:
                    models.append(item.name)
    
    return sorted(models)

def main():
    parser = argparse.ArgumentParser(
        description='Generate Pass@K result files for model evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --benchmark ./results/bbh_cot_fewshot_pass16 --list
  %(prog)s --benchmark ./results/mmlu_pro_pass16 --model awesome_7343
  %(prog)s --benchmark ./results/bbh_cot_fewshot_pass16
        """
    )
    parser.add_argument('-benchmark', '--benchmark', 
                       required=True,
                       help='Benchmark results directory path')
    parser.add_argument('--task', 
                       choices=['bbh', 'mmlu_pro', 'mmlu_flan'],
                       help='Task type: bbh (BigBench Hard), mmlu_pro (MMLU-Pro), or mmlu_flan (MMLU FLAN). If not specified, will auto-detect from path.')
    parser.add_argument('--model', 
                       help='Specific model name to process (optional, processes all models by default)')
    parser.add_argument('--list', 
                       action='store_true',
                       help='List all available models for the detected/specified task')
    
    args = parser.parse_args()
    
    benchmark_dir = args.benchmark
    
    if not Path(benchmark_dir).exists():
        print(f"Error: Benchmark directory {benchmark_dir} does not exist!")
        return
    
    # Auto-detect or use specified task type
    if args.task:
        task_type = args.task
        print(f"Using specified task type: {task_type}")
    else:
        print("Auto-detecting task type...")
        detection_result = detect_task_type(benchmark_dir)
        
        if isinstance(detection_result, tuple):
            # Case where benchmark_dir is the task directory itself
            task_type, benchmark_dir = detection_result
        elif detection_result:
            task_type = detection_result
        else:
            print("Error: Could not auto-detect task type from benchmark directory.")
            print("Please specify --task explicitly:")
            print("  --task bbh (for BigBench Hard)")
            print("  --task mmlu_pro (for MMLU-Pro)")
            return
    
    # Find all models for the specified task
    all_models = find_all_models(benchmark_dir, task_type)
    
    if args.list:
        print(f"Found {len(all_models)} models in {benchmark_dir} for task '{task_type}':")
        for i, model in enumerate(all_models, 1):
            print(f"  {i:2d}. {model}")
        return
    
    if args.model:
        if args.model in all_models:
            models_to_process = [args.model]
        else:
            print(f"Error: Model '{args.model}' not found for task '{task_type}'!")
            print("Available models:")
            for model in all_models[:10]:  # Show first 10
                print(f"  - {model}")
            return
    else:
        models_to_process = all_models
    
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Task type: {task_type}")
    print(f"Found {len(all_models)} total models")
    print(f"Will process {len(models_to_process)} model(s)")
    
    # Process models
    successful_models = 0
    failed_models = 0
    
    for model_name in models_to_process:
        try:
            success = process_model(benchmark_dir, model_name, task_type)
            if success:
                successful_models += 1
            else:
                failed_models += 1
        except Exception as e:
            print(f"  Error processing model {model_name}: {e}")
            failed_models += 1
    
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Task type: {task_type}")
    print(f"Successfully processed: {successful_models} models")
    print(f"Failed to process: {failed_models} models")
    print(f"Total processed: {successful_models + failed_models} models")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 