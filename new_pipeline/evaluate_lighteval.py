#!/usr/bin/env python
"""
Evaluation manager script for MMLU Redux and GPQA Diamond using lighteval.
Follows the pattern of evaluate_gsm8k.py for consistency.
"""

import os
import json
import sys
import time
import re
import subprocess
import argparse
import yaml
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import get_model_map_by_type

# =====================
# Task Configurations
# =====================

# MMLU Redux subjects (57 total)
MMLU_REDUX_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions"
]

TASK_CONFIGS = {
    "mmlu_redux": {
        "BASE_OUT": "/mnt/sharefs/users/haolong.jia/result/mmlu_redux_sft",
        "TASK_LIST": ",".join([f"lighteval|mmlu_redux_2:{s}|0" for s in MMLU_REDUX_SUBJECTS]),
        "GPUS_PER_TASK": 1,
        "TIME_LIMIT": "8:00:00",
        "PARTITION": "lowprio",
        "QOS": "lowprio",
        "MEM": "100G",
        "CONDA_ENV": "harness-eval",
        "LIGHTEVAL_PATH": "/mnt/weka/home/haolong.jia/eval/RL-eval/lighteval",
    },
    "gpqa_diamond": {
        "BASE_OUT": "/mnt/sharefs/users/haolong.jia/result/gpqa_diamond_sft",
        "TASK_LIST": "lighteval|gpqa:diamond|0",
        "GPUS_PER_TASK": 1,
        "TIME_LIMIT": "4:00:00",
        "PARTITION": "lowprio",
        "QOS": "lowprio",
        "MEM": "100G",
        "CONDA_ENV": "harness-eval",
        "LIGHTEVAL_PATH": "/mnt/weka/home/haolong.jia/eval/RL-eval/lighteval",
    }
}

# =====================
# Utility Functions
# =====================

def is_job_completed(model_out_dir: str) -> bool:
    """Check if a job has already completed by looking for results.json."""
    results_file = os.path.join(model_out_dir, "results.json")
    return os.path.exists(results_file)

def parse_lighteval_results(results_path: str) -> Dict:
    """Parse lighteval results.json and extract metrics."""
    with open(results_path, 'r') as f:
        data = json.load(f)

    metrics = {}

    # Extract results from lighteval format
    if "results" in data:
        for task_name, task_results in data["results"].items():
            if isinstance(task_results, dict):
                # Get accuracy or other metrics
                if "accuracy" in task_results:
                    metrics[task_name] = task_results["accuracy"]
                elif "acc" in task_results:
                    metrics[task_name] = task_results["acc"]
                elif "exact_match" in task_results:
                    metrics[task_name] = task_results["exact_match"]

    return metrics

def write_metrics_file(model_out_dir: str, metrics: Dict) -> None:
    """Write formatted metrics to metrics.txt."""
    metrics_path = os.path.join(model_out_dir, "metrics.txt")

    with open(metrics_path, 'w') as f:
        if "mmlu_redux" in model_out_dir:
            # For MMLU Redux, calculate average across subjects
            subject_scores = [v for k, v in metrics.items() if "mmlu_redux" in k]
            if subject_scores:
                avg_score = sum(subject_scores) / len(subject_scores)
                f.write(f"average_accuracy: {avg_score:.4f}\n")
                f.write(f"num_subjects: {len(subject_scores)}\n\n")

            # Write individual subject scores
            f.write("=== Subject Scores ===\n")
            for task_name, score in sorted(metrics.items()):
                subject = task_name.replace("lighteval|mmlu_redux_2:", "").replace("|0", "")
                f.write(f"{subject}: {score:.4f}\n")
        else:
            # For GPQA Diamond
            for task_name, score in metrics.items():
                f.write(f"accuracy: {score:.4f}\n")

def load_yaml_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_model_config_file(base_config: Dict, model_path: str, output_dir: str) -> str:
    """Create a model-specific config file from base config."""
    config = deepcopy(base_config)

    # Create model-specific config
    model_config = {
        'model_parameters': config['model_parameters'].copy()
    }

    # Add model path
    model_config['model_parameters']['model_name'] = model_path

    # Ensure generation_parameters exists
    if 'generation_parameters' not in model_config['model_parameters']:
        model_config['model_parameters']['generation_parameters'] = {}

    # Write config file
    config_file = os.path.join(output_dir, "model_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    return config_file

def submit_single_model_job(
    task_name: str,
    task_config: Dict,
    model_path: str,
    model_name: str,
    model_out_dir: str,
    reforce: bool,
    yaml_config: Optional[Dict] = None
) -> str:
    """Submit a SLURM job for a single model evaluation."""

    # Check if already completed
    if not reforce and is_job_completed(model_out_dir):
        print(f"Skipping {model_name} - results already exist")
        return None

    os.makedirs(model_out_dir, exist_ok=True)

    # Create model config file (yaml_config should always be provided now)
    if yaml_config:
        model_config_file = create_model_config_file(yaml_config, model_path, model_out_dir)
        # Build task list from config
        if 'subjects' in yaml_config.get('task', {}):
            task_list = ",".join([f"lighteval|mmlu_redux_2:{s}|0" for s in yaml_config['task']['subjects']])
        else:
            task_list = yaml_config.get('task', {}).get('task_list', task_config.get('TASK_LIST', ''))
    else:
        # Fallback - create minimal config
        minimal_config = {
            'model_parameters': {
                'dtype': 'auto',
                'trust_remote_code': True,
                'tensor_parallel_size': 1,
                'gpu_memory_utilization': 0.9,
                'max_model_length': 8192,
                'seed': 1234,
                'generation_parameters': {
                    'temperature': 0.0,
                    'max_new_tokens': 1024,
                    'seed': 42
                }
            }
        }
        model_config_file = create_model_config_file(minimal_config, model_path, model_out_dir)
        task_list = task_config.get('TASK_LIST', '')

    job_name = f"{task_name}_{model_name}"
    job_script = os.path.join(model_out_dir, f"{job_name}.sh")

    # Get SLURM config from YAML or use defaults
    if yaml_config:
        slurm_config = yaml_config.get('slurm', {})
        env_config = yaml_config.get('environment', {})
        dataset_config = yaml_config.get('dataset', {})
        chat_config = yaml_config.get('chat_template', {})

        gpus = slurm_config.get('gpus_per_task', 1)
        cpus = slurm_config.get('cpus_per_task', 16)
        time_limit = slurm_config.get('time_limit', '8:00:00')
        partition = slurm_config.get('partition', 'lowprio')
        qos = slurm_config.get('qos', 'lowprio')
        mem = slurm_config.get('mem', '100G')
        conda_env = env_config.get('conda_env', 'harness-eval')
        lighteval_path = env_config.get('lighteval_path', '/mnt/weka/home/haolong.jia/eval/RL-eval/lighteval')
        loading_processes = dataset_config.get('loading_processes', 8)
        max_samples = dataset_config.get('max_samples', -1)
        use_chat_template = chat_config.get('use_chat_template', True)
    else:
        gpus = task_config['GPUS_PER_TASK']
        cpus = 16
        time_limit = task_config['TIME_LIMIT']
        partition = task_config['PARTITION']
        qos = task_config['QOS']
        mem = task_config['MEM']
        conda_env = task_config['CONDA_ENV']
        lighteval_path = task_config['LIGHTEVAL_PATH']
        loading_processes = 8
        max_samples = -1
        use_chat_template = True

    # Create job script
    with open(job_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={model_out_dir}/slurm.out
#SBATCH --error={model_out_dir}/slurm.err
#SBATCH --gres=gpu:{gpus}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --mem={mem}

# Activate conda environment
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate {conda_env}

# Change to lighteval directory
cd {lighteval_path}

# Run lighteval with vLLM backend
""")

        # lighteval vllm expects: MODEL_ARGS TASKS as positional arguments
        if model_config_file:
            # Use YAML config file
            f.write(f"""python -m lighteval vllm \\
    "{model_config_file}" \\
    "{task_list}" \\
    --output-dir="{model_out_dir}" \\
    --save-details \\
    --dataset-loading-processes={loading_processes} \\
    --max-samples={max_samples}""")
        else:
            # Fallback: direct model args (should rarely happen now)
            f.write(f"""python -m lighteval vllm \\
    "model_name={model_path},dtype=auto,trust_remote_code=true,tensor_parallel_size=1" \\
    "{task_list}" \\
    --output-dir="{model_out_dir}" \\
    --save-details \\
    --dataset-loading-processes={loading_processes} \\
    --max-samples={max_samples}""")

        # Add the post-processing script
        f.write(f"""

# Post-process: Find and move the results file from nested directory structure
echo "Post-processing results..."

# Find the actual results JSON file (lighteval creates it in a nested path)
RESULTS_FILE=$(find "{model_out_dir}/results" -name "results_*.json" -type f 2>/dev/null | head -1)

if [ -n "$RESULTS_FILE" ]; then
    echo "Found results file: $RESULTS_FILE"
    # Copy to the expected location
    cp "$RESULTS_FILE" "{model_out_dir}/results.json"
    echo "Copied to: {model_out_dir}/results.json"
else
    echo "Warning: No results file found in {model_out_dir}/results/"
fi

# Process results to generate metrics.txt
python -c "
import json
import os
import glob

# First try the standard location
results_path = '{model_out_dir}/results.json'

# If not found, search for it in the nested structure
if not os.path.exists(results_path):
    pattern = '{model_out_dir}/results/**/results_*.json'
    files = glob.glob(pattern, recursive=True)
    if files:
        results_path = files[0]
        print(f'Found results at: {{results_path}}')

if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)

    metrics = {{}}
    if 'results' in data:
        for task, result in data['results'].items():
            if isinstance(result, dict):
                # For GPQA, look for pass@k metrics
                if 'gpqa' in '{task_name}'.lower():
                    for metric_name, metric_value in result.items():
                        if 'pass@k' in metric_name.lower() or 'pass_at_k' in metric_name.lower():
                            metrics[task] = metric_value
                            break
                # For other tasks, look for accuracy
                else:
                    for metric_name, metric_value in result.items():
                        if 'accuracy' in metric_name.lower() or 'acc' in metric_name.lower() or 'exact_match' in metric_name.lower():
                            metrics[task] = metric_value
                            break

    # Write metrics.txt
    metrics_path = '{model_out_dir}/metrics.txt'
    with open(metrics_path, 'w') as f:
        if 'mmlu_redux' in '{task_name}':
            scores = list(metrics.values())
            if scores:
                avg = sum(scores) / len(scores)
                f.write(f'average_accuracy: {{avg:.4f}}\\n')
                f.write(f'num_subjects: {{len(scores)}}\\n\\n')
            for task, score in sorted(metrics.items()):
                subject = task.split(':')[-1].replace('|0', '')
                f.write(f'{{subject}}: {{score:.4f}}\\n')
        elif 'gpqa' in '{task_name}'.lower():
            # For GPQA, write pass@1 score
            for task, score in metrics.items():
                f.write(f'pass@1: {{score:.4f}}\\n')
        else:
            for task, score in metrics.items():
                f.write(f'accuracy: {{score:.4f}}\\n')

    print(f'Metrics saved to {{metrics_path}}')
else:
    print(f'Results file not found')
"
""")

    # Submit job
    try:
        result = subprocess.check_output(f"sbatch {job_script}", shell=True, text=True)
        match = re.search(r'Submitted batch job (\d+)', result)
        if match:
            job_id = match.group(1)
            print(f"✓ Submitted job for {model_name} (Job ID: {job_id})")
            return job_id
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job for {model_name}: {e}")
        return None

def wait_for_jobs(job_ids: List[str]) -> None:
    """Wait for all submitted jobs to complete."""
    if not job_ids:
        return

    print(f"\n⏳ Waiting for {len(job_ids)} jobs to complete...")

    while job_ids:
        time.sleep(30)

        # Check job status
        try:
            job_ids_str = ",".join(job_ids)
            squeue_output = subprocess.check_output(
                f"squeue -h -j {job_ids_str} -o '%i %t'",
                shell=True,
                text=True
            )

            running_jobs = set()
            for line in squeue_output.strip().split('\n'):
                if line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id = parts[0]
                        status = parts[1]
                        if status in ['R', 'PD', 'CG']:
                            running_jobs.add(job_id)

            job_ids = list(running_jobs)

            if job_ids:
                print(f"📊 {len(job_ids)} jobs still running/pending...")
        except subprocess.CalledProcessError:
            # All jobs completed
            break

    print("✅ All jobs completed!")

def aggregate_results(task_config: Dict, model_map: Dict) -> None:
    """Aggregate results from all models and create summary."""
    base_out = task_config["BASE_OUT"]
    summary = {}

    for _, model_name in model_map.items():
        model_out_dir = os.path.join(base_out, model_name)
        metrics_file = os.path.join(model_out_dir, "metrics.txt")

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                # Extract metrics: average_accuracy for MMLU, pass@1 for GPQA, or accuracy
                for line in lines:
                    if "average_accuracy:" in line:
                        score = float(line.split(":")[-1].strip())
                        summary[model_name] = {"average_accuracy": score}
                        break
                    elif "pass@1:" in line:
                        score = float(line.split(":")[-1].strip())
                        summary[model_name] = {"pass@1": score}
                        break
                    elif "accuracy:" in line:
                        score = float(line.split(":")[-1].strip())
                        summary[model_name] = {"accuracy": score}
                        break

    # Write summary
    summary_path = os.path.join(base_out, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Summary saved to {summary_path}")
    print("\n=== Results Summary ===")

    # Sort by score value (handle nested dict structure)
    def get_score(item):
        model_name, metrics = item
        if isinstance(metrics, dict):
            # Get the first metric value
            return next(iter(metrics.values()))
        return 0

    for model, metrics in sorted(summary.items(), key=get_score, reverse=True):
        if isinstance(metrics, dict):
            metric_name = next(iter(metrics.keys()))
            score = metrics[metric_name]
            print(f"{model}: {score:.4f} ({metric_name})")
        else:
            print(f"{model}: {metrics}")

# =====================
# Main Entry Point
# =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models using lighteval")
    parser.add_argument("--task", type=str, required=False,
                       choices=["mmlu_redux", "gpqa_diamond"],
                       help="Task to evaluate (can be overridden by config)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file (auto-loads default if not provided)")
    parser.add_argument("--type", type=str, default="sft",
                       choices=["base", "sft"],
                       help="Model type to evaluate")
    parser.add_argument("--submit_jobs", action="store_true",
                       help="Submit SLURM jobs for all models")
    parser.add_argument("--reforce", action="store_true",
                       help="Force re-evaluation even if results exist")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Single model path for testing")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Single model name for testing")
    return parser.parse_args()

def main():
    args = parse_args()

    # Always use YAML config - either provided or default
    yaml_config = None

    if args.config and isinstance(args.config, str) and args.config != 'True':
        # User provided a specific config file path
        yaml_config = load_yaml_config(args.config)
        # Get task name from config
        if 'task' in yaml_config and 'name' in yaml_config['task']:
            args.task = yaml_config['task']['name']
    else:
        # Load default config based on task
        if not args.task:
            print("Error: --task is required when not using a custom config file")
            sys.exit(1)

        config_file = f"configs/{args.task}_config.yaml"
        config_path = os.path.join(os.path.dirname(__file__), config_file)

        if os.path.exists(config_path):
            yaml_config = load_yaml_config(config_path)
            print(f"Using default config: {config_file}")
        else:
            print(f"Warning: No config file found at {config_path}, using fallback")
            yaml_config = None

    # Set task_config from yaml or fallback
    if yaml_config and 'task' in yaml_config and 'output_base' in yaml_config['task']:
        task_config = {'BASE_OUT': yaml_config['task']['output_base']}
    else:
        task_config = TASK_CONFIGS.get(args.task, {'BASE_OUT': f'/mnt/sharefs/users/haolong.jia/result/{args.task}'})

    if args.submit_jobs:
        # Get model map
        model_map = get_model_map_by_type(args.type)

        print(f"\n🚀 Evaluating {len(model_map)} models on {args.task}")
        print(f"Output directory: {task_config['BASE_OUT']}")

        os.makedirs(task_config['BASE_OUT'], exist_ok=True)

        # Submit jobs for all models
        job_ids = []
        for model_path, model_name in model_map.items():
            model_out_dir = os.path.join(task_config['BASE_OUT'], model_name)
            job_id = submit_single_model_job(
                args.task,
                task_config,
                model_path,
                model_name,
                model_out_dir,
                args.reforce,
                yaml_config
            )
            if job_id:
                job_ids.append(job_id)
            time.sleep(0.2)  # Small delay between submissions

        # Wait for completion
        wait_for_jobs(job_ids)

        # Aggregate results
        aggregate_results(task_config, model_map)

    else:
        # Single model test mode
        if args.model_path and args.model_name:
            model_out_dir = os.path.join(task_config['BASE_OUT'], args.model_name)
            submit_single_model_job(
                args.task,
                task_config,
                args.model_path,
                args.model_name,
                model_out_dir,
                args.reforce,
                yaml_config
            )
        else:
            print("Error: For single model testing, provide --model_path and --model_name")

if __name__ == "__main__":
    main()