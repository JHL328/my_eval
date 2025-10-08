import os
import json
import re
import sys
import time
import argparse
import subprocess
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Model_map, get_model_map_by_type

# =====================
# Task Configurations
# =====================
TASK_CONFIGS = {
    "mmlu_redux": {
        "BASE_OUT": "/mnt/sharefs/users/haolong.jia/result/mmlu_redux",
        "BASE_OUT_SFT": "/mnt/sharefs/users/haolong.jia/result/mmlu_redux_sft",
        "DATASET": "edinburgh-dawg/mmlu-redux-2.0",
        "N_SAMPLING": 1,
        "TEMPERATURE": 0.7,
        "TOP_P": 0.95,
        "MAX_TOKENS": 1024,
        "NUM_SHOTS": 4,
        "GPUS_PER_TASK": 1,
        "TIME_LIMIT": "8:00:00",
        "PARTITION": "lowprio",
        "QOS": "lowprio",
        "MEM": "150G",
        "CONDA_ACTIVATE_PATH": "source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval",
        "CD_PATH_IN_JOB_SCRIPT": "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline",
    }
}

# Few-shot examples for CoT (can be customized per subject)
DEFAULT_FEWSHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["London", "Berlin", "Paris", "Madrid"],
        "answer": 2,
        "reasoning": "Let's think step by step. France is a country in Western Europe. The capital city of France is Paris, which is known for landmarks like the Eiffel Tower and the Louvre Museum. Therefore, the answer is (C)."
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
        "answer": 1,
        "reasoning": "Let's think step by step. The Red Planet refers to a planet that appears reddish in color. Mars appears red due to iron oxide (rust) on its surface. Venus is often called Earth's twin, Jupiter is the largest planet, and Saturn is known for its rings. Therefore, the answer is (B)."
    },
    {
        "question": "What is the largest ocean on Earth?",
        "choices": ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"],
        "answer": 3,
        "reasoning": "Let's think step by step. Earth has five major oceans. The Pacific Ocean covers about 165 million square kilometers, making it larger than all of Earth's land area combined. The Atlantic is the second-largest, followed by the Indian Ocean, with the Arctic being the smallest. Therefore, the answer is (D)."
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        "answer": 1,
        "reasoning": "Let's think step by step. 'Romeo and Juliet' is a famous tragedy about two young star-crossed lovers. This play was written by William Shakespeare in the early part of his career, around 1594-1596. Shakespeare is known for many famous plays including Hamlet, Macbeth, and A Midsummer Night's Dream. Therefore, the answer is (B)."
    }
]

# =====================
# Utility Functions
# =====================
def build_fewshot_prompt(examples, subject):
    """Build few-shot prompt with examples"""
    prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    
    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        for i, choice in enumerate(ex['choices']):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += f"A: {ex['reasoning']}\n\n"
    
    return prompt

def build_test_prompt(question, choices, subject, fewshot_prompt=""):
    """Build prompt for a test question"""
    prompt = fewshot_prompt if fewshot_prompt else f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    
    prompt += f"Q: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "A: Let's think step by step."
    
    return prompt

def extract_answer(text):
    """Extract answer from model output with CoT reasoning"""
    # Convert text to uppercase for matching
    text_upper = text.upper()
    
    # Pattern 1: "The answer is (X)" or "the answer is X"
    patterns = [
        r"THE ANSWER IS \(?([A-D])\)?",
        r"ANSWER IS \(?([A-D])\)?",
        r"THEREFORE,? THE ANSWER IS \(?([A-D])\)?",
        r"SO THE ANSWER IS \(?([A-D])\)?",
        r"THUS,? THE ANSWER IS \(?([A-D])\)?",
        r"HENCE,? THE ANSWER IS \(?([A-D])\)?",
        r"CORRECT ANSWER IS \(?([A-D])\)?",
        r"CORRECT OPTION IS \(?([A-D])\)?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)
    
    # Pattern 2: Look for standalone (X) at the end
    match = re.search(r"\(([A-D])\)[^A-D]*$", text_upper)
    if match:
        return match.group(1)
    
    # Pattern 3: Look for "X." or "X)" patterns
    match = re.search(r"^([A-D])[\.\)]", text_upper[-20:])
    if match:
        return match.group(1)
    
    # Pattern 4: Just look for the last occurrence of A, B, C, or D
    matches = re.findall(r"\b([A-D])\b", text_upper)
    if matches:
        return matches[-1]
    
    return ""

def evaluate_single_subject(subject, model_path, model_name, output_dir, task_config, model_type="base"):
    """Evaluate a single MMLU-Redux subject"""
    print(f"\n=== Evaluating {subject} for {model_name} ===")
    
    # Load dataset
    try:
        dataset = load_dataset(task_config["DATASET"], subject, split="test")
    except Exception as e:
        print(f"Error loading dataset for {subject}: {e}")
        return None
    
    # Prepare few-shot prompt
    fewshot_prompt = build_fewshot_prompt(DEFAULT_FEWSHOT_EXAMPLES, subject)
    
    # Prepare all prompts
    prompts = []
    targets = []
    for item in dataset:
        prompt = build_test_prompt(
            item["question"], 
            item["choices"], 
            subject,
            fewshot_prompt
        )
        prompts.append(prompt)
        targets.append(item["answer"])
    
    # Apply chat template for SFT models
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
    
    # Load model and run inference
    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(
        temperature=task_config["TEMPERATURE"],
        top_p=task_config["TOP_P"],
        max_tokens=task_config["MAX_TOKENS"],
        n=task_config["N_SAMPLING"]
    )
    
    print(f"Running inference on {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Parse results and calculate accuracy
    results = []
    correct = 0
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        pred_answer = extract_answer(response)
        gold_answer = chr(65 + targets[i])  # Convert index to letter
        is_correct = (pred_answer == gold_answer)
        correct += is_correct
        
        results.append({
            "question": dataset[i]["question"],
            "choices": dataset[i]["choices"],
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "response": response,
            "correct": int(is_correct),
            "error_type": dataset[i].get("error_type", ""),
        })
        
        # Print first example for debugging
        if i == 0:
            print(f"\n=== First Example ===")
            print(f"Question: {dataset[i]['question']}")
            print(f"Response: {response}")
            print(f"Extracted: {pred_answer}")
            print(f"Gold: {gold_answer}")
            print(f"Correct: {is_correct}")
            print("=" * 50)
    
    accuracy = correct / len(results) if results else 0
    print(f"Accuracy for {subject}: {accuracy:.4f} ({correct}/{len(results)})")
    
    # Save results
    subject_dir = os.path.join(output_dir, model_name)
    os.makedirs(subject_dir, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(subject_dir, f"{subject}.csv"), index=False)
    
    # Save metrics
    metrics = {
        "subject": subject,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results)
    }
    
    with open(os.path.join(subject_dir, f"{subject}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Clean up
    del llm
    import gc
    gc.collect()
    
    return metrics

def get_all_subjects(dataset_name):
    """Get all available subjects from the dataset"""
    # For MMLU-Redux 2.0, we need to manually list or discover subjects
    # This is a placeholder - you may need to update based on actual dataset structure
    subjects = [
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
    return subjects

def submit_jobs_for_all_models(args, task_config):
    """Submit SLURM jobs for all models"""
    base_out = task_config["BASE_OUT_SFT"] if args.type == "sft" else task_config["BASE_OUT"]
    os.makedirs(base_out, exist_ok=True)
    
    # Get model map
    model_map = get_model_map_by_type(args.type)
    
    # Get all subjects
    subjects = get_all_subjects(task_config["DATASET"])
    print(f"Found {len(subjects)} subjects to evaluate")
    
    submitted_job_ids = []
    
    for model_path, model_name in model_map.items():
        model_dir = os.path.join(base_out, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        for subject in subjects:
            # Check if already done
            metrics_file = os.path.join(model_dir, f"{subject}_metrics.json")
            if os.path.exists(metrics_file) and not args.reforce:
                print(f"Skipping {model_name}/{subject} - already done")
                continue
            
            # Create job script
            job_name = f"mmlu_redux_{model_name}_{subject}"
            job_script = os.path.join(model_dir, f"{job_name}.sh")
            
            with open(job_script, 'w') as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={model_dir}/{subject}.out
#SBATCH --error={model_dir}/{subject}.err
#SBATCH --gres=gpu:{task_config['GPUS_PER_TASK']}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time={task_config['TIME_LIMIT']}
#SBATCH --partition={task_config['PARTITION']}
#SBATCH --qos={task_config['QOS']}
#SBATCH --mem={task_config['MEM']}

cd {task_config['CD_PATH_IN_JOB_SCRIPT']}
{task_config['CONDA_ACTIVATE_PATH']}
which python
export TOKENIZERS_PARALLELISM=false

python3 -u {os.path.abspath(__file__)} \\
    --subject {subject} \\
    --model_path {model_path} \\
    --model_name {model_name} \\
    --output_dir {base_out} \\
    --type {args.type}
""")
            
            # Submit job
            try:
                result = subprocess.check_output(f"sbatch {job_script}", shell=True, text=True)
                match = re.search(r'Submitted batch job (\d+)', result)
                if match:
                    job_id = match.group(1)
                    submitted_job_ids.append(job_id)
                    print(f"Submitted job for {model_name}/{subject}: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit job for {model_name}/{subject}: {e}")
            
            time.sleep(0.1)  # Small delay between submissions
    
    return submitted_job_ids

def wait_for_jobs(job_ids):
    """Wait for all SLURM jobs to complete"""
    if not job_ids:
        return
    
    print(f"\nWaiting for {len(job_ids)} jobs to complete...")
    while job_ids:
        job_ids_str = ",".join(job_ids)
        try:
            output = subprocess.check_output(
                f"squeue -h -j {job_ids_str} -o '%i'",
                shell=True,
                text=True
            )
            running_jobs = set(output.strip().split('\n')) if output.strip() else set()
            job_ids = [jid for jid in job_ids if jid in running_jobs]
            
            if job_ids:
                print(f"{len(job_ids)} jobs still running. Checking again in 30 seconds...")
                time.sleep(30)
        except subprocess.CalledProcessError:
            print("All jobs completed!")
            break

def summarize_results(task_config, model_type="base"):
    """Summarize results across all models and subjects"""
    base_out = task_config["BASE_OUT_SFT"] if model_type == "sft" else task_config["BASE_OUT"]
    model_map = get_model_map_by_type(model_type)
    subjects = get_all_subjects(task_config["DATASET"])
    
    all_results = {}
    
    for model_path, model_name in model_map.items():
        model_dir = os.path.join(base_out, model_name)
        if not os.path.exists(model_dir):
            continue
        
        model_results = {}
        total_correct = 0
        total_questions = 0
        
        for subject in subjects:
            metrics_file = os.path.join(model_dir, f"{subject}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    model_results[subject] = metrics["accuracy"]
                    total_correct += metrics["correct"]
                    total_questions += metrics["total"]
        
        if model_results:
            avg_accuracy = total_correct / total_questions if total_questions > 0 else 0
            all_results[model_name] = {
                "average_accuracy": avg_accuracy,
                "subject_scores": model_results,
                "total_correct": total_correct,
                "total_questions": total_questions
            }
    
    # Save summary
    summary_file = os.path.join(base_out, "accuracy_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults summary saved to {summary_file}")
    
    # Print summary
    print("\n=== Accuracy Summary ===")
    for model_name, results in all_results.items():
        print(f"{model_name}: {results['average_accuracy']:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on MMLU-Redux dataset")
    parser.add_argument("--submit_jobs", action="store_true", help="Submit SLURM jobs for all models")
    parser.add_argument("--subject", type=str, help="Subject to evaluate")
    parser.add_argument("--model_path", type=str, help="Path to model")
    parser.add_argument("--model_name", type=str, help="Model name for output")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--type", type=str, default="base", choices=["base", "sft"], help="Model type")
    parser.add_argument("--reforce", action="store_true", help="Rerun even if results exist")
    parser.add_argument("--summarize", action="store_true", help="Summarize existing results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    task_config = TASK_CONFIGS["mmlu_redux"]
    
    if args.submit_jobs:
        # Submit batch jobs
        job_ids = submit_jobs_for_all_models(args, task_config)
        wait_for_jobs(job_ids)
        summarize_results(task_config, args.type)
    elif args.summarize:
        # Just summarize existing results
        summarize_results(task_config, args.type)
    else:
        # Run single evaluation
        if not all([args.subject, args.model_path, args.model_name, args.output_dir]):
            print("Error: For single evaluation, must provide --subject, --model_path, --model_name, --output_dir")
            sys.exit(1)
        
        metrics = evaluate_single_subject(
            args.subject,
            args.model_path,
            args.model_name,
            args.output_dir,
            task_config,
            args.type
        )
        
        if metrics:
            print(f"\nFinal accuracy for {args.subject}: {metrics['accuracy']:.4f}")