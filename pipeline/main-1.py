import argparse
import os
import subprocess
import sys
import time
from typing import List, Dict, Any
import signal
import json
from pathlib import Path


# load model list from model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.model import Model_list

# supported benchmarks parameters from ../main.py
# can add/modify benchmarks here
SUPPORTED_BENCHMARKS = {
    # this is for pass@256
    # "aime24": {
    #     "n_fewshot": 0,
    #     "n_sampling": 256,
    #     "temperature": 0.6,
    #     "top_p": 0.95,
    #     "tokens": 2048
    # },
    # this is for pass@1
    "aime24": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 2048
    },
    "math": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 32768
    },
    "math500_pass1": {
        "n_fewshot": 5,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 8192
    },
    "math500_pass64": {
        "n_fewshot": 5,
        "n_sampling": 64,
        "temperature": 0.6,
        "top_p": 0.95,
        "tokens": 8192
    },
    "humaneval": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 1024
    },
    "gpqa_diamond": {
        "n_fewshot": 5,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "gpqa_diamond_pass32": {
        "n_fewshot": 5,
        "n_sampling": 32,
        "temperature": 0.6,
        "top_p": 0.9,
        "tokens": 4096
    },
    "ifeval": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "mmlu": {
        "n_fewshot": 4,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "mmlu_flan_cot_fewshot": {
        "n_fewshot": 4,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "mmlu_flan_cot_fewshot_pass16": {
        "n_fewshot": 4,
        "n_sampling": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "tokens": 4096
    },
    "mmlu_pro": {
        "n_fewshot": 5,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "mmlu_pro_pass16": {
        "n_fewshot": 5,
        "n_sampling": 16,
        "temperature": 0.7,
        "top_p": 0.9,
        "tokens": 4096
    },
    "mmlu_stem": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 32768
    },
    "bigbench_extrahard": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 32768
    },
    "bigbench_extrahard_verbal": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 32768
    },
    "bbh":{
        "n_fewshot": 3,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 2048
    },
    "bbh_cot_fewshot_pass16":{
        "n_fewshot": 3,
        "n_sampling": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "tokens": 4096,
    },
    "bbh_cot_fewshot":{
        "n_fewshot": 3,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "drop":{
        "n_fewshot": 1,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "arc_easy":{
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "arc_challenge":{
        "n_fewshot": 25,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "hellaswag":{
        "n_fewshot": 10,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "piqa":{
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "winogrande":{
        "n_fewshot": 5,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "triviaqa":{
        "n_fewshot": 5,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 8192
    },
    "nq_open":{
        "n_fewshot": 5,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 8192
    },
    "agieval":{
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "commonsense_qa":{
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "openbookqa":{
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "social_iqa":{
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "truthfulqa":{
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    # gsm8k, pass@1
    "gsm8k_pass1":{
        "n_fewshot": 8,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "gsm8k_pass16":{
        "n_fewshot": 8,
        "n_sampling": 16,
        "temperature": 0.6,
        "top_p": 0.95,
        "tokens": 8192
    }
}

OUTPUT_DIR = "/mnt/sharefs/users/haolong.jia/result"
GPUS_PER_NODE = 8  # 8 GPUs per node
SKIP_COMPLETED = False

class ModelEvaluator:
    def __init__(self, num_nodes: int, benchmark: str, prompt_type: str, model_size: float, debug: bool = False, gpus_per_task: int = 1):
        self.num_nodes = num_nodes
        self.total_gpus = num_nodes * GPUS_PER_NODE
        self.benchmark = benchmark
        self.prompt_type = prompt_type
        self.model_size = model_size
        self.debug = debug
        self.gpus_per_task = gpus_per_task
        
        # create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{self.benchmark}", exist_ok=True)
        
        # initialize task queue, running jobs and completed tasks
        self.model_queue = list(Model_list)
        self.running_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_models = []
        
        # state file path for restoring
        self.state_file = Path(f"{OUTPUT_DIR}/{self.benchmark}/evaluation_state_{self.benchmark}.json")
        
        # set signal handler to exit
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
        
        # log file
        self.log_file = f"{OUTPUT_DIR}/{benchmark}/evaluation_log_{benchmark}.txt"
        
    def log(self, message: str):
        """write log and print to console"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
    
    def is_completed(self, model_path: str) -> bool:
        """check if the model evaluation on the specified benchmark is completed"""
        model_name = os.path.basename(model_path)
        model_output_dir = os.path.join(OUTPUT_DIR, self.benchmark, model_name)
        return os.path.exists(model_output_dir) and os.listdir(model_output_dir) and not os.path.isfile(model_output_dir)
    
    def save_state(self):
        """do nothing (no state file)"""
        pass
            
    def load_state(self) -> bool:
        """do nothing (no state file)"""
        return False
    
    def handle_interrupt(self, sig, frame):
        """handle interrupt signal"""
        self.log("\nInterrupted. Exiting...")
        sys.exit(0)
        
    def submit_job(self, model_path: str) -> str:
        """submit a single model evaluation job"""
        model_name = os.path.basename(model_path)
        model_output_dir = os.path.join(OUTPUT_DIR, self.benchmark, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # create job script
        job_script = f"{model_output_dir}/{model_name}_{self.benchmark}.sh"
        
        # select different evaluation scripts for different types of benchmarks
        if self.benchmark in [
            "aime24", "aime25", "math", "gpqa_diamond", "gpqa_diamond_pass32",
            "math500_pass1", "math500_pass64", "mmlu_stem", "gsm8k_pass1", "gsm8k_pass16"
        ]:
            # For GSM8K benchmarks, use 'gsm8k' as the actual data name
            data_name = "gsm8k" if self.benchmark in ["gsm8k_pass1", "gsm8k_pass16"] else self.benchmark
            data_name = "math500" if self.benchmark in ["math500_pass1", "math500_pass64"] else self.benchmark
            data_name = "gpqa_diamond" if self.benchmark in ["gpqa_diamond_pass32", "gpqa_diamond"] else self.benchmark
            with open(job_script, 'w') as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name={self.benchmark}_{model_name}
#SBATCH --output={model_output_dir}/slurm.out
#SBATCH --error={model_output_dir}/slurm.err
#SBATCH --gres=gpu:{self.gpus_per_task}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={12 * self.gpus_per_task}
#SBATCH --mem={32 * self.gpus_per_task}G
#SBATCH --time={12 * self.gpus_per_task}:00:00
#SBATCH --exclude=fs-mbz-gpu-[156,407,034,015,039,031,160,497,161,114,638,765,794,664,140,864,027,117,007,088,099,361,358,413,425,420,472,436,496,487,958,951,612,655,591,670,706,693,811,971]
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

# Let Slurm manage GPU allocation - don't override CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
cd /mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval
which python

# Set environment variable
export TOKENIZERS_PARALLELISM=false

# Directly call python script without srun
python3 -u math_eval.py \
    --model_name_or_path {model_path} \
    --data_names {data_name} \
    --output_dir {model_output_dir} \
    --split test \
    --prompt_type {self.prompt_type} \
    --num_test_sample -1 \
    --seed 0 \
    --temperature {SUPPORTED_BENCHMARKS[self.benchmark]["temperature"]} \
    --n_sampling {SUPPORTED_BENCHMARKS[self.benchmark]["n_sampling"]} \
    --top_p {SUPPORTED_BENCHMARKS[self.benchmark]["top_p"]} \
    --max_tokens_per_call {SUPPORTED_BENCHMARKS[self.benchmark]["tokens"]} \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --num_shots {SUPPORTED_BENCHMARKS[self.benchmark]["n_fewshot"]}
""")
        else:
            # the single-GPU script for single model of harness framework
            if self.gpus_per_task == 1:
                with open(job_script, 'w') as f:
                    f.write(f"""#!/bin/bash
#SBATCH --job-name={self.benchmark}_{model_name}
#SBATCH --output={model_output_dir}/slurm.out
#SBATCH --error={model_output_dir}/slurm.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=fs-mbz-gpu-[156,407,034,015,039,031,160,497,161,114,638,765,794,664,140,864,027,117,007,088,099,361,358,413,425,420,472,436,496,487,958,951,612,655,591,670,706,693,811,971]
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

echo "Working directory: $(pwd)"
echo "PATH before: $PATH"
export PATH=/mnt/weka/home/haolong.jia/miniconda3/bin:$PATH
source /mnt/weka/home/haolong.jia/miniconda3/etc/profile.d/conda.sh

cd /mnt/weka/home/haolong.jia/eval/RL-eval || echo "Failed to change directory"
conda activate harness-eval

echo "PATH after: $PATH"
which lm_eval || echo "lm_eval not found"

# Let Slurm manage GPU allocation - don't override CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# Use TP size of 1 since we're using a single GPU
TP_SIZE=1
MAX_MODEL_LEN={SUPPORTED_BENCHMARKS[self.benchmark]['tokens']}
GEN_KWARGS="temperature={SUPPORTED_BENCHMARKS[self.benchmark]['temperature']},top_p={SUPPORTED_BENCHMARKS[self.benchmark]['top_p']}"
NUM_FEWSHOT={SUPPORTED_BENCHMARKS[self.benchmark]['n_fewshot']}

# Directly call lm_eval command without srun
if [[ "{self.benchmark}" = "ifeval" ]]; then
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks {self.benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path {model_output_dir}/{self.benchmark}
elif [[ "{self.benchmark}" = "social_iqa" ]]; then
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks {self.benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --trust_remote_code \
        --output_path {model_output_dir}/{self.benchmark}
else
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks {self.benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --output_path {model_output_dir}/{self.benchmark}
fi
""")
            else:
                # the multi-GPU script for single model of harness framework
                with open(job_script, 'w') as f:
                    f.write(f"""#!/bin/bash
#SBATCH --job-name={self.benchmark}_{model_name}
#SBATCH --output={model_output_dir}/slurm.out
#SBATCH --error={model_output_dir}/slurm.err
#SBATCH --gres=gpu:{self.gpus_per_task}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={12 * self.gpus_per_task}
#SBATCH --mem={32 * self.gpus_per_task}G
#SBATCH --time={12 * self.gpus_per_task}:00:00
#SBATCH --exclude=fs-mbz-gpu-[156,407,034,015,039,031,160,497,161,114,638,765,794,664,140,864,027,117,007,088,099,361,358,413,425,420,472,436,496,487,958,951,612,655,591,670,706,693,811,971]
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

echo "Working directory: $(pwd)"
echo "PATH before: $PATH"
export PATH=/mnt/weka/home/haolong.jia/miniconda3/bin:$PATH
source /mnt/weka/home/haolong.jia/miniconda3/etc/profile.d/conda.sh

cd /mnt/weka/home/haolong.jia/eval/RL-eval || echo "Failed to change directory"
conda activate harness-eval

echo "PATH after: $PATH"
which lm_eval || echo "lm_eval not found"

# Let Slurm manage GPU allocation - don't override CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# Use TP size of {self.gpus_per_task} for multi-GPU
TP_SIZE={self.gpus_per_task}
MAX_MODEL_LEN={SUPPORTED_BENCHMARKS[self.benchmark]['tokens']}
GEN_KWARGS="temperature={SUPPORTED_BENCHMARKS[self.benchmark]['temperature']},top_p={SUPPORTED_BENCHMARKS[self.benchmark]['top_p']}"
NUM_FEWSHOT={SUPPORTED_BENCHMARKS[self.benchmark]['n_fewshot']}

# Directly call lm_eval command without srun
if [[ "{self.benchmark}" = "ifeval" ]]; then
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks {self.benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path {model_output_dir}/{self.benchmark}
elif [[ "{self.benchmark}" = "social_iqa" ]]; then
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks {self.benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --trust_remote_code \
        --output_path {model_output_dir}/{self.benchmark}
else
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks {self.benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --output_path {model_output_dir}/{self.benchmark}
fi
""")
                
        # submit job - no need to pass specific GPU ID to script as Slurm will manage it
        process = subprocess.run(["sbatch", job_script], check=True, capture_output=True, text=True)
        job_id = process.stdout.strip().split()[-1]
        
        self.log(f"Submitted job {job_id} for model {model_name}")
        return job_id
    
    def check_job_status(self, job_id: str) -> str:
        """check job status"""
        try:
            process = subprocess.run(["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                               check=True, capture_output=True, text=True)
            state = process.stdout.strip().split('\n')[0]
            return state
        except Exception as e:
            self.log(f"Error checking job {job_id}: {e}")
            return "UNKNOWN"
    
    def run_evaluation(self):
        """run evaluation on all models"""
        # try to restore previous state
        if not self.load_state():
            # if no state file or load failed, check which models are completed
            if SKIP_COMPLETED:
                self.model_queue = [model for model in self.model_queue 
                                  if not self.is_completed(model)]
            
        self.log(f"Starting evaluation on {self.benchmark} benchmark with {self.num_nodes} nodes")
        self.log(f"Total models to evaluate: {len(self.model_queue)}")
        
        # Maximum concurrent jobs = total available GPUs divided by gpus_per_task
        max_concurrent_jobs = self.total_gpus // self.gpus_per_task
        
        # update running job status
        for job_id, job_info in list(self.running_jobs.items()):
            status = self.check_job_status(job_id)
            if status in ["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"]:
                model_path = job_info["model_path"]
                model_name = os.path.basename(model_path)
                
                self.log(f"Job {job_id} for model {model_name} {status}")
                if status == "COMPLETED":
                    self.completed_models.append(model_path)
                else:
                    # failed job will be added back to the queue
                    self.model_queue.append(model_path)
                    
                # remove from running jobs
                del self.running_jobs[job_id]
        
        # main loop
        try:
            while self.model_queue or self.running_jobs:
                # submit new jobs if slots are available
                while len(self.running_jobs) < max_concurrent_jobs and self.model_queue:
                    model_path = self.model_queue.pop(0)
                    model_name = os.path.basename(model_path)
                    
                    # if model is already evaluated, skip
                    if SKIP_COMPLETED and self.is_completed(model_path):
                        self.log(f"Model {model_name} already evaluated, skipping")
                        self.completed_models.append(model_path)
                        continue
                    
                    # submit job (no need to pass specific GPU ID)
                    try:
                        job_id = self.submit_job(model_path)
                        self.running_jobs[job_id] = {
                            "model_path": model_path,
                            "start_time": time.time()
                        }
                    except Exception as e:
                        self.log(f"Error submitting job for {model_name}: {e}")
                        # when error occurs, add model back to the queue
                        self.model_queue.append(model_path)
                
                # check running job status
                for job_id, job_info in list(self.running_jobs.items()):
                    status = self.check_job_status(job_id)
                    if status in ["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"]:
                        model_path = job_info["model_path"]
                        model_name = os.path.basename(model_path)
                        
                        self.log(f"Job {job_id} for model {model_name} {status}")
                        if status == "COMPLETED":
                            self.completed_models.append(model_path)
                        else:
                            # failed job will be added back to the queue
                            self.model_queue.append(model_path)
                            
                        # remove from running jobs
                        del self.running_jobs[job_id]
                
                # status report
                self.log(f"Status: {len(self.completed_models)} completed, {len(self.running_jobs)} running, {len(self.model_queue)} queued")
                
                # wait for a while before checking again
                time.sleep(60)
                
            self.log(f"All evaluations completed! Total models evaluated: {len(self.completed_models)}")
            
        except Exception as e:
            self.log(f"Error during evaluation: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(prog="RL-eval-pipeline")
    parser.add_argument("--num_node", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark to evaluate on")
    parser.add_argument("--prompt_type", type=str, required=True, help="Prompt type (e.g., deepseek-distill-qwen)")
    parser.add_argument("--model_size", type=float, default=3.0, help="Model size in billions of parameters")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    parser.add_argument("--gpus_per_task", type=int, default=1, help="Number of GPUs per evaluation job, now this is only used for harness framework")
    args = parser.parse_args()
    
    # check if benchmark is supported
    if args.benchmark not in SUPPORTED_BENCHMARKS:
        print(f"Error: Benchmark '{args.benchmark}' not supported.")
        print(f"Supported benchmarks: {', '.join(SUPPORTED_BENCHMARKS.keys())}")
        sys.exit(1)
    
    # create evaluator and run
    evaluator = ModelEvaluator(
        num_nodes=args.num_node,
        benchmark=args.benchmark,
        prompt_type=args.prompt_type,
        model_size=args.model_size,
        debug=args.debug,
        gpus_per_task=args.gpus_per_task
    )
    
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
