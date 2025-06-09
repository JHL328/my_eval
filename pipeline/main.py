import argparse
import os
import subprocess
import sys
import time
import signal
from typing import Dict, Any
from pathlib import Path

# load model and benchmark configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.model import Model_map
from pipeline.task import SUPPORTED_BENCHMARKS

OUTPUT_DIR = "/mnt/sharefs/users/haolong.jia/result"
GPUS_PER_NODE = 8  # number of GPUs per node
SKIP_COMPLETED = False

SPECIAL_BENCHMARKS = [
    "aime24", "aime25", "math", "gpqa_diamond", "gpqa_diamond_pass32",
    "math500_pass1", "math500_pass64", "mmlu_stem", "gsm8k_pass1", "gsm8k_pass16"
] # this is the benchmark that use qwen2.5-math to evaluate

class ModelEvaluator:
    def __init__(self, num_nodes: int, gpus_per_task: int = 8):
        self.num_nodes = num_nodes
        self.gpus_per_task = gpus_per_task
        self.total_gpus = num_nodes * GPUS_PER_NODE
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.model_queue = self.generate_task_queue()
        self.running_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_models = []
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def generate_task_queue(self):
        queue = []
        for model_path in Model_map:
            for benchmark, eval_args in SUPPORTED_BENCHMARKS.items():
                queue.append({
                    "model_path": model_path,
                    "benchmark": benchmark,
                    "eval_args": eval_args,
                })
        return queue

    def log(self, message: str):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)

    def is_completed(self, model_path: str, benchmark: str) -> bool:
        model_name = os.path.basename(model_path)
        model_output_dir = os.path.join(OUTPUT_DIR, benchmark, model_name)
        return os.path.exists(model_output_dir) and os.listdir(model_output_dir) and not os.path.isfile(model_output_dir)

    def handle_interrupt(self, sig, frame):
        self.log("\nInterrupted. Exiting...")
        sys.exit(0)

    def submit_job(self, model_path: str, benchmark: str, eval_args: dict):
        model_name = Model_map.get(model_path, os.path.basename(model_path))
        model_output_dir = os.path.join(OUTPUT_DIR, benchmark, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        job_script = f"{model_output_dir}/{model_name}_{benchmark}.sh"
        if benchmark in SPECIAL_BENCHMARKS:
            # this is the benchmark that use qwen2.5-math to evaluate
            if benchmark in ["gsm8k_pass1", "gsm8k_pass16"]:
                data_name = "gsm8k"
            elif benchmark in ["math500_pass1", "math500_pass64"]:
                data_name = "math500"
            elif benchmark in ["gpqa_diamond", "gpqa_diamond_pass32"]:
                data_name = "gpqa_diamond"
            else:
                data_name = benchmark
            with open(job_script, 'w') as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name={benchmark}_{model_name}
#SBATCH --output={model_output_dir}/slurm.out
#SBATCH --error={model_output_dir}/slurm.err
#SBATCH --gres=gpu:{self.gpus_per_task}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={16 * self.gpus_per_task}
#SBATCH --time=12:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

cd /mnt/weka/home/haolong.jia/eval/RL-eval/qwen2.5-math/evaluation
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate qwen-eval
which python
export TOKENIZERS_PARALLELISM=false
python3 -u math_eval.py \
    --model_name_or_path {model_path} \
    --data_names {data_name} \
    --output_dir {model_output_dir} \
    --split test \
    --prompt_type cot \
    --num_test_sample -1 \
    --seed 0 \
    --temperature {eval_args['temperature']} \
    --n_sampling {eval_args['n_sampling']} \
    --top_p {eval_args['top_p']} \
    --max_tokens_per_call {eval_args['tokens']} \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --num_shots {eval_args.get('n_fewshot', 0)}
""")
        else:
            # harness/lm_eval framework, data parallel
            with open(job_script, 'w') as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name={benchmark}_{model_name}
#SBATCH --output={model_output_dir}/slurm.out
#SBATCH --error={model_output_dir}/slurm.err
#SBATCH --gres=gpu:{self.gpus_per_task}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={16 * self.gpus_per_task}
#SBATCH --time=12:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

cd /mnt/weka/home/haolong.jia/eval/RL-eval
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval
TP_SIZE=1
DP_SIZE={self.gpus_per_task}
MAX_MODEL_LEN={eval_args['tokens']}
GEN_KWARGS=\"temperature={eval_args['temperature']},top_p={eval_args['top_p']}\"
NUM_FEWSHOT={eval_args.get('n_fewshot', 0)}

if [[ \"{benchmark}\" = \"ifeval\" ]]; then
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,data_parallel_size=$DP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7 \
        --tasks {benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path {model_output_dir}/{benchmark}
elif [[ \"{benchmark}\" = \"social_iqa\" ]]; then
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,data_parallel_size=$DP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7 \
        --tasks {benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --trust_remote_code \
        --output_path {model_output_dir}/{benchmark}
else
    lm_eval --model vllm \
        --model_args pretrained={model_path},tensor_parallel_size=$TP_SIZE,data_parallel_size=$DP_SIZE,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.7 \
        --tasks {benchmark} \
        --batch_size auto \
        --log_samples \
        --gen_kwargs $GEN_KWARGS \
        --num_fewshot $NUM_FEWSHOT \
        --output_path {model_output_dir}/{benchmark}
fi
""")
        process = subprocess.run(["sbatch", job_script], check=True, capture_output=True, text=True)
        job_id = process.stdout.strip().split()[-1]
        self.log(f"Submitted job {job_id} for model {model_name} on {benchmark}")
        return job_id

    def check_job_status(self, job_id: str) -> str:
        try:
            process = subprocess.run(["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                               check=True, capture_output=True, text=True)
            state = process.stdout.strip().split('\n')[0]
            return state
        except Exception as e:
            self.log(f"Error checking job {job_id}: {e}")
            return "UNKNOWN"

    def run_evaluation(self):
        max_concurrent_jobs = self.total_gpus // self.gpus_per_task
        self.log(f"Total models to evaluate: {len(self.model_queue)}")
        try:
            while self.model_queue or self.running_jobs:
                # submit new jobs if slots are available
                while len(self.running_jobs) < max_concurrent_jobs and self.model_queue:
                    model_info = self.model_queue.pop(0)
                    model_path = model_info["model_path"]
                    benchmark = model_info["benchmark"]
                    eval_args = model_info["eval_args"]
                    model_name = os.path.basename(model_path)
                    if SKIP_COMPLETED and self.is_completed(model_path, benchmark):
                        self.log(f"Model {model_name} on {benchmark} already evaluated, skipping")
                        self.completed_models.append((model_path, benchmark))
                        continue
                    try:
                        job_id = self.submit_job(model_path, benchmark, eval_args)
                        self.running_jobs[job_id] = {
                            "model_path": model_path,
                            "benchmark": benchmark,
                            "start_time": time.time()
                        }
                    except Exception as e:
                        self.log(f"Error submitting job for {model_name} on {benchmark}: {e}")
                        self.model_queue.append(model_info)
                # check running job status
                for job_id, job_info in list(self.running_jobs.items()):
                    status = self.check_job_status(job_id)
                    if status in ["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"]:
                        model_path = job_info["model_path"]
                        benchmark = job_info["benchmark"]
                        model_name = os.path.basename(model_path)
                        self.log(f"Job {job_id} for model {model_name} on {benchmark} {status}")
                        if status == "COMPLETED":
                            self.completed_models.append((model_path, benchmark))
                        else:
                            self.model_queue.append({
                                "model_path": model_path,
                                "benchmark": benchmark,
                                "eval_args": SUPPORTED_BENCHMARKS[benchmark]
                            })
                        del self.running_jobs[job_id]
                self.log(f"Status: {len(self.completed_models)} completed, {len(self.running_jobs)} running, {len(self.model_queue)} queued")
                time.sleep(60)
            self.log(f"All evaluations completed! Total: {len(self.completed_models)}")
        except Exception as e:
            self.log(f"Error during evaluation: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(prog="RL-eval-pipeline")
    parser.add_argument("--num_node", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--gpus_per_task", type=int, default=8, help="Number of GPUs per evaluation job (data parallel size)")
    args = parser.parse_args()
    evaluator = ModelEvaluator(
        num_nodes=args.num_node,
        gpus_per_task=args.gpus_per_task
    )
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
