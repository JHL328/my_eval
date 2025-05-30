import argparse
import os
import subprocess


SUPPORTED_BENCHMARKS = {
    "aime24": {
        "n_fewshot": 0,
        "n_sampling": 16,
        "temperature": 0.6,
        "top_p": 0.95,
        "tokens": 32768
    },
    "aime25": {
        "n_fewshot": 0,
        "n_sampling": 16,
        "temperature": 0.6,
        "top_p": 0.95,
        "tokens": 32768
    },
    "math": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 32768
    },
    "math500": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 32768
    },
    "gpqa_diamond": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 32768
    },
    "ifeval": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "mmlu": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
        "tokens": 4096
    },
    "mmlu_pro": {
        "n_fewshot": 5,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1,
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
    }
}

OUTPUT_DIR = "./eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SKIP_COMPLETED = False
parser = argparse.ArgumentParser(prog="RL-eval")
parser.add_argument("-p", "--model_path", type=str, required=True)
parser.add_argument("--benchmarks", action="extend", nargs="+", type=str, required=True)
parser.add_argument("--prompt_type", type=str, required=True)
parser.add_argument("--model_size", type=float, required=True)
args = parser.parse_args()
print(args)
MODEL_NAME_OR_PATH = args.model_path
BENCHMARKS_TO_RUN = args.benchmarks
PROMPT_TYPE = args.prompt_type


def is_completed(path):
    return os.path.exists(path) and os.listdir(path) and not os.path.isfile(path)


model_name_or_path_re = MODEL_NAME_OR_PATH.replace("/", "__")
print(f"target model: {MODEL_NAME_OR_PATH}")
for benchmark in BENCHMARKS_TO_RUN:
    # skip completed eval when enabled
    if SKIP_COMPLETED and is_completed(os.path.join(OUTPUT_DIR, benchmark, model_name_or_path_re)):
        print(f"skipping {benchmark}...")
        print("=" * 50)
        continue
    # supported by qwen2.5-math
    if benchmark in ["aime24", "aime25", "math", "gpqa_diamond", "math500", "mmlu_stem"]:
        print(f"qwen2.5-math running target benchmark: {benchmark}...")
        print("=" * 50)
        completed_process = subprocess.run([
            "sbatch",
            "./sbatch_scripts/qwen2.5-math.sh",
            # ./qwen2.5-math/evaluation/utils.py
            # choose from PROMPT_TEMPLATES
            # deepseek-distill-qwen
            args.prompt_type,
            MODEL_NAME_OR_PATH,
            benchmark,
            str(SUPPORTED_BENCHMARKS[benchmark]["n_fewshot"]),
            str(SUPPORTED_BENCHMARKS[benchmark]["n_sampling"]),
            str(SUPPORTED_BENCHMARKS[benchmark]["tokens"]),
            str(SUPPORTED_BENCHMARKS[benchmark]["temperature"]),
            str(SUPPORTED_BENCHMARKS[benchmark]["top_p"]),
            os.path.abspath(OUTPUT_DIR),
            args.model_size,
        ], check=True)
    # supported by lm-evalulation-harness
    else:
        print(f"lm-evalulation-harness running target benchmark: {benchmark}...")
        print("=" * 50)
        completed_process = subprocess.run([
            "sbatch",
            "./sbatch_scripts/harness.sh",
            MODEL_NAME_OR_PATH,
            benchmark,
            str(SUPPORTED_BENCHMARKS[benchmark]["n_fewshot"]),
            str(SUPPORTED_BENCHMARKS[benchmark]["tokens"]),
            f"temperature={SUPPORTED_BENCHMARKS[benchmark]['temperature']},top_p={SUPPORTED_BENCHMARKS[benchmark]['top_p']}",
            OUTPUT_DIR
        ], check=True)
    print(completed_process.args)
