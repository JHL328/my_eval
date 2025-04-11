import argparse
import os
import subprocess


SUPPORTED_BENCHMARKS = {
    "aime24": {
        "n_fewshot": 0,
        "n_sampling": 16,
        "temperature": 0.6,
        "top_p": 0.95
    },
    "aime25": {
        "n_fewshot": 0,
        "n_sampling": 16,
        "temperature": 0.6,
        "top_p": 0.95
    },
    "math": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "math500": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "gpqa_diamond": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "ifeval": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "mmlu": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "mmlu_pro": {
        "n_fewshot": 5,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "mmlu_stem": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "bigbench_extrahard": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "bigbench_extrahard_verbal": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    }
}

OUTPUT_DIR = "./eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SKIP_COMPLETED = False
parser = argparse.ArgumentParser(prog="RL-eval")
parser.add_argument("-p", "--model_path", type=str, required=True)
parser.add_argument("--benchmarks", action="extend", nargs="+", type=str, required=True)
args = parser.parse_args()
print(args)
MODEL_NAME_OR_PATH = args.model_path
BENCHMARKS_TO_RUN = args.benchmarks

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
            "qwen25-math-cot",
            MODEL_NAME_OR_PATH,
            benchmark,
            str(SUPPORTED_BENCHMARKS[benchmark]["n_fewshot"]),
            str(SUPPORTED_BENCHMARKS[benchmark]["n_sampling"]),
            # max tokens per call
            "32768",
            str(SUPPORTED_BENCHMARKS[benchmark]["temperature"]),
            str(SUPPORTED_BENCHMARKS[benchmark]["top_p"]),
            os.path.abspath(OUTPUT_DIR)
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
            "4096",
            f"temperature={SUPPORTED_BENCHMARKS[benchmark]['temperature']},top_p={SUPPORTED_BENCHMARKS[benchmark]['top_p']}",
            OUTPUT_DIR
        ], check=True)
    print(completed_process.args)
