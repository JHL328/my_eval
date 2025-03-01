import os
import subprocess


MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-32B-instruct"
# BENCHMARKS_TO_RUN = ["aime24", "math", "gpqa_main", "gpqa_diamond", "mmlu_pro", "mmlu_stem", "ifeval", "mmlu"]
BENCHMARKS_TO_RUN = ["mmlu"]
SUPPORTED_BENCHMARKS = {
    "aime24": {
        "n_fewshot": 0,
        "n_sampling": 16
    },
    "math": {
        "n_fewshot": 4,
        "n_sampling": 1
    },
    "gpqa_main": {
        "n_fewshot": 5,
        "n_sampling": 1
    },
    "gpqa_diamond": {
        "n_fewshot": 5,
        "n_sampling": 1
    },
    "ifeval": {
        "n_fewshot": 0,
        "n_sampling": 1
    },
    "mmlu": {
        "n_fewshot": 5,
        "n_sampling": 1
    },
    "mmlu_pro": {
        "n_fewshot": 5,
        "n_sampling": 1
    },
    "mmlu_stem": {
        "n_fewshot": 0,
        "n_sampling": 1
    }
}
SUPPORTED_TEMPLATES = {
    "Qwen/Qwen2.5-32B": "qwen25",
    "Qwen/Qwen2.5-32B-instruct": "qwen25-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-distill-qwen"
}
OUTPUT_DIR = "./results"
SKIP_COMPLETED = False


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
    if benchmark in ["aime24", "math", "gpqa_diamond", "gpqa_main", "mmlu_stem", "mmlu_pro", "mmlu"]:
        prompt_type = SUPPORTED_TEMPLATES.get(MODEL_NAME_OR_PATH, "internal-rl")
        print(f"target benchmark: {benchmark}...")
        subprocess.run([
            "sbatch",
            "./sbatch_scripts/qwen2.5-math.sh",
            # ./qwen2.5-math/evaluation/utils.py
            # choose from PROMPT_TEMPLATES
            prompt_type,
            MODEL_NAME_OR_PATH,
            benchmark,
            str(SUPPORTED_BENCHMARKS[benchmark]["n_fewshot"]),
            str(SUPPORTED_BENCHMARKS[benchmark]["n_sampling"]),
            # max tokens per call
            "131072",
            os.path.abspath(OUTPUT_DIR)
        ], check=True)
    # supported by lm-evalulation-harness
    else:
        print(f"target benchmark: {benchmark}...")
        subprocess.run([
            "sbatch",
            "./sbatch_scripts/harness.sh",
            MODEL_NAME_OR_PATH,
            benchmark,
            str(SUPPORTED_BENCHMARKS[benchmark]["n_fewshot"]),
            OUTPUT_DIR
        ], check=True)
