import os
import subprocess


MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
BENCHMARKS=["aime24", "math", "gpqa", "ifeval", "mmlu", "mmlu_pro"]
BENCHMARKS=["aime24", "math"]
OUTPUT_DIR = "./results"
SKIP_COMPLETED = False


def is_completed(path):
    return os.path.exists(path) and os.listdir(path) and not os.path.isfile(path)


model_name_or_path_re = MODEL_NAME_OR_PATH.replace("/", "__")
for benchmark in BENCHMARKS:
    # skip completed eval when enabled
    if SKIP_COMPLETED and is_completed(os.path.join(OUTPUT_DIR, benchmark, model_name_or_path_re)):
        print(f"skipping {benchmark}...")
        continue
    # supported by qwen2.5-math
    if benchmark in ["aime24", "math"]:
        subprocess.run([
            "sbatch",
            "./sbatch_scripts/qwen2.5-math.sh",
            # prompt type
            # ./qwen2.5-math/evaluation/utils.py
            # choose from PROMPT_TEMPLATES
            "qwen25-math-cot",
            MODEL_NAME_OR_PATH,
            benchmark,
            # max tokens per call
            "32768",
            os.path.abspath(OUTPUT_DIR)
        ], check=True)
    # supported by lm-evalulation-harness
    else:
        subprocess.run([
            "sbatch",
            "./sbatch_scripts/harness.sh",
            MODEL_NAME_OR_PATH,
            benchmark,
            OUTPUT_DIR
        ], check=True)
