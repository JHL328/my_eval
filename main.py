import os
import subprocess


MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
BENCHMARKS=["aime24", "math", "gpqa", "ifeval", "mmlu", "mmlu_pro"]
BENCHMARKS=["aime24", "math"]
OUTPUT_DIR = "./results"
SKIP_COMPLETED = True

qwen25_math_bool = False
completed_benchmarks = os.listdir(OUTPUT_DIR)
print(f"Found completed benchmarks: {completed_benchmarks}")
for benchmark in BENCHMARKS:
    # skip completed eval when enabled
    if SKIP_COMPLETED and benchmark in completed_benchmarks:
        print(f"skipping {benchmark}...")
        continue
    # supported by qwen2.5-math
    if benchmark in ["aime24", "math"]:
        if not qwen25_math_bool:
            qwen25_math_bool = True
            subprocess.run([
                "sbatch",
                "./sbatch_scripts/qwen2.5-math.sh",
                # prompt type
                # ./qwen2.5-math/evaluation/utils.py
                # choose from PROMPT_TEMPLATES
                "qwen25-math-cot",
                MODEL_NAME_OR_PATH,
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
