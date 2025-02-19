import subprocess


MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
BENCHMARKS=["aime24", "math", "gpqa", "ifeval", "mmlu", "mmlu_pro"]

qwen25_math_bool = False
for benchmark in BENCHMARKS:
    # supported by qwen2.5-math
    if benchmark in ["aime24", "math"]:
        if not qwen25_math_bool:
            qwen25_math_bool = True
            subprocess.run([
                "sbatch",
                "./sbatch_scripts/qwen2.5-math.sh",
                MODEL_NAME_OR_PATH,
                # prompt type
                # ./qwen2.5-math/evaluation/utils.py
                # choose from PROMPT_TEMPLATES
                "qwen25-math-cot",
            ], check=True)
    # supported by lm-evalulation-harness
    else:
        subprocess.run([
            "sbatch",
            "./sbatch_scripts/harness.sh",
            MODEL_NAME_OR_PATH,
            benchmark
        ], check=True)
