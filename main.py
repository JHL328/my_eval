import subprocess


MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
BENCHMARKS=["AIME", "gpqa", "ifeval", "mmlu", "mmlu_pro"]


for benchmark in BENCHMARKS:
    if benchmark == "AIME":
        subprocess.run([
            "sbatch",
            "./sbatch_scripts/aime.sh",
            MODEL_NAME_OR_PATH,
            # prompt type
            # ./qwen2.5-math/evaluation/utils.py
            # choose from PROMPT_TEMPLATES
            "qwen25-math-cot",
        ], check=True)
    else:
        subprocess.run([
            "sbatch",
            "./sbatch_scripts/harness.sh",
            MODEL_NAME_OR_PATH,
            benchmark
        ], check=True)
