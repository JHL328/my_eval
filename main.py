import os
import subprocess

# LIMO
# MODEL_NAME_OR_PATH = "/mbz/users/yuqi.wang/LIMO/train/saves/qwen2.5-32b-instruct/full/limo_sft/checkpoint-390"
# IQ
# MODEL_NAME_OR_PATH = "/mbz/users/yuqi.wang/LLaMA-Factory/saves/qwen2.5-32b-instruct/full/iq_sft/checkpoint-1000/"
# COLDSTART
# MODEL_NAME_OR_PATH = "/mbz/users/richard.fan/LLaMA-Factory/qwen32b_instruct_coldstart"
# MODEL_NAME_OR_PATH = "/mbz/users/richard.fan/LLaMA-Factory/saves/qwen2.5-32b-instruct-5epochs/full/sft/checkpoint-80"
# MODEL_NAME_OR_PATH = "/mbz/users/richard.fan/LLaMA-Factory/saves/qwen2.5-32b-instruct-16-gpu-5epochs/full/sft/checkpoint-315"
# QWEN2.5-32B
# MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-32B"
# QWEN2.5-32B-instruct
# MODEL_NAME_OR_PATH =  "Qwen/Qwen2.5-32B-instruct"
# DEEPSEEK-R1-DISTILL-QWEN-32B
MODEL_NAME_OR_PATH =  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# INTERNAL_RL
# MODEL_NAME_OR_PATH = "/mbz/users/shibo.hao/Reasoning360/checkpoints/Reasoning360/shibo-math-grpo-32nodes-setting2-Qwen2.5-32B/global_step_480/actor/huggingface"

# BENCHMARKS_TO_RUN = ["aime24", "math", "gpqa", "mmlu_pro", "mmlu", "ifeval"]
BENCHMARKS_TO_RUN = ["aime24", "math"]

SUPPORTED_BENCHMARKS = {
    "aime24": {
        "n_fewshot": 0,
        "n_sampling": 1
    },
    "math": {
        "n_fewshot": 0,
        "n_sampling": 1
    },
    "gpqa": {
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
# SUPPORTED_TEMPLATES = {
#     "Qwen/Qwen2.5-32B": "qwen25",
#     "Qwen/Qwen2.5-32B-instruct": "qwen25-instruct",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-distill-qwen"
# }
OUTPUT_DIR = "./eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SKIP_COMPLETED = False
TEMPERATURE=0

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
    if benchmark in ["aime24", "math", "gpqa"]:
        # prompt_type = SUPPORTED_TEMPLATES.get(MODEL_NAME_OR_PATH, "internal-rl")
        print(f"qwen2.5-math running target benchmark: {benchmark}...")
        print("=" * 50)
        subprocess.run([
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
            str(TEMPERATURE),
            os.path.abspath(OUTPUT_DIR)
        ], check=True)
    # supported by lm-evalulation-harness
    else:
        print(f"lm-evalulation-harness running target benchmark: {benchmark}...")
        print("=" * 50)
        subprocess.run([
            "sbatch",
            "./sbatch_scripts/harness.sh",
            MODEL_NAME_OR_PATH,
            benchmark,
            str(SUPPORTED_BENCHMARKS[benchmark]["n_fewshot"]),
            f"temperature={TEMPERATURE}",
            OUTPUT_DIR
        ], check=True)
