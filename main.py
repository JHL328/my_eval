import os
import subprocess

# LIMO
# MODEL_NAME_OR_PATH = "/mbz/users/yuqi.wang/LIMO/train/saves/qwen2.5-32b-instruct/full/limo_sft/checkpoint-390"
# MODEL_NAME_OR_PATH = "GAIR/LIMO"
# IQ
# MODEL_NAME_OR_PATH = "/mbz/users/yuqi.wang/LLaMA-Factory/saves/qwen2.5-32b-instruct/full/iq_sft/checkpoint-1000/"
# COLDSTART
# MODEL_NAME_OR_PATH = "/mbz/users/richard.fan/LLaMA-Factory/qwen32b_instruct_coldstart"
# MODEL_NAME_OR_PATH = "/mbz/users/richard.fan/LLaMA-Factory/saves/qwen2.5-32b-instruct-5epochs/full/sft/checkpoint-80"
# MODEL_NAME_OR_PATH = "/mbz/users/yuqi.wang/RL-eval/models/coldstart-2"
# MODEL_NAME_OR_PATH = "/mbz/users/yuqi.wang/RL-eval/models/coldstart-3"
# MODEL_NAME_OR_PATH = "/mbz/users/yuqi.wang/RL-eval/models/coldstart-4"
# MODEL_NAME_OR_PATH = "/mbz/users/richard.fan/LLaMA-Factory/saves/qwen2.5-32b-instruct-16-gpu-5epochs-20kcutoff-112cpu/full/sft/checkpoint-315"
# MODEL_NAME_OR_PATH = "/mbz/users/richard.fan/LLaMA-Factory/saves/qwen2.5-32b-instruct-16-gpu-5epochs/full/sft/checkpoint-315"
# QWEN2.5-32B
# MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-32B"
# QWEN2.5-32B-instruct
# MODEL_NAME_OR_PATH =  "Qwen/Qwen2.5-32B-instruct"
# DEEPSEEK-R1-DISTILL-QWEN-32B
MODEL_NAME_OR_PATH =  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# LLAMA-3.3-70B-instruct
# MODEL_NAME_OR_PATH = "meta-llama/Llama-3.3-70B-Instruct"
# MODEL_NAME_OR_PATH = "Qwen/QwQ-32B-Preview"
# INTERNAL_RL (n = 48, 96, 144, 192, 240, 288)
# MODEL_NAME_OR_PATH = "/mbz/users/shibo.hao/Reasoning360/checkpoints/Reasoning360/shibo-math-grpo-32nodes-setting2-Qwen2.5-32B/global_step_288/actor/huggingface"

# BENCHMARKS_TO_RUN = ["aime24", "aime25", "math", "math500", "gpqa_diamond", "mmlu_pro", "mmlu", "ifeval"]
BENCHMARKS_TO_RUN = ["aime24"]

SUPPORTED_BENCHMARKS = {
    "aime24": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    #     "n_sampling": 16,
    #     "temperature": 0.6,
    #     "top_p": 0.95
    },
    "aime25": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
        # "n_sampling": 16,
        # "temperature": 0.6,
        # "top_p": 0.95
    },
    "math": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "math500": {
        "n_fewshot": 0,
        "n_sampling": 4,
        "temperature": 0.6,
        "top_p": 0.95
    },
    "gpqa_diamond": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "gpqa_main": {
        "n_fewshot": 0,
        "n_sampling": 1,
        "temperature": 0,
        "top_p": 1
    },
    "gpqa": {
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
    }
}

OUTPUT_DIR = "./eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    if benchmark in ["aime24", "aime25", "math", "gpqa_diamond", "math500"]:
        # prompt_type = SUPPORTED_TEMPLATES.get(MODEL_NAME_OR_PATH, "internal-rl")
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
            f"temperature={SUPPORTED_BENCHMARKS[benchmark]['temperature']},top_p={SUPPORTED_BENCHMARKS[benchmark]['top_p']}",
            OUTPUT_DIR
        ], check=True)
    print(completed_process.args)
