import os
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import sys
import csv
import fcntl
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Model_map

# 8-shot CoT fewshot examples
FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    },
]

# Prompt construction
FEWSHOT_PROMPT = ""
for ex in FEWSHOT_EXAMPLES:
    FEWSHOT_PROMPT += f"Q: {ex['question']}\nA: {ex['target']}\n\n"

# Answer parsing: more robust than harness
ANSWER_PATTERNS = [
    r"The answer is ([\-0-9\.,]+)",
    r"#### ([\-0-9\.,]+)",
    r"([\-0-9\.,]+)$"
]

# 全局采样参数（模型和采样参数在slurm脚本中初始化）
SAMPLING_PARAMS = dict(
    temperature=0.6,
    top_p=0.95,
    n=16,
    max_tokens=2048,
    stop=["Q:", "</s>", "<|im_end|>", "\n\nQ:", "\n\nHuman:", "\n\nAssistant:", "Human:", "Assistant:"],
    seed=42,
)

def parse_answer(text):
    # 更全面的答案提取模式
    answer_patterns = [
        # 标准格式
        r"The answer is:?\s*\$?([\-0-9\.,]+)",
        r"#### ?\$?([\-0-9\.,]+)",
        # 变体格式
        r"Therefore,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"So,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Thus,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Hence,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Final answer:?\s*\$?([\-0-9\.,]+)",
        r"The final answer is:?\s*\$?([\-0-9\.,]+)",
        # 带单位的答案（如 miles, minutes, dollars等）
        r"The answer is:?\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?",
        r"=\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?\.?\s*(?:The answer|$)",
    ]
    
    # 尝试所有模式
    for pat in answer_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            # 取最后一个匹配（通常是最终答案）
            ans = matches[-1].replace(",", "").strip().rstrip(".")
            if ans:
                return ans
    
    # 改进的fallback：寻找句末的数字
    # 优先匹配句子结尾的数字
    sentence_end_pattern = r"(?:is|are|equals?|makes?|has|have|gets?|arrives?|covers?|travels?)\s+\$?([\-0-9\.,]+)(?:\s*(?:miles?|minutes?|hours?|dollars?|GB))?\.?\s*$"
    m = re.search(sentence_end_pattern, text, re.MULTILINE | re.IGNORECASE)
    if m:
        ans = m.group(1).replace(",", "").strip().rstrip(".")
        if ans:
            return ans
    
    # 最后的fallback：找最后一个完整句子中的数字
    sentences = text.split('.')
    for sent in reversed(sentences):
        # 跳过包含Human/Assistant的句子（可能是无关内容）
        if 'Human:' in sent or 'Assistant:' in sent:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sent)
        if numbers:
            return numbers[-1].lstrip('0') or '0'
    
    return ""

def call_model(prompt, n=16, llm=None):
    # 使用传入的llm对象，不再重复初始化
    params = SamplingParams(**{**SAMPLING_PARAMS, "n": n})
    outputs = llm.generate([prompt], params)
    generations = [o.outputs[i].text for i in range(n) for o in outputs]
    return generations

def call_model_batch(prompts, n=16, llm=None):
    # 批量推理版本
    params = SamplingParams(**{**SAMPLING_PARAMS, "n": n})
    outputs = llm.generate(prompts, params)
    # 返回每个prompt的n个生成结果
    batch_generations = []
    for output in outputs:
        generations = [output.outputs[i].text for i in range(n)]
        batch_generations.append(generations)
    return batch_generations

def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def update_passk_json(passk_path, model_name, passk_result, overwrite=False):
    # add lock to update passk.json
    with open(passk_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        if overwrite:
            all_results = {}
        else:
            try:
                all_results = json.load(f)
            except Exception:
                all_results = {}
        all_results[model_name] = passk_result
        f.seek(0)
        f.truncate()
        json.dump(all_results, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)

def evaluate_gsm8k(gsm8k_path, output_dir, n_sampling=16, model_path=None, model_name=None, base_out=None, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    # 初始化模型，只加载一次
    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, dtype="auto", tensor_parallel_size=1)
    print("Model loaded successfully!")
    
    results = []
    csv_rows = []
    with open(gsm8k_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # 准备所有prompts
    print("Preparing prompts...")
    prompts = []
    golds = []
    questions = []
    for item in tqdm(data, desc="Preparing data"):
        q = item["question"]
        gold = parse_answer(item["answer"])
        prompt = FEWSHOT_PROMPT + f"Q: {q}\nA: Let's think step by step."
        prompts.append(prompt)
        golds.append(gold)
        questions.append(q)
    
    # 一次性批量推理，让vLLM自动处理批处理
    print(f"Running inference on {len(prompts)} prompts...")
    params = SamplingParams(**{**SAMPLING_PARAMS, "n": n_sampling})
    outputs = llm.generate(prompts, params)
    
    # 处理结果
    print("Processing results...")
    for idx, (q, gold, output) in enumerate(tqdm(zip(questions, golds, outputs), total=len(questions), desc="Processing results")):
        generations = [output.outputs[i].text for i in range(n_sampling)]
        parsed = [parse_answer(gen) for gen in generations]
        em = [p == gold for p in parsed]
        passk = any(em)
        results.append({
            "question": q,
            "gold": gold,
            "generations": generations,
            "parsed": parsed,
            "exact_match": em,
            "pass@16": passk
        })
        
        # 第一个样本的调试信息
        if idx == 0:
            print("\n==== First Sample Debug Info ====")
            print(f"Prompt:\n{prompts[idx]}")
            print("\nGenerations:")
            for i, g in enumerate(generations):
                print(f"[{i+1}] {g}")
            print(f"\nGT: {gold}")
            print(f"Parsed: {parsed}")
            print(f"EM: {em}")
            print("===============================\n")
        
        row = {
            "question": q,
            "gt": gold,
        }
        for i in range(n_sampling):
            row[f"gen_{i+1}"] = generations[i]
            row[f"parse_{i+1}"] = parsed[i]
            row[f"em_{i+1}"] = int(em[i])
        csv_rows.append(row)
    
    passk_rate = sum(r["pass@16"] for r in results) / len(results)
    em_total = sum(sum(r["exact_match"]) for r in results) / (len(results) * n_sampling)
    passk_dict = {}
    for k in [1,2,4,8,16]:
        count = 0
        for r in results:
            c = sum(r["exact_match"])
            count += pass_at_k(n_sampling, c, k)
        passk_dict[f"pass@{k}"] = count / len(results)
    print(f"pass@16: {passk_rate:.4f}, exact match: {em_total:.4f}")
    for k in [1,2,4,8,16]:
        print(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}")
    with open(os.path.join(output_dir, "gsm8k_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"pass@16: {passk_rate:.4f}\nexact_match: {em_total:.4f}\n")
        for k in [1,2,4,8,16]:
            f.write(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}\n")
    csv_path = os.path.join(output_dir, "result.csv")
    with open(csv_path, "w", newline='') as csvfile:
        # 只保存0/1的匹配结果
        fieldnames = [f"em_{i+1}" for i in range(n_sampling)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            # 只提取em_1到em_16的值
            em_row = {f"em_{i+1}": row[f"em_{i+1}"] for i in range(n_sampling)}
            writer.writerow(em_row)
    # 写入base_out/passk.json
    if model_name is not None and base_out is not None:
        passk_path = os.path.join(base_out, "passk.json")
        update_passk_json(passk_path, model_name, passk_dict, overwrite=overwrite)

def is_job_running_or_done(model_out_dir):
    # 只检查result.csv是否存在
    result_csv = os.path.join(model_out_dir, "result.csv")
    return os.path.exists(result_csv)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gsm8k_path",
        type=str,
        default="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/gsm8k_test.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--n_sampling",
        type=int,
        default=16
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--submit_jobs",
        action="store_true",
        help="If set, submit slurm jobs for all models."
    )
    parser.add_argument(
        "--reforce",
        action="store_true",
        help="If set, rerun evaluation even if result.csv already exists."
    )
    args = parser.parse_args()

    if args.submit_jobs:
        BASE_OUT = "/mnt/sharefs/users/haolong.jia/result/gsm8k_pass16"
        os.makedirs(BASE_OUT, exist_ok=True)
        GSM8K_PATH = args.gsm8k_path
        for model_path, model_name in Model_map.items():
            model_out_dir = os.path.join(BASE_OUT, model_name)
            os.makedirs(model_out_dir, exist_ok=True)
            if not args.reforce and is_job_running_or_done(model_out_dir):
                print(f"Skip {model_name}: result.csv already exists. Use --reforce to rerun.")
                continue
            job_name = f"gsm8k_{model_name}"
            job_script = os.path.join(model_out_dir, f"{job_name}.sh")
            with open(job_script, 'w') as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={model_out_dir}/slurm.out
#SBATCH --error={model_out_dir}/slurm.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

cd {os.getcwd()}
which python
export TOKENIZERS_PARALLELISM=false
python3 -u {os.path.abspath(__file__)} \
    --gsm8k_path {GSM8K_PATH} \
    --output_dir {model_out_dir} \
    --n_sampling 16 \
    --model_path {model_path} \
    --model_name {model_name}
""")
            os.system(f"sbatch {job_script}")
            print(f"Submitted job for {model_name}")
    else:
        assert args.model_path is not None and args.output_dir is not None
        # 需要传入model_name和base_out
        base_out = "/mnt/sharefs/users/haolong.jia/result/gsm8k_pass16"
        model_name = args.model_name
        overwrite = args.reforce
        evaluate_gsm8k(args.gsm8k_path, args.output_dir, args.n_sampling, args.model_path, model_name, base_out, overwrite=overwrite)
