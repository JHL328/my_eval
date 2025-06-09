import os
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import sys
import csv
import fcntl
import time

# 5-shot CoT fewshot examples from gpqa_diamond train split (前5条)
FEWSHOT_EXAMPLES = [
    {
        "question": "A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?\n\nA. R-loops.\nB. lariat.\nC. polyA tail.\nD. antisense.\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "Let's think step by step. The text describes the dystrophin gene and the FDA-approved oligonucleotide therapy that causes exon skipping by creating a functional, albeit shorter, dystrophin protein. Morpholino is bound to the pre-mRNA in an antisense orientation. Every splicing mechanism creates the lariat molecule that is circular with a 3' tail and soon degraded. The spliced RNA is polyadenylated at the 3' end. R-loops are triple helix of DNA and the pre-mRNA and a consequence of the RNA transcription, not splicing and RNA maturation. Therefore, the answer is \\boxed{A}."
    },
    {
        "question": "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\nA. 10^-9 eV\nB. 10^-8 eV\nC. 10^-11 eV\nD. 10^-4 eV\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "Let's think step by step. According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the width of the energy level. With Delta t=10^-9 s==> Delta E1= 3.3 10^-7 ev. And Delta t=10^-8 s gives Delta E2=3.3*10^-8 eV. Therefore, the energy difference between the two states must be significantly greater than 10^-7 ev. So the answer is \\boxed{D}."
    },
    {
        "question": "How many of the following compounds exhibit optical activity?\n1-methyl-4-(prop-1-en-2-yl)cyclohex-1-ene\n2,3,3,3-tetrafluoroprop-1-ene\ndi(cyclohex-2-en-1-ylidene)methane\n5-(5-methylhexan-2-ylidene)cyclopenta-1,3-diene\n3-(2-methylbut-1-en-1-ylidene)cyclohex-1-ene\n[1,1'-biphenyl]-3,3'-diol\n8,8-dichlorobicyclo[4.2.0]octan-7-one\ncyclopent-2-en-1-one\n\nA. 3\nB. 4\nC. 5\nD. 6\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "Let's think step by step. The compounds 1-methyl-4-(prop-1-en-2-yl)cyclohex-1-ene, 3-(2-methylbut-1-en-1-ylidene)cyclohex-1-ene, di(cyclohex-2-en-1-ylidene)methane, and 8,8-dichlorobicyclo[4.2.0]octan-7-one are chiral molecules, and thus will be optically active. All the others have a mirror plane of symmetry, and will be achiral. Therefore, the answer is \\boxed{B}."
    },
    {
        "question": "A coating is applied to a substrate resulting in a perfectly smooth surface. The measured contact angles of this smooth coating are 132° and 102° for water and hexadecane respectively. The coating formulation is then modified and when now applied to the same type of substrate, a rough surface is produced. When a droplet of water or oil sits on the rough surface, the wettability of the surface can now be described by the Cassie-Baxter state. The water contact angle on the rough surface is now 148°. What would be the best estimate of the contact angle of a droplet of octane on the rough surface?\n\nA. 129°\nB. 134°\nC. 124°\nD. 139°\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "Let's think step by step. In the Cassie-Baxter state, droplets are in contact with a non-uniform surface: some of the droplet is in contact with the coating and some with air. The contact angle (θCB) of a droplet in the Cassie-Baxter state is given by: cosθCB = f1.cosθ1 + f2.cosθ2, where f1 and f2 are the area fractions of the two components of the surface, in this case coating (f1) and air (f2). θ1 is the contact angle if the droplet was purely in contact with the coating, and θ2 is the contact angle if the droplet was purely in contact with air. First we need to calculate the f1 and f2 using the data we are given for water. We have θCB = 148°, θ1 = 132°, and θ2 is taken to be 180° (contact angle on air). We then have cos(148) = f1.cos(132) + f2.cos(180). By using f1 + f2 = 1, we can solve to give f1 = 0.46 and f2 = 0.54. Next we need to calculate the contact angle of hexadecane on the rough surface, we have θ1 = 102°, f1 = 0.46, f2 = 0.54, and θ2 is taken to be 180° (contact angle on air). Therefore, θCB = 129° for hexadecane. The question however asks about a droplet of octane. Octane is a shorter oil molecule than hexadecane and has a lower surface tension than hexadecane. For a given surface, the contact angle of octane is therefore always lower than for hexadecane. Therefore, the answer is \\boxed{C}."
    },
    {
        "question": "In a parallel universe where a magnet can have an isolated North or South pole, Maxwell's equations look different. But, specifically, which of those equations are different?\n\nA. The ones related to the circulation of the electric field and the divergence of the magnetic field.\nB. The ones related to the divergence and the curl of the magnetic field.\nC. The one related to the divergence of the magnetic field.\nD. The one related to the circulation of the magnetic field and the flux of the electric field.\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "Let's think step by step. Let's call E and B the electric and magnetic fields, respectively: The ones related to the circulation of the electric field and the divergence of the magnetic field is correct, since knowing that magnets can have an isolated pole means that magnetic monopoles exist and, thus, the contributions of magnetic charges and magnetic currents must be included in the equations. The way to include them is to \"symmetry-copy\" the other equations, with the following dictionary: E <-> B; electric charge <-> magnetic charge; electric current <-> magnetic current. In this way, the equations that become modified, with added terms, are the ones related to the circulation (or curl, in differential form) of E, and to the divergence (or flux in integral form) of B. The ones related to the divergence and the curl of the magnetic field is incorrect, because the one with the curl does not change, since it already includes all symmetric contributions appearing in its symmetric equation (curl of electric field). The one related to the divergence of the magnetic field is incorrect because that equation does get changed, but it's not the only one; the equation for the curl (or circulation) of E also changes. The one related to the circulation of the magnetic field and the flux of the electric field is incorrect because none of those equations are changed, since they already include the symmetric terms appearing in their symmetric equations (circulation of E and flux of B). Therefore, the answer is \\boxed{A}."
    }
]

# Prompt construction
FEWSHOT_PROMPT = ""
for ex in FEWSHOT_EXAMPLES:
    FEWSHOT_PROMPT += f"Q: {ex['question']}\nA: {ex['target']}\n\n"

# Parser for extracting boxed answer (A/B/C/D)
def parse_answer(text):
    # 优先匹配 \boxed{A} 或 \boxed {A}，容忍空格、单双斜杠、花括号、全角括号、以及A-D的大小写
    m = re.search(r"\\?\\?boxed[\s\u3000]*[\{\(\uff08][\s\u3000]*([A-Da-d])[\s\u3000]*[\}\)\uff09]", text)
    if m:
        return m.group(1).upper()
    # 匹配 LaTeX 形式 $\boxed{A}$
    m2 = re.search(r"\\?\$?\\?boxed[\s\u3000]*[\{\(\uff08][\s\u3000]*([A-Da-d])[\s\u3000]*[\}\)\uff09]", text)
    if m2:
        return m2.group(1).upper()
    
    # 只在最后5行找单独一行的A/B/C/D，且要有合理的上下文
    lines = text.strip().splitlines()
    if len(lines) > 0:
        # 查看最后5行
        last_lines = lines[-5:]
        for i, line in enumerate(last_lines):
            # 检查是否是单独的选项字母
            m3 = re.match(r"^[\s\u3000]*([A-Da-d])[\s\u3000]*\.?[\s\u3000]*$", line)
            if m3:
                # 检查上下文：前面应该有"answer"、"因此"、"所以"、"therefore"等词
                context = " ".join(last_lines[:i+1]).lower() if i > 0 else text[-200:].lower()
                if any(keyword in context for keyword in ["answer", "therefore", "thus", "so", "因此", "所以", "答案"]):
                    return m3.group(1).upper()
    
    # 匹配如"The answer is A"或"答案是B"，但要求更严格的格式
    # 必须是明确的答案陈述，而不是随意提到的字母
    m4 = re.search(r"(?:the\s+)?answer\s+is\s*:?\s*\\?\\?boxed[\s\u3000]*[\{\(\uff08]?\s*([A-Da-d])[\s\u3000]*[\}\)\uff09]?", text, re.IGNORECASE)
    if m4:
        return m4.group(1).upper()
    
    # 更严格的答案陈述匹配
    m5 = re.search(r"(?:因此|所以|therefore|thus|hence).*?(?:答案|answer)\s*[:：是为]\s*([A-Da-d])\b", text, re.IGNORECASE | re.DOTALL)
    if m5:
        # 确保这个匹配在文本的后半部分
        match_pos = m5.start()
        if match_pos > len(text) * 0.5:  # 在文本后半部分
            return m5.group(1).upper()
    
    # 如果什么都没匹配到，返回空字符串
    return ""

# 采样参数
SAMPLING_PARAMS = dict(
    temperature=0.6,
    top_p=0.95,
    n=32,
    max_tokens=4096,
    stop=["Q:", "</s>", "<|im_end|>", "\n\nQ:", "\n\nHuman:", "\n\nAssistant:", "Human:", "Assistant:"],
    seed=42,
)

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

def evaluate_gpqa(gpqa_path, output_dir, n_sampling=32, model_path=None, model_name=None, base_out=None, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, dtype="auto", tensor_parallel_size=1)
    print("Model loaded successfully!")

    results = []
    csv_rows = []
    with open(gpqa_path, 'r') as f:
        data = [json.loads(line) for line in f]

    print("Preparing prompts...")
    prompts = []
    golds = []
    questions = []
    for item in tqdm(data, desc="Preparing data"):
        q = item["question"]
        gold = item["answer"].strip().upper()
        prompt = FEWSHOT_PROMPT + f"Q: {q}\nA: Let's think step by step."
        prompts.append(prompt)
        golds.append(gold)
        questions.append(q)

    print(f"Running inference on {len(prompts)} prompts...")
    params = SamplingParams(**{**SAMPLING_PARAMS, "n": n_sampling})
    outputs = llm.generate(prompts, params)

    print("Processing results...")
    debug_n = 3  # 打印前三个
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
            "pass@32": passk
        })
        if idx < debug_n:
            print(f"\n==== Sample Debug Info #{idx+1} ====")
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

    passk_rate = sum(r["pass@32"] for r in results) / len(results)
    em_total = sum(sum(r["exact_match"]) for r in results) / (len(results) * n_sampling)
    passk_dict = {}
    for k in [1,2,4,8,16,32]:
        count = 0
        for r in results:
            c = sum(r["exact_match"])
            count += pass_at_k(n_sampling, c, k)
        passk_dict[f"pass@{k}"] = count / len(results)
    print(f"pass@32: {passk_rate:.4f}, exact match: {em_total:.4f}")
    for k in [1,2,4,8,16,32]:
        print(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}")
    with open(os.path.join(output_dir, "gpqa_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"pass@32: {passk_rate:.4f}\nexact_match: {em_total:.4f}\n")
        for k in [1,2,4,8,16,32]:
            f.write(f"pass@{k}: {passk_dict[f'pass@{k}']:.4f}\n")
    csv_path = os.path.join(output_dir, "result.csv")
    with open(csv_path, "w", newline='') as csvfile:
        fieldnames = [f"em_{i+1}" for i in range(n_sampling)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            em_row = {f"em_{i+1}": row[f"em_{i+1}"] for i in range(n_sampling)}
            writer.writerow(em_row)
    if model_name is not None and base_out is not None:
        passk_path = os.path.join(base_out, "passk.json")
        update_passk_json(passk_path, model_name, passk_dict, overwrite=overwrite)

def is_job_running_or_done(model_out_dir):
    result_csv = os.path.join(model_out_dir, "result.csv")
    return os.path.exists(result_csv)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpqa_path",
        type=str,
        default="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/gpqa_diamond_test.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--n_sampling",
        type=int,
        default=32
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
        BASE_OUT = "/mnt/sharefs/users/haolong.jia/result/gpqa_pass32"
        os.makedirs(BASE_OUT, exist_ok=True)
        GPQA_PATH = args.gpqa_path
        from model import Model_map
        for model_path, model_name in Model_map.items():
            model_out_dir = os.path.join(BASE_OUT, model_name)
            os.makedirs(model_out_dir, exist_ok=True)
            if not args.reforce and is_job_running_or_done(model_out_dir):
                print(f"Skip {model_name}: result.csv already exists. Use --reforce to rerun.")
                continue
            job_name = f"gpqa_{model_name}"
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
    --gpqa_path {GPQA_PATH} \
    --output_dir {model_out_dir} \
    --n_sampling 32 \
    --model_path {model_path} \
    --model_name {model_name}
""")
            os.system(f"sbatch {job_script}")
            print(f"Submitted job for {model_name}")
    else:
        assert args.model_path is not None and args.output_dir is not None
        base_out = "/mnt/sharefs/users/haolong.jia/result/gpqa_pass32"
        model_name = args.model_name
        overwrite = args.reforce
        evaluate_gpqa(args.gpqa_path, args.output_dir, args.n_sampling, args.model_path, model_name, base_out, overwrite=overwrite)
