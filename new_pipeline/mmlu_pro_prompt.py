import json
from datasets import load_dataset

choices = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"
]

def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

subjects = [
    "biology", "business", "chemistry", "computer science", "economics", "engineering",
    "health", "history", "law", "math", "other", "philosophy", "physics", "psychology"
]

dataset_test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
dataset_val = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

all_prompts = {}
for subject in subjects:
    # 取validation split中同subject的前5个few-shot
    val_examples = [ex for ex in dataset_val if ex["category"] == subject][:5]
    fewshot = ""
    for ex in val_examples:
        fewshot += format_cot_example(ex, including_answer=True) + "\n"
    # test部分
    subject_examples = [ex for ex in dataset_test if ex["category"] == subject]
    all_prompts[subject] = []
    for idx, ex in enumerate(subject_examples):
        prompt = fewshot + format_cot_example(ex, including_answer=False)
        all_prompts[subject].append({
            "idx": idx,
            "prompt": prompt,
            "answer": ex["answer"]
        })

with open("/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_pro_prompts.json", "w") as f:
    json.dump(all_prompts, f, indent=2)
