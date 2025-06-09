import json
from datasets import get_dataset_config_names, load_dataset

# 读取 bbh prompt 里的 task
with open("/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/bbh_cot_prompts.json", "r", encoding="utf-8") as f:
    bbh_subjects = set(json.load(f).keys())

# 读取 hf 上 bbh 的所有 task
hf_subjects = set(get_dataset_config_names("lukaemon/bbh"))

print("Only in bbh_cot_prompts.json:", bbh_subjects - hf_subjects)
print("Only in hf:", hf_subjects - bbh_subjects)
print("In both:", bbh_subjects & hf_subjects)
print("bbh_cot_prompts.json 任务数:", len(bbh_subjects))
print("hf 任务数:", len(hf_subjects))

# 统计每个任务的 test split 样本数
subject_counts = {}
total_samples = 0
for subject in sorted(bbh_subjects):
    dataset = load_dataset(
        "lukaemon/bbh",
        subject,
        cache_dir="/mnt/sharefs/users/haolong.jia/eval_data",
        trust_remote_code=True
    )
    n = len(dataset['test'])
    subject_counts[subject] = n
    total_samples += n
    print(f"{subject}: {n} samples in test split")

print("\n总任务数:", len(subject_counts))
print("总 test 样本数:", total_samples)

train_dataset = load_dataset(
    "EleutherAI/drop",
    split="train",
    cache_dir="/mnt/sharefs/users/haolong.jia/eval_data",
    trust_remote_code=True
)
print(train_dataset[0])