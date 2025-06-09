import os
import json
import csv
import glob
import shutil
import numpy as np
import pandas as pd

from model import Model_map

RESULT_ROOT = "/mnt/sharefs/users/haolong.jia/result/math500_pass64"
PASSK_JSON = os.path.join(RESULT_ROOT, "passk.json")
K_LIST = [1, 2, 4, 8, 16, 32, 64] # change it 

def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

all_results = {}

for model_path, model_name in Model_map.items():
    model_dir = os.path.join(RESULT_ROOT, model_name)
    if not os.path.isdir(model_dir):
        continue
    math500_dir = os.path.join(model_dir, "math500")
    base_name = os.path.basename(model_path)
    target_dir = os.path.join(math500_dir, base_name)
    if os.path.isdir(target_dir):
        # 处理jsonl
        jsonl_files = glob.glob(os.path.join(target_dir, "*.jsonl"))
        if jsonl_files:
            shutil.move(jsonl_files[0], os.path.join(model_dir, "sample.jsonl"))
        # 处理json
        json_files = glob.glob(os.path.join(target_dir, "*.json"))
        if json_files:
            shutil.move(json_files[0], os.path.join(model_dir, "result.json"))
        shutil.rmtree(target_dir)
    # 读取sample.jsonl，生成result.csv
    sample_jsonl = os.path.join(model_dir, "sample.jsonl")
    if not os.path.exists(sample_jsonl):
        continue
    all_scores = []
    with open(sample_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            scores = obj.get("score", [])
            if isinstance(scores, list):
                row = [1 if s is True or s == 1 else 0 for s in scores]
            else:
                row = [1 if scores is True or scores == 1 else 0]
            all_scores.append(row)
    csv_path = os.path.join(model_dir, "result.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_scores)
    # 计算pass@k
    data = pd.read_csv(csv_path, header=None).values
    all_samples = data.tolist()
    results = {}
    for k in K_LIST:
        pass_at_k_scores = []
        for sample_attempts in all_samples:
            n = len(sample_attempts)
            c = sum(sample_attempts)
            sample_pass_k = pass_at_k(n, c, k)
            pass_at_k_scores.append(sample_pass_k)
        results[f"pass@{k}"] = float(np.mean(pass_at_k_scores))
    all_results[model_name] = results
# 写入passk.json
with open(PASSK_JSON, 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("文件整理、csv生成和pass@k计算完成！")

