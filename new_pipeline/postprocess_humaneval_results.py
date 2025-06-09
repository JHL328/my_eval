import os
import json
import glob
from collections import defaultdict
import numpy as np

# === 新增：自动读取 Model_map ===
import sys
sys.path.append(os.path.dirname(__file__))
from model import Model_map

def load_all_results(result_dir):
    # 匹配所有 evaluation_results_{idx_start}_{idx_end}.json
    pattern = os.path.join(result_dir, "evaluation_results_*_*.json")
    files = sorted(glob.glob(pattern))
    all_results = []
    for f in files:
        with open(f, "r") as fin:
            data = json.load(fin)
            if isinstance(data, dict) and "evaluation_results" in data:
                all_results.extend(data["evaluation_results"])
            elif isinstance(data, list):
                all_results.extend(data)
    return all_results

def calculate_pass_at_k(results, k_values=[1, 8, 16, 32, 64]):
    grouped_results = defaultdict(list)
    for result in results:
        grouped_results[result["task_id"]].append(result)
    def estimate_pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    base_pass_at_k = {}
    plus_pass_at_k = {}
    for k in k_values:
        base_correct = []
        plus_correct = []
        total_samples = []
        for task_id, task_results in grouped_results.items():
            n = len(task_results)
            base_passed = sum(1 for r in task_results if r["base_status"].lower() == "pass")
            plus_passed = sum(1 for r in task_results if r["base_status"].lower() == "pass" and r["plus_status"].lower() == "pass")
            total_samples.append(n)
            base_correct.append(base_passed)
            plus_correct.append(plus_passed)
        if min(total_samples) >= k:
            base_pass_at_k[f"pass@{k}"] = np.mean([
                estimate_pass_at_k(n, c, k) for n, c in zip(total_samples, base_correct)
            ])
            plus_pass_at_k[f"pass@{k}"] = np.mean([
                estimate_pass_at_k(n, c, k) for n, c in zip(total_samples, plus_correct)
            ])
    return base_pass_at_k, plus_pass_at_k

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/mnt/sharefs/users/haolong.jia/result/humaneval", type=str, help="主目录，如 /mnt/sharefs/users/haolong.jia/result/humaneval")
    args = parser.parse_args()

    all_model_base_passk = {}
    all_model_plus_passk = {}
    for model_path, model_name in Model_map.items():
        result_dir = os.path.join(args.root_dir, model_name)
        if not os.path.isdir(result_dir):
            print(f"[SKIP] {result_dir} 不存在")
            continue
        # 检查是否有 evaluation_results_*.json
        if not glob.glob(os.path.join(result_dir, "evaluation_results_*_*.json")):
            print(f"[SKIP] {result_dir} 没有分片结果")
            continue
        print(f"[MERGE] {model_name} in {result_dir}")
        all_results = load_all_results(result_dir)
        print(f"Loaded {len(all_results)} results.")
        base_pass_at_k, plus_pass_at_k = calculate_pass_at_k(all_results)
        ks = [1, 8, 16, 32, 64]
        metrics_txt_path = os.path.join(result_dir, "metrics_merged.txt")
        with open(metrics_txt_path, "w") as f:
            f.write("Base pass@k (robust):\n")
            for k in ks:
                if f"pass@{k}" in base_pass_at_k:
                    f.write(f"pass@{k}: {base_pass_at_k[f'pass@{k}']:.4f}\n")
            f.write("\nPlus pass@k (robust):\n")
            for k in ks:
                if f"pass@{k}" in plus_pass_at_k:
                    f.write(f"pass@{k}: {plus_pass_at_k[f'pass@{k}']:.4f}\n")
        print(f"Metrics saved to {metrics_txt_path}")
        summary_file = os.path.join(result_dir, "summary_merged.txt")
        with open(summary_file, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write("\nHumanEval (base tests):\n")
            for k, v in base_pass_at_k.items():
                f.write(f"  {k}: {v:.3f}\n")
            f.write("\nHumanEval+ (base + extra tests):\n")
            for k, v in plus_pass_at_k.items():
                f.write(f"  {k}: {v:.3f}\n")
        print(f"Summary saved to {summary_file}")
        # 新增：收集 base_pass_at_k 和 plus_pass_at_k 结果
        all_model_base_passk[model_name] = base_pass_at_k
        all_model_plus_passk[model_name] = plus_pass_at_k
    # 新增：分别保存 base_passk.json 和 plus_passk.json
    base_passk_json_path = os.path.join(args.root_dir, "base_passk.json")
    with open(base_passk_json_path, "w") as f:
        json.dump(all_model_base_passk, f, indent=2)
    print(f"All models' base pass@k saved to {base_passk_json_path}")
    plus_passk_json_path = os.path.join(args.root_dir, "plus_passk.json")
    with open(plus_passk_json_path, "w") as f:
        json.dump(all_model_plus_passk, f, indent=2)
    print(f"All models' plus pass@k saved to {plus_passk_json_path}") 