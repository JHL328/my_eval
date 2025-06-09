import os
import json
import csv
import fcntl
from tqdm import tqdm

def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def calculate_passk_from_csv(csv_path, n_sampling=32):
    passk_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    total_pass_at_k = {k: 0.0 for k in [1, 2, 4, 8, 16, 32]}
    for row in rows:
        correct_count = sum(int(row[f'em_{i+1}']) for i in range(n_sampling))
        for k in [1, 2, 4, 8, 16, 32]:
            if k <= n_sampling:
                total_pass_at_k[k] += pass_at_k(n_sampling, correct_count, k)
    n_samples = len(rows)
    for k in [1, 2, 4, 8, 16, 32]:
        passk_dict[f'pass@{k}'] = total_pass_at_k[k] / n_samples
    return passk_dict

def update_passk_json(passk_path, model_name, passk_result):
    with open(passk_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            all_results = json.load(f)
        except Exception:
            all_results = {}
        all_results[model_name] = passk_result
        f.seek(0)
        f.truncate()
        json.dump(all_results, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)

def main():
    BASE_OUT = "/mnt/sharefs/users/haolong.jia/result/gpqa_pass32"
    passk_path = os.path.join(BASE_OUT, "passk.json")
    model_dirs = []
    for item in os.listdir(BASE_OUT):
        item_path = os.path.join(BASE_OUT, item)
        if os.path.isdir(item_path):
            csv_file = os.path.join(item_path, "result.csv")
            if os.path.exists(csv_file):
                model_dirs.append(item)
    print(f"Found {len(model_dirs)} models with result.csv files")
    for model_name in tqdm(sorted(model_dirs), desc="Processing models"):
        csv_file = os.path.join(BASE_OUT, model_name, "result.csv")
        try:
            passk_dict = calculate_passk_from_csv(csv_file, n_sampling=32)
            update_passk_json(passk_path, model_name, passk_dict)
        except Exception as e:
            print(f"\nError processing {model_name}: {e}")
    print(f"\nDone! Processed {len(model_dirs)} models.")

if __name__ == "__main__":
    main()
