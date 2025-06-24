import json
from datasets import get_dataset_config_names, load_dataset
import os
import pandas as pd
import numpy as np
import fcntl



def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def update_passk_json_atomic(passk_path, model_name, passk_dict):
    os.makedirs(os.path.dirname(passk_path), exist_ok=True)
    with open(passk_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            all_results = json.load(f)
        except Exception:
            all_results = {}
        all_results[model_name] = passk_dict
        f.seek(0)
        f.truncate()
        json.dump(all_results, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)

def main():
    output_root = "/mnt/sharefs/users/haolong.jia/result/mmlu"
    passk_path = os.path.join(output_root, "passk.json")
    model_names = [
        "untidy_dish_57_71525","chocolate_brushstroke_67_71525","classy_fractals_55_71525",
        "close_pretzel_63_71525","complicated_tetrad_60_71525","disagreeable_cookie_56_71525",
        "futuristic_composition_59_71525","marxist_configuration_66_71525","meaty_refrain_61_71525",
        "novel_sine_64_71525","occasional_emmentaler_62_71525","rectilinear_firewall_65_71525",
        "sharing_radian_58_71525"
    ]
    ks = [1,2,4,8,16]
    for model_name in model_names:
        model_dir = os.path.join(output_root, model_name)
        result_csv = os.path.join(model_dir, "result.csv")
        if not os.path.exists(result_csv):
            print(f"[SKIP] {model_name}: result.csv not found")
            continue
        try:
            df = pd.read_csv(result_csv)
            sample_cols = [col for col in df.columns if col.isdigit() or col.startswith("em_")]
            n_sampling = len(sample_cols)
            if n_sampling == 0:
                print(f"[ERROR] {model_name}: cannot infer n_sampling")
                continue
            valid_ks = [k for k in ks if k <= n_sampling]
            total_pass_at_k = {k: 0.0 for k in valid_ks}
            for _, row in df.iterrows():
                if all(col.isdigit() for col in df.columns[:n_sampling]):
                    correct_count = sum(int(row[str(i)]) for i in range(n_sampling))
                else:
                    correct_count = sum(int(row[f'em_{i+1}']) for i in range(n_sampling))
                for k in valid_ks:
                    total_pass_at_k[k] += pass_at_k(n_sampling, correct_count, k)
            n_samples = len(df)
            passk_dict = {f'pass@{k}': (total_pass_at_k[k] / n_samples if n_samples > 0 else 0.0) for k in valid_ks}
            update_passk_json_atomic(passk_path, model_name, passk_dict)
            print(f"[OK] {model_name} updated passk.json")
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")

if __name__ == "__main__":
    main()