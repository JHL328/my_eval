import os
import pandas as pd
import numpy as np
import json
import argparse
import fcntl

def concat_csvs(model_dir):
    csv_files = [f for f in os.listdir(model_dir) if f.endswith('.csv') and f != 'result.csv']
    dfs = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(model_dir, f))
        dfs.append(df)
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
        result_df.to_csv(os.path.join(model_dir, 'result.csv'), index=False)
        print(f"Saved concatenated result to {os.path.join(model_dir, 'result.csv')}")
    else:
        print(f"No CSV files found in {model_dir}")

def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def update_passk_json(passk_path, model_name, passk_result):
    # add lock to update passk.json
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

def calc_passk(model_dir, k_values=[1,2,4,8,16]):
    result_csv = os.path.join(model_dir, 'result.csv')
    if not os.path.exists(result_csv):
        print(f"result.csv not found in {model_dir}, skipping pass@k calculation.")
        return None
    # read as numpy array, no header
    data = pd.read_csv(result_csv, header=None).values
    all_samples = data.tolist()
    results = {}
    for k in k_values:
        if k <= 16:
            pass_at_k_scores = []
            for sample_attempts in all_samples:
                n = len(sample_attempts)
                c = sum(sample_attempts)
                sample_pass_k = pass_at_k(n, c, k)
                pass_at_k_scores.append(sample_pass_k)
            results[f"pass@{k}"] = float(np.mean(pass_at_k_scores))
    print(f"Calculated pass@k for {model_dir}")
    # add lock to update passk.json
    passk_path = os.path.join(os.path.dirname(model_dir), 'passk.json')
    model_name = os.path.basename(model_dir)
    update_passk_json(passk_path, model_name, results)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to process')
    args = parser.parse_args()
    output_dir = args.output_dir

    all_passk = {}
    for model_name in os.listdir(output_dir):
        model_dir = os.path.join(output_dir, model_name)
        if os.path.isdir(model_dir):
            concat_csvs(model_dir)
            passk_result = calc_passk(model_dir)
            if passk_result is not None:
                all_passk[model_name] = passk_result

    with open(os.path.join(output_dir, 'passk.json'), 'w') as f:
        json.dump(all_passk, f, indent=2)
    print(f"Saved all pass@k results to {os.path.join(output_dir, 'passk.json')}")

if __name__ == "__main__":
    main()
