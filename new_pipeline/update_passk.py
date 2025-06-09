#!/usr/bin/env python3
import os
import json
import fcntl

def update_passk_json(passk_path, model_name, passk_result):
    """Update passk.json with file locking"""
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
    BASE_OUT = "/mnt/sharefs/users/haolong.jia/result/gsm8k_pass16"
    passk_path = os.path.join(BASE_OUT, "passk.json")
    
    # List of open source models to check
    open_source_models = [
        "Llama-3.2-1B",
        "Llama-3.2-3B",
        "Qwen2.5-1.5B",
        "Qwen2.5-3B",
        "Qwen3-1.7B-Base",
        "Qwen3-4B-Base",
        "SmolLM2-1.7B",
        "Mistral-7B"
    ]
    
    for model_name in open_source_models:
        metrics_file = os.path.join(BASE_OUT, model_name, "metrics.txt")
        if os.path.exists(metrics_file):
            print(f"Processing {model_name}...")
            passk_dict = {}
            with open(metrics_file, 'r') as f:
                for line in f:
                    if line.startswith("pass@"):
                        key, value = line.strip().split(": ")
                        passk_dict[key] = float(value)
            
            if passk_dict:
                print(f"  Found pass@k results: {passk_dict}")
                update_passk_json(passk_path, model_name, passk_dict)
                print(f"  Updated passk.json for {model_name}")
            else:
                print(f"  No pass@k results found in metrics.txt")
        else:
            print(f"Skipping {model_name}: metrics.txt not found")
    
    print("\nDone! passk.json has been updated.")

if __name__ == "__main__":
    main() 