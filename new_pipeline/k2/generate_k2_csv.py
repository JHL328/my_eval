import os
import json
import pandas as pd
import sys
import re # Added for regex parsing of model names

# Ensure parent directory is in sys.path for k2 imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from k2 import k2_model
from k2 import k2_evaluate # Import k2_evaluate to get task configs

# =====================
# config
# =====================
# metric name mapping: original metric name -> display name
METRIC_NAME_MAP = {
    'pass@1': 'pass@1',
    'pass@2': 'pass@2',
    'pass@4': 'pass@4',
    'pass@8': 'pass@8',
    'pass@16': 'pass@16',
    'pass@32': 'pass@32',
    'pass@64': 'pass@64',
    'acc_norm,none': 'acc_norm',
    'f1,none': 'f1',
    'exact_match,remove_whitespace': 'exact_match',
}

# each benchmark's result.json path and metrics to extract
# This is hardcoded for clarity, mirroring the approach of the original script.
K2_BENCHMARKS = {
    'bbh': {
        'path': os.path.join(k2_evaluate.TASK_CONFIGS['bbh']['output_dir'], "result.json"),
        'metrics': ['pass@1'],
    },
    'mmlu': {
        'path': os.path.join(k2_evaluate.TASK_CONFIGS['mmlu']['output_dir'], "result.json"),
        'metrics': ['pass@1'],
    },
    'mmlu_flan': {
        'path': os.path.join(k2_evaluate.TASK_CONFIGS['mmlu_flan']['output_dir'], "result.json"),
        'metrics': ['pass@1'],
    },
    'mmlu_pro': {
        'path': os.path.join(k2_evaluate.TASK_CONFIGS['mmlu_pro']['output_dir'], "result.json"),
        'metrics': ['pass@1'],
    },
    'gsm8k': {
        'path': os.path.join(k2_evaluate.TASK_CONFIGS['gsm8k']['output_dir'], "result.json"),
        'metrics': ['pass@1', 'pass@4', 'pass@16'],
    },
    'math500': {
        'path': os.path.join(k2_evaluate.TASK_CONFIGS['math500']['output_dir'], "result.json"),
        'metrics': ['pass@1', 'pass@4', 'pass@16'],
    }
}

# =====================
# dynamically generate model list (name, index)
# =====================
MODEL_INFOS = []
# Get model names and sort them alphabetically for consistent ordering
sorted_model_names = sorted(list(k2_model.model_map.values()))

# Assign a simple 1-based sequential index
for i, model_name_full in enumerate(sorted_model_names, 1):
    MODEL_INFOS.append((model_name_full, i))

# No further sorting is needed as models are already ordered alphabetically.

# =====================
# generate header order
# =====================
header = ['Index', 'Model Name'] # Added Model Name column
for bench, info in K2_BENCHMARKS.items():
    for metric in info['metrics']:
        display_metric = METRIC_NAME_MAP.get(metric, metric)
        header.append(f"{bench}_{display_metric}")

# =====================
# main table generation
# =====================
# result table
rows = []
for model_name_full, index in MODEL_INFOS:
    row = {'Index': index, 'Model Name': model_name_full} # Populate Model Name
    for bench, info in K2_BENCHMARKS.items():
        # 读取json
        try:
            with open(info['path'], 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load {info['path']}: {e}")
            for metric in info['metrics']:
                display_metric = METRIC_NAME_MAP.get(metric, metric)
                row[f"{bench}_{display_metric}"] = -1
            continue
        
        # In k2_evaluate.py, result.json is a dict where keys are model names
        if model_name_full in data:
            model_data = data[model_name_full]
            for metric in info['metrics']:
                display_metric = METRIC_NAME_MAP.get(metric, metric)
                value = -1
                if metric in model_data:
                    value = model_data[metric]
                else:
                    print(f"⚠️ Missing {model_name_full} {metric} in {info['path']}")
                row[f"{bench}_{display_metric}"] = value
        else:
            print(f"⚠️ Missing model {model_name_full} in {info['path']}")
            for metric in info['metrics']:
                display_metric = METRIC_NAME_MAP.get(metric, metric)
                row[f"{bench}_{display_metric}"] = -1
    rows.append(row)

# =====================
# save as csv, column order consistent with header
# =====================
out_path = '/mnt/sharefs/users/haolong.jia/result-k2/k2_summary_all.csv' # Updated output path
df = pd.DataFrame(rows)
df = df[header] # force column order

# save as csv
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"🎉 Saved to {out_path}") 