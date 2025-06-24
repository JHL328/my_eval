import os
import json
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model_map

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

BENCHMARKS = {
    'bbh_3shot_cot': {
        'path': '/mnt/sharefs/users/haolong.jia/result/bbh_pass16/passk.json',
        'metrics': ['pass@1', 'pass@4'],
    },
    'mmlu_5shot_wcot': {
        'path': '/mnt/sharefs/users/haolong.jia/result/mmlu/passk.json',
        'metrics': ['pass@1', 'pass@2'],
    },
    'mmlu_4shot_cot': {
        'path': '/mnt/sharefs/users/haolong.jia/result/mmlu_flan_pass16/passk.json',
        'metrics': ['pass@1', 'pass@2'],
    },
    'mmlu_pro_5shot_cot': {
        'path': '/mnt/sharefs/users/haolong.jia/result/mmlu_pro_pass16/passk.json',
        'metrics': ['pass@1', 'pass@2'],
    },
    'drop': {
        'path': '/mnt/sharefs/users/haolong.jia/result/drop/result.json',
        'metrics': ['f1,none'],
    },
    'arc_easy': {
        'path': '/mnt/sharefs/users/haolong.jia/result/arc_easy/result.json',
        'metrics': ['acc_norm,none'],
    },
    'arc_challenge': {
        'path': '/mnt/sharefs/users/haolong.jia/result/arc_challenge/result.json',
        'metrics': ['acc_norm,none'],
    },
    'hellaswag': {
        'path': '/mnt/sharefs/users/haolong.jia/result/hellaswag/result.json',
        'metrics': ['acc_norm,none'],
    },
    'piqa': {
        'path': '/mnt/sharefs/users/haolong.jia/result/piqa/result.json',
        'metrics': ['acc_norm,none'],
    },
    'winogrande': {
        'path': '/mnt/sharefs/users/haolong.jia/result/winogrande/result.json',
        'metrics': ['acc_norm,none'],
    },
    'triviaqa': {
        'path': '/mnt/sharefs/users/haolong.jia/result/triviaqa/result.json',
        'metrics': ['exact_match,remove_whitespace'],
    },
    'nq_open': {
        'path': '/mnt/sharefs/users/haolong.jia/result/nq_open/result.json',
        'metrics': ['exact_match,remove_whitespace'],
    },
    'agieval': {
        'path': '/mnt/sharefs/users/haolong.jia/result/agieval/result.json',
        'metrics': ['acc_norm,none'],
    },
    'commonsense_qa': {
        'path': '/mnt/sharefs/users/haolong.jia/result/commonsense_qa/result.json',
        'metrics': ['acc_norm,none'],
    },
    'openbookqa': {
        'path': '/mnt/sharefs/users/haolong.jia/result/openbookqa/result.json',
        'metrics': ['acc_norm,none'],
    },
    'social_iqa': {
        'path': '/mnt/sharefs/users/haolong.jia/result/social_iqa/result.json',
        'metrics': ['acc_norm,none'],
    },
    'truthfulqa_mc2': {
        'path': '/mnt/sharefs/users/haolong.jia/result/truthfulqa_mc2/result.json',
        'metrics': ['acc_norm,none'],
    },
    'math500': {
        'path': '/mnt/sharefs/users/haolong.jia/result/math500_pass64/passk.json',
        'metrics': ['pass@1', 'pass@8', 'pass@16', 'pass@32'],
    },
    'gsm8k': {
        'path': '/mnt/sharefs/users/haolong.jia/result/gsm8k_pass16/passk.json',
        'metrics': ['pass@1', 'pass@4', 'pass@8', 'pass@16'],
    },
}

OPEN_SOURCE_MODEL_NAMES = [
    "Llama-3.2-3B","Qwen2.5-1.5B","SmolLM2-1.7B","Llama-3.2-1B",
    "Mistral-7B","Qwen2.5-3B","Qwen3-1.7B-Base","Qwen3-4B-Base"
]

header = ['Model']
for bench, info in BENCHMARKS.items():
    for metric in info['metrics']:
        display_metric = METRIC_NAME_MAP.get(metric, metric)
        header.append(f"{bench}_{display_metric}")

rows = []
for model_name in OPEN_SOURCE_MODEL_NAMES:
    row = {'Model': model_name}
    for bench, info in BENCHMARKS.items():
        try:
            with open(info['path'], 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load {info['path']}: {e}")
            for metric in info['metrics']:
                display_metric = METRIC_NAME_MAP.get(metric, metric)
                row[f"{bench}_{display_metric}"] = "-1"
            continue
        for metric in info['metrics']:
            display_metric = METRIC_NAME_MAP.get(metric, metric)
            value = "-1"
            if model_name in data and metric in data[model_name]:
                value = str(data[model_name][metric])
            else:
                print(f"⚠️ Missing {model_name} {metric} in {info['path']}")
            row[f"{bench}_{display_metric}"] = value
    rows.append(row)

out_path = '/mnt/sharefs/users/haolong.jia/result/opensource.csv'
df = pd.DataFrame(rows)
df = df[header]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"🎉 Saved to {out_path}")
