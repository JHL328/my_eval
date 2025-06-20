import os
import json
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model_map

# =====================
# config
# =====================
# metric name mapping: original metric name -> display name
METRIC_NAME_MAP = {
    'pass@1': 'pass@1',
    'pass@2': 'pass@2',
    'pass@4': 'pass@4',
    'acc_norm,none': 'acc_norm',
    'f1,none': 'f1',
    'exact_match,remove_whitespace': 'exact_match',
}

# each benchmark's result.json path and metrics to extract
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
    # other benchmarks can be added here
}

# =====================
# dynamically generate model list (name, index, 71525)
# =====================
MODEL_INFOS = []
TARGET_SUFFIX = '_71525'
for model_name_full in Model_map.values():
    if model_name_full.endswith(TARGET_SUFFIX):
        base = model_name_full[:-len(TARGET_SUFFIX)]
        parts = base.rsplit('_', 1)
        if len(parts) == 2:
            name = parts[0]
            try:
                index = int(parts[1])
                MODEL_INFOS.append((name, index, 71525))
            except Exception:
                print(f"⚠️ Warning: index parse failed for {model_name_full}")
        else:
            print(f"⚠️ Warning: model name {model_name_full} does not match 'name_index_71525' pattern.")

# sort by index, ensure Index column is increasing
MODEL_INFOS.sort(key=lambda x: x[1])

# =====================
# generate header order
# =====================
header = ['Index']
for bench, info in BENCHMARKS.items():
    for metric in info['metrics']:
        display_metric = METRIC_NAME_MAP.get(metric, metric)
        header.append(f"{bench}_{display_metric}")

# =====================
# main table generation
# =====================
def make_model_key(name, index, suffix):
    return f"{name}_{index}_{suffix}"

# result table
rows = []
for name, index, suffix in MODEL_INFOS:
    row = {'Index': index}
    model_key = make_model_key(name, index, suffix)
    for bench, info in BENCHMARKS.items():
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
        for metric in info['metrics']:
            display_metric = METRIC_NAME_MAP.get(metric, metric)
            value = -1
            if model_key in data and metric in data[model_key]:
                value = data[model_key][metric]
            else:
                print(f"⚠️ Missing {model_key} {metric} in {info['path']}")
            row[f"{bench}_{display_metric}"] = value
    rows.append(row)

# =====================
# save as csv, column order consistent with header
# =====================
out_path = '/mnt/sharefs/users/haolong.jia/result/mixup_1_summary.csv'
df = pd.DataFrame(rows)
df = df[header] # force column order

# save as csv
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"🎉 Saved to {out_path}")