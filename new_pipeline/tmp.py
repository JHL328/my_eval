import os
import json
import pandas as pd

model_map = {
    "/mnt/sharefs/users/haolong.jia/checkpoint_587bins_final4/tokenmix_ablation_usable_model_0/usable_model_0_71525": "usable_model_0_71525",
    "/mnt/sharefs/users/haolong.jia/checkpoint_587bins_final4/tokenmix_ablation_solitary_instruction_1/solitary_instruction_1_71525": "solitary_instruction_1_71525",
    "/mnt/sharefs/users/haolong.jia/checkpoint_587bins_final4/tokenmix_ablation_adorable_axis_2/adorable_axis_2_71525": "adorable_axis_2_71525",
    "/mnt/sharefs/users/haolong.jia/checkpoint_587bins_final4/tokenmix_ablation_cooperative_matrix_3/cooperative_matrix_3_71525": "cooperative_matrix_3_71525",
}

# =====================
# Generate CSV code
# =====================

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

# Parse model names to extract index
MODEL_INFOS = []
for model_name in model_map.values():
    # model name format: name_index_71525
    if model_name.endswith('_71525'):
        base = model_name[:-len('_71525')]
        parts = base.rsplit('_', 1)
        if len(parts) == 2:
            name = parts[0]
            try:
                index = int(parts[1])
                MODEL_INFOS.append((name, index, model_name))
                print(f"✅ Parsed: {model_name} -> name: {name}, index: {index}")
            except ValueError:
                print(f"⚠️ Warning: index parse failed for {model_name}")
        else:
            print(f"⚠️ Warning: model name {model_name} does not match 'name_index_71525' pattern.")

# Sort by index
MODEL_INFOS.sort(key=lambda x: x[1])
print(f"\n📊 Total models to process: {len(MODEL_INFOS)}")
if MODEL_INFOS:
    print(f"📈 Index range: {MODEL_INFOS[0][1]} to {MODEL_INFOS[-1][1]}")

# Generate header order
header = ['Index']
for bench, info in BENCHMARKS.items():
    for metric in info['metrics']:
        display_metric = METRIC_NAME_MAP.get(metric, metric)
        header.append(f"{bench}_{display_metric}")

# Main table generation
rows = []
for i, (name, index, full_model_name) in enumerate(MODEL_INFOS):
    print(f"\n🔍 Processing model {i+1}/{len(MODEL_INFOS)}: {full_model_name} (index: {index})")
    row = {'Index': index}
    
    for bench, info in BENCHMARKS.items():
        # Read json - READ ONLY, no modifications
        try:
            with open(info['path'], 'r') as f:
                data = json.load(f)
            print(f"  ✅ Loaded {bench} from {info['path']}")
        except Exception as e:
            print(f"  ❌ Failed to load {bench} from {info['path']}: {e}")
            for metric in info['metrics']:
                display_metric = METRIC_NAME_MAP.get(metric, metric)
                row[f"{bench}_{display_metric}"] = -1
            continue
        
        # Extract metrics for this specific model only
        for metric in info['metrics']:
            display_metric = METRIC_NAME_MAP.get(metric, metric)
            value = -1
            
            # Only extract if this exact model exists in the data
            if full_model_name in data and metric in data[full_model_name]:
                value = data[full_model_name][metric]
                print(f"    ✓ Found {bench}_{display_metric}: {value}")
            else:
                print(f"    ⚠️ Missing {full_model_name} {metric} in {bench}")
                
            row[f"{bench}_{display_metric}"] = value
    
    rows.append(row)

# Save as CSV - change output filename to avoid overwriting
out_path = '/mnt/sharefs/users/haolong.jia/result/mixup_final4_summary.csv'
df = pd.DataFrame(rows)
df = df[header]  # Force column order

# Save
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\n🎉 Successfully saved to {out_path}")
print(f"📈 Total rows: {len(rows)}")
print(f"📊 Total columns: {len(header)}")

# Show which indices are included
indices = [info[1] for info in MODEL_INFOS]
print(f"\n📋 Included indices: {sorted(indices)}")

# Check for missing indices in the expected range
if indices:
    expected_range = range(min(indices), max(indices) + 1)
    missing_indices = set(expected_range) - set(indices)
    if missing_indices:
        print(f"⚠️  Missing indices: {sorted(missing_indices)}")

print(f"\n🔢 First few rows preview:")
print(df.head())

# Confirm that we're only reading, not modifying the source files
print(f"\n✅ Data extraction completed. Source result files remain unchanged.")
print(f"📁 Only these {len(model_map)} models were extracted:")
for model in model_map.values():
    print(f"   - {model}")