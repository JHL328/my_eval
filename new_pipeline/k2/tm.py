import os
import json
import pandas as pd
import sys
import re

# Ensure parent directory is in sys.path for k2 imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from k2 import k2_evaluate # Import k2_evaluate to get task configs

# =====================
# Selected models - only these 6 models
# =====================
SELECTED_MODELS = {
   "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/k2+_midtraining_mix_final/final-a:10-bp:5-g:5-m:30-mm:0-oc:15-p:10-qa:5-r1:10-r:10/hf_format/samples_3773124.0":"final-a-10-bp-5-g-5-m-30-mm-0-oc-15-p-10-qa-5-r1-10-r-10-3773124",
   "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/k2+_midtraining_mix_final/final-a:10-bp:5-g:5-m:30-mm:0-oc:15-p:10-qa:5-r1:10-r:10/hf_format/samples_7546248.0":"final-a-10-bp-5-g-5-m-30-mm-0-oc-15-p-10-qa-5-r1-10-r-10-7546248",
   "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/k2+_midtraining_mix_final/final-a:7-bp:5-g:5-m:30-mm:2-oc:10-p:7-qa:5-r1:20-r:7/hf_format/samples_3674102.0":"final-a-7-bp-5-g-5-m-30-mm-2-oc-10-p-7-qa-5-r1-20-r-7-3674102",
   "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/k2+_midtraining_mix_final/final-a:7-bp:5-g:5-m:30-mm:2-oc:10-p:7-qa:5-r1:20-r:7/hf_format/samples_7348204.0":"final-a-7-bp-5-g-5-m-30-mm-2-oc-10-p-7-qa-5-r1-20-r-7-7348204",
   "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/k2+_midtraining_mix_final/final-a:20-bp:17-g:5-m:17-mm:2-oc:10-p:7-qa:5-r1:7-r:7/hf_format/samples_3667672.0":"final-a-20-bp-17-g-5-m-17-mm-2-oc-10-p-7-qa-5-r1-7-r-7-3667672",
   "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/k2+_midtraining_mix_final/final-a:20-bp:17-g:5-m:17-mm:2-oc:10-p:7-qa:5-r1:7-r:7/hf_format/samples_7335344.0":"final-a-20-bp-17-g-5-m-17-mm-2-oc-10-p-7-qa-5-r1-7-r-7-7335344",
}

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
}

# Benchmarks including the new LiveCodeBench
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

# LiveCodeBench configuration
LIVECODEBENCH_DIR = '/mnt/sharefs/users/haolong.jia/result-k2/livecode_test/'

# =====================
# Helper functions
# =====================
def parse_model_groups():
    """Parse models into groups and determine first/second based on number"""
    groups = {}
    
    for model_name in SELECTED_MODELS.values():
        # Extract the base name without the final number
        match = re.match(r'^(.*)-(\d+)$', model_name)
        if match:
            base_name = match.group(1)
            number = int(match.group(2))
            
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append((model_name, number))
    
    # Sort each group by number to determine first/second
    model_suffixes = {}
    for base_name, models in groups.items():
        sorted_models = sorted(models, key=lambda x: x[1])
        for i, (model_name, _) in enumerate(sorted_models):
            if i == 0:
                model_suffixes[model_name] = 'first'
            elif i == 1:
                model_suffixes[model_name] = 'second'
            else:
                model_suffixes[model_name] = f'{i+1}th'  # In case there are more than 2
    
    return model_suffixes

def format_model_name(model_name, suffix):
    """Format model name by replacing parameter-value separators with : while keeping group separators as -"""
    # Remove the final number
    match = re.match(r'^(.*)-(\d+)$', model_name)
    if match:
        base_name = match.group(1)
        
        # Split by '-' and process each part
        parts = base_name.split('-')
        formatted_parts = []
        
        # First part is 'final', keep it as is
        formatted_parts.append(parts[0])
        
        # Known parameter names
        param_names = {'a', 'bp', 'g', 'm', 'mm', 'oc', 'p', 'qa', 'r1', 'r'}
        
        # Process the rest of the parts
        i = 1
        while i < len(parts):
            if parts[i] in param_names and i + 1 < len(parts):
                # This is a parameter name followed by its value
                formatted_parts.append(f"{parts[i]}:{parts[i+1]}")
                i += 2
            else:
                # This shouldn't happen with well-formed names, but just in case
                formatted_parts.append(parts[i])
                i += 1
        
        # Join with '-' and add suffix
        formatted_name = '-'.join(formatted_parts)
        return f"{formatted_name}-{suffix}"
    
    return model_name

def read_livecodebench_results(model_name):
    """Read LiveCodeBench results for a specific model"""
    file_path = os.path.join(LIVECODEBENCH_DIR, f"{model_name}_codegen__livecodebench_279_metric.json")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded LiveCodeBench results for {model_name}")
        return data
    except Exception as e:
        print(f"⚠️ Failed to load LiveCodeBench results for {model_name}: {e}")
        return None

# =====================
# Main processing
# =====================
# Parse model groups to determine suffixes
model_suffixes = parse_model_groups()
print("📊 Model grouping:")
for model, suffix in model_suffixes.items():
    print(f"  {model} -> {suffix}")

# Generate header (no Index column)
header = ['Model Name']
for bench, info in K2_BENCHMARKS.items():
    for metric in info['metrics']:
        display_metric = METRIC_NAME_MAP.get(metric, metric)
        header.append(f"{bench}_{display_metric}")

# Add LiveCodeBench columns
header.extend(['livecodebench_pass@4', 'livecodebench_pass@16'])

# Process each model
rows = []
for model_path, model_name in SELECTED_MODELS.items():
    print(f"\n🔍 Processing: {model_name}")
    
    # Format the model name for CSV display only
    # Note: We still use the original model_name (with -) for looking up results
    suffix = model_suffixes.get(model_name, 'unknown')
    formatted_model_name = format_model_name(model_name, suffix)
    print(f"  📝 Formatted name for CSV: {formatted_model_name}")
    
    row = {'Model Name': formatted_model_name}
    
    # Process existing benchmarks - use original model_name (with -) to lookup in result.json
    for bench, info in K2_BENCHMARKS.items():
        try:
            with open(info['path'], 'r') as f:
                data = json.load(f)
            
            # Use the original model_name (with -) to find results
            if model_name in data:
                model_data = data[model_name]
                for metric in info['metrics']:
                    display_metric = METRIC_NAME_MAP.get(metric, metric)
                    value = model_data.get(metric, -1)
                    row[f"{bench}_{display_metric}"] = value
                    if value != -1:
                        print(f"  ✓ {bench}_{display_metric}: {value}")
            else:
                print(f"  ⚠️ Model {model_name} not found in {bench}")
                for metric in info['metrics']:
                    display_metric = METRIC_NAME_MAP.get(metric, metric)
                    row[f"{bench}_{display_metric}"] = -1
                    
        except Exception as e:
            print(f"  ❌ Failed to load {bench}: {e}")
            for metric in info['metrics']:
                display_metric = METRIC_NAME_MAP.get(metric, metric)
                row[f"{bench}_{display_metric}"] = -1
    
    # Process LiveCodeBench results - also use original model_name (with -)
    livecodebench_data = read_livecodebench_results(model_name)
    if livecodebench_data:
        row['livecodebench_pass@4'] = livecodebench_data.get('pass@4', -1)
        row['livecodebench_pass@16'] = livecodebench_data.get('pass@16', -1)
        print(f"  ✓ livecodebench_pass@4: {row['livecodebench_pass@4']}")
        print(f"  ✓ livecodebench_pass@16: {row['livecodebench_pass@16']}")
    else:
        row['livecodebench_pass@4'] = -1
        row['livecodebench_pass@16'] = -1
    
    rows.append(row)

# Create DataFrame and save
out_path = '/mnt/sharefs/users/haolong.jia/result-k2/k2_final_format.csv'
df = pd.DataFrame(rows)
df = df[header]  # Force column order

# Save as CSV
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\n🎉 Successfully saved to {out_path}")
print(f"📈 Total rows: {len(rows)}")
print(f"📊 Total columns: {len(header)}")
print(f"\n🔢 Preview:")
print(df)