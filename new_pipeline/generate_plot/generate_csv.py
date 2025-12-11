import json
import argparse
import re
import pandas as pd
import os
from collections import defaultdict

# Define model groups
# 7B Models
GROUP_7B = {
    # "all_fuchsia_ipaddress_": "mix-all",
    # "math_acrylic_beethoven_": "mix-math",
    # "regmix_holistic_plane_": "mix-regmix",
    # "baseline_congruent_cocoa_": "mix-baseline",
    # "nvidia_cynical_hydrogenfuel_": "mix-nvidia",
}

# 1.5B Models
GROUP_1P5B = {
    "social_candy_0": "test-mix",
    "t35-m30-g35": "t35-m30-g35",
    "t60-m30-r10": "t60-m30-r10",
    "t70-m30": "t70-m30",
    "t40-m30-o0-r4-p3-a3-g10-ma10": "t40-m30-o0-r4-p3-a3-g10-ma10",
    "t20-m25-o0-r7-p7-a7-g9-ma25": "t20-m25-o0-r7-p7-a7-g9-ma25",
    "t60-m30-o0-g0-r0-p0-ma10": "t60-m30-o0-g0-r0-p0-ma10",
    "t70-m10-o0-g0-r0-p0-ma20": "t70-m10-o0-g0-r0-p0-ma20",
    "t30-m15-o10-r5-p5-g20-ma15": "t30-m15-o10-r5-p5-g20-ma15",
    "t50-m30-o20-r0-p0-g0-ma0": "t50-m30-o20-r0-p0-g0-ma0",
    "lonely_cone_0": "final-mix",
}

# Combine for lookup
ALL_GROUPS = {**GROUP_7B, **GROUP_1P5B}

def get_model_size_and_name(raw_group):
    if raw_group in GROUP_7B:
        return '7B', GROUP_7B[raw_group]
    if raw_group in GROUP_1P5B:
        return '1.5B', GROUP_1P5B[raw_group]
    return None, None

def generate_csv_for_subset(data_subset, output_path, size_label):
    if not data_subset:
        return

    # Identify all steps
    all_steps = set()
    for model_data in data_subset.values():
        all_steps.update(model_data.keys())
    
    sorted_steps = sorted(list(all_steps))
    
    # Sort models (optional, but good for consistency)
    # Fixed order: mix-all, mix-math, mix-regmix
    desired_order = ['mix-all', 'mix-math', 'mix-regmix']
    
    ordered_models = []
    for name in desired_order:
        if name in data_subset:
            ordered_models.append(name)
            
    # Add any others found
    for name in data_subset.keys():
        if name not in ordered_models:
            ordered_models.append(name)

    rows = []
    for name in ordered_models:
        row = {'Model': name}
        for step in sorted_steps:
            row[step] = data_subset[name].get(step, None)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {size_label} CSV to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--passk', type=str, required=True, help='Path to passk.json')
    parser.add_argument('--metric', type=str, required=True, help='Metric name, e.g., pass@16')
    parser.add_argument('--task', type=str, default=None, help='Task name for filename prefix, e.g. gsm8k')
    
    # New arguments for separated output roots
    parser.add_argument('--output_root_7b', type=str, default=None, help='Root output directory for 7B results')
    parser.add_argument('--output_root_1p5b', type=str, default=None, help='Root output directory for 1.5B results')
    parser.add_argument('--subdir', type=str, default="", help='Subdirectory (e.g. base, math, code) to append to root')
    
    # Deprecated argument, but keeping it just in case, mapped to current logic if possible
    parser.add_argument('--output_dir', type=str, default=None, help='[Deprecated] Directory to save output CSVs')
    
    args = parser.parse_args()

    if not os.path.exists(args.passk):
        print(f"Error: File {args.passk} not found.")
        return

    with open(args.passk, 'r') as f:
        data = json.load(f)

    # Data structures
    data_7b = defaultdict(dict)
    data_1p5b = defaultdict(dict)

    for model_name, result in data.items():
        # Match model name to groups
        matched_group = None
        step = None

        for group_prefix in ALL_GROUPS.keys():
            if model_name.startswith(group_prefix):
                matched_group = group_prefix
                try:
                    # Extract the part after the prefix
                    suffix = model_name[len(group_prefix):]
                    # Remove leading separators like _ or - and parse the number
                    step = int(suffix.lstrip('_-'))
                except (ValueError, IndexError):
                    continue
                break
        
        if matched_group and args.metric in result:
            size, display_name = get_model_size_and_name(matched_group)
            score = result[args.metric]
            
            if size == '7B':
                if step == 74688:
                    # Skip this specific incorrect step for 7B models
                    continue
                data_7b[display_name][step] = score
            elif size == '1.5B':
                data_1p5b[display_name][step] = score

    # Determine output directories
    # Default fallback if roots are not provided: use passk dir or output_dir
    fallback_dir = args.output_dir if args.output_dir else os.path.dirname(args.passk)
    
    out_dir_7b = args.output_root_7b if args.output_root_7b else fallback_dir
    out_dir_1p5b = args.output_root_1p5b if args.output_root_1p5b else fallback_dir
    
    # Append subdir if provided
    if args.subdir:
        out_dir_7b = os.path.join(out_dir_7b, args.subdir)
        out_dir_1p5b = os.path.join(out_dir_1p5b, args.subdir)

    # Determine filename prefix
    if args.task:
        prefix = args.task
    else:
        prefix = os.path.basename(args.passk).replace('.json', '')
        
    # Generate 7B CSV
    if data_7b:
        # No need for suffix if directories are separated, but maybe safer to keep? 
        # Usually separate folders imply separate contexts. 
        # If the user wants exact filenames like "gsm8k_pass@1.csv", we shouldn't add suffix.
        # But to be safe let's keep suffix OR check if paths are different.
        # Let's stick to the suffix convention for clarity unless asked otherwise.
        # Actually, if they are in different folders, suffix is redundant but harmless.
        filename_7b = f"{prefix}_{args.metric}.csv"
        generate_csv_for_subset(data_7b, os.path.join(out_dir_7b, filename_7b), "7B")
        
    if not data_7b:
        print(f"No 7B data found for {args.metric}")

    # Generate 1.5B CSV
    if data_1p5b:
        filename_1p5b = f"{prefix}_{args.metric}.csv"
        generate_csv_for_subset(data_1p5b, os.path.join(out_dir_1p5b, filename_1p5b), "1.5B")

    if not data_1p5b:
         print(f"No 1.5B data found for {args.metric}")

if __name__ == "__main__":
    main()
