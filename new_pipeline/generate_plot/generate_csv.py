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
    # pt-mask-ablation 1p5B base ckpts (bbq_ablations): name_<step> -> display row
    "mix-bbq-all-mask": "mix-bbq-all-mask",
    "mix-bbq-all-baseline": "mix-bbq-all-baseline",
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

    # MERGE into existing CSV instead of overwriting, so rows from other mixes
    # (e.g. mix-all, mix-bbq-all, ...) already in the file are preserved.
    # master: model_name -> {step(int): value}; existing_order preserves row order.
    master = {}
    existing_order = []
    if os.path.exists(output_path):
        try:
            # Read as strings so existing values are preserved verbatim (no float re-formatting),
            # keeping the git diff to genuine additions only.
            old = pd.read_csv(output_path, dtype=str, keep_default_na=True)
            step_cols = [c for c in old.columns if c != 'Model']
            for _, r in old.iterrows():
                m = r['Model']
                existing_order.append(m)
                master[m] = {}
                for c in step_cols:
                    try:
                        ci = int(c)
                    except (ValueError, TypeError):
                        continue
                    v = r[c]
                    if pd.notna(v) and str(v).strip() != '':
                        master[m][ci] = v
        except Exception as e:
            print(f"⚠️ Could not read existing {output_path}: {e}; writing fresh.")
            master, existing_order = {}, []

    # Overlay new data (this run's models): add new rows / update specific (model, step) cells.
    for name, step_scores in data_subset.items():
        if name not in master:
            master[name] = {}
            existing_order.append(name)
        for step, score in step_scores.items():
            master[name][int(step)] = score

    # Union of all steps across all rows, sorted numerically -> column order.
    all_steps = sorted({s for d in master.values() for s in d.keys()})

    rows = []
    for name in existing_order:
        row = {'Model': name}
        for step in all_steps:
            row[step] = master[name].get(step, None)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {size_label} CSV to {output_path} ({len(rows)} rows, {len(all_steps)} steps)")

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

    # Optional allowlist of display names to emit (comma-separated). Default: all matched models.
    # Used to restrict output to specific mixes (e.g. only the bbq-ablation rows) so cumulative
    # passk.json files don't pull in unrelated models from other studies.
    parser.add_argument('--only', type=str, default=None, help='Comma-separated display names to include')

    args = parser.parse_args()
    only_set = set(s.strip() for s in args.only.split(',')) if args.only else None

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
            if only_set is not None and display_name not in only_set:
                continue
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
