import json
import argparse
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# group name to label, can add more groups here
def group_to_label(group):
    # example: t35-m30-g35 -> txt:35, mm:30, general:35
    mapping = {
        't': 'txt',
        'm': 'mm',
        'g': 'general',
        'o': 'opencoder',
        'r': 'reasoning',
        'p': 'planning',
        'a': 'ai',
        'ma': 'math'
    }
    # note the regex here
    parts = re.findall(r'([a-z]+)(\d+)', group)
    label = []
    for k, v in parts:
        k = k.lower()
        if k in mapping:
            label.append(f"{mapping[k]}:{v}")
        else:
            label.append(f"{k}:{v}")
    return ', '.join(label)

parser = argparse.ArgumentParser()
parser.add_argument('--passk', type=str, required=True, help='Path to passk.json')
parser.add_argument('--output', type=str, required=True, help='Output PDF path')
parser.add_argument('--metric', type=str, required=True, help='Metric name in passk.json, e.g., pass@16')
args = parser.parse_args()

with open(args.passk, 'r') as f:
    data = json.load(f)

group_lines = defaultdict(list)  # group -> list of (step, metric)
open_source_lines = {}           # model_name -> metric

for model_name, result in data.items():
    # group-step 格式
    match = re.match(r'([a-z0-9\-]+)-(\d+)$', model_name)
    if match and args.metric in result:
        group = match.group(1)
        step = int(match.group(2))
        group_lines[group].append((step, result[args.metric]))
    elif args.metric in result:
        # 只保留 Llama-3.2-3B 和 Qwen3-1.7B-Base
        if model_name not in ["Llama-3.2-3B", "Qwen3-1.7B-Base"]:
            continue
        open_source_lines[model_name] = result[args.metric]

plt.figure(figsize=(16, 7))

# plot group curves
for group, points in group_lines.items():
    if len(points) < 2:
        # group with only one point is not plotted
        continue
    points = sorted(points)
    steps, metrics = zip(*points)
    label = group_to_label(group)
    plt.plot(steps, metrics, marker='o', label=label)

# assign different colors to open source models
open_source_colors = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'
]
for idx, (model_name, metric) in enumerate(open_source_lines.items()):
    color = open_source_colors[idx % len(open_source_colors)]
    plt.axhline(y=metric, linestyle='--', color=color, label=model_name)

plt.xlabel('Step')
plt.ylabel(args.metric)
plt.title(f'{args.metric} vs Step')
plt.legend(
    title='Experiment',
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    fontsize=9,
    borderaxespad=0.
)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(args.output, format='pdf', bbox_inches='tight')
print(f'Saved plot to {args.output}')
