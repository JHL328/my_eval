"""
plot the performance of the models
support two types of models:
1. self-trained model: step interval 7343, 30B tokens per step
2. HuggingFace stage1 model: step interval 40000, 94B tokens per step

example:
python smol.py --passk results.json --output plot.pdf --metric "pass@16" --task "Benchmark"
"""
import json
import argparse
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# define the allowed model groups
ALLOWED_MODEL_GROUPS = [
    't60-m30-r10',
    't70-m30', 
    't35-m30-g35',
    't40-m30-o0-r4-p3-a3-g10-ma10',
    't20-m25-o0-r7-p7-a7-g9-ma25',
    't60-m30-o0-g0-r0-p0-ma10',
    't70-m10-o0-g0-r0-p0-ma20',
    't30-m15-o10-r5-p5-g20-ma15',
    't50-m30-o20-r0-p0-g0-ma0',
    'social_candy_0',
    'lonely_cone_0',
    'stage1',  # new add smolLM3 base model
]

def step_to_tokens(model_name, step):
    """
    convert step to token number (in B)
    - stage1 model: 40000 steps = 94B tokens
    - other models: 7343 steps = 10B tokens
    """
    if 'stage1' in model_name:
        # HuggingFace model: step 40000 = 94B
        tokens = (step / 40000) * 94
        # according to the need, can slightly adjust the display value
        # 40000 is displayed as 90B, 80000 is displayed as 180B
        if step == 40000:
            return 90
        elif step == 80000:
            return 180
        else:
            return round(tokens, 0)
    else:
        # self-trained model: step 7343 = 30B
        return (step / 7343) * 30

def get_canonical_color_key(raw_group):
    """get the canonical color key"""
    if raw_group == 'stage1':
        return 'stage1'
    if raw_group.startswith('social_candy_0'):
        return 'social_candy_0'
    if raw_group.startswith('lonely_cone_0'):
        return 'lonely_cone_0'
    
    # parse the parameters
    mapping = {
        't': 't',
        'm': 'mm',
        'o': 'code',
        'r': 'reasoning',
        'p': 'planning',
        'a': 'ai',
        'g': 'general',
        'ma': 'thinking'
    }
    
    parts = re.findall(r'([a-z]+)(\d+)', raw_group)
    items = []
    for k, v in parts:
        k = k.lower()
        if k in mapping:
            if k in ['t', 'm'] or int(v) != 0:
                items.append(f"{mapping[k]}{v}")
        else:
            items.append(f"{k}{v}")
    return '-'.join(items)

def group_to_label(group):
    """convert the group name to the display label"""
    if group == 'stage1':
        return 'HuggingFace Stage1'
    if group == 'social_candy_0':
        return 'test-mix'
    if group == 'lonely_cone_0':
        return 'final-mix'
    
    mapping = {
        't': 't',
        'm': 'mm',
        'g': 'general',
        'o': 'coder',
        'r': 'reasoning',
        'p': 'planning',
        'a': 'ai',
        'ma': 'thinking'
    }
    
    parts = re.findall(r'([a-z]+)(\d+)', group)
    label = []
    for k, v in parts:
        k = k.lower()
        if k in mapping:
            label.append(f"{mapping[k]}:{v}")
        else:
            label.append(f"{k}:{v}")
    return ', '.join(label)

# color mapping
color_map = {
    't70-mm30': '#1f77b4',
    't50-mm30-code20': '#ff7f0e',
    't35-mm30-general35': '#2ca02c',
    't60-mm30-reasoning10': '#d62728',
    't40-mm30-reasoning4-planning3-ai3-general10-thinking10': '#9467bd',
    't20-mm25-reasoning7-planning7-ai7-general9-thinking25': '#8c564b',
    't60-mm30-thinking10': '#e377c2',
    't30-mm15-code10-reasoning5-planning5-general20-thinking15': '#7f7f7f',
    't70-mm10-thinking20': '#bcbd22',
    'social_candy_0': '#17becf',
    'lonely_cone_0': '#ffd700',
    'stage1': '#ff1493',  # assign a unique color to stage1 (deep pink)
    'Llama-3.2-3B': '#e41a1c',
    'Qwen3-1.7B-Base': '#377eb8',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--passk', type=str, required=True, help='Path to passk.json')
    parser.add_argument('--output', type=str, required=True, help='Output PDF path')
    parser.add_argument('--metric', type=str, required=True, help='Metric name in passk.json')
    parser.add_argument('--task', type=str, required=True, help='Benchmark name for plot title')
    args = parser.parse_args()

    with open(args.passk, 'r') as f:
        data = json.load(f)

    group_lines = defaultdict(list)  # group -> list of (tokens, metric)
    open_source_lines = {}           # model_name -> metric

    for model_name, result in data.items():
        if args.metric not in result:
            continue
            
        # process stage1 model
        if model_name.startswith('stage1-step-'):
            group = 'stage1'
            if group not in ALLOWED_MODEL_GROUPS:
                continue
            # extract the step number
            match = re.match(r'stage1-step-(\d+)', model_name)
            if match:
                step = int(match.group(1))
                tokens = step_to_tokens(model_name, step)
                group_lines[group].append((tokens, result[args.metric]))
            continue
        
        # process social_candy_0 model
        if model_name.startswith('social_candy_0_'):
            group = 'social_candy_0'
            if group not in ALLOWED_MODEL_GROUPS:
                continue
            step = int(model_name.split('_')[-1])
            tokens = step_to_tokens(model_name, step)
            group_lines[group].append((tokens, result[args.metric]))
            continue
        
        # process lonely_cone_0 model
        if model_name.startswith('lonely_cone_0_'):
            group = 'lonely_cone_0'
            if group not in ALLOWED_MODEL_GROUPS:
                continue
            step = int(model_name.split('_')[-1])
            tokens = step_to_tokens(model_name, step)
            group_lines[group].append((tokens, result[args.metric]))
            continue
        
        # process other self-trained models
        match = re.match(r'([a-z0-9_\-]+)-(\d+)$', model_name)
        if match:
            raw_group = match.group(1)
            if raw_group not in ALLOWED_MODEL_GROUPS:
                continue
            step = int(match.group(2))
            tokens = step_to_tokens(model_name, step)
            group = get_canonical_color_key(raw_group)
            group_lines[group].append((tokens, result[args.metric]))
        else:
            # open-source models (only keep specific ones)
            if model_name in ["Llama-3.2-3B", "Qwen3-1.7B-Base"]:
                open_source_lines[model_name] = result[args.metric]

    # plot
    plt.figure(figsize=(16, 7), facecolor='#f7f7f7')

    # plot the model curves
    for group, points in group_lines.items():
        if len(points) < 2:
            continue
        points = sorted(points)
        tokens, metrics = zip(*points)
        label = group_to_label(group)
        color = color_map.get(group, None)
        plt.plot(tokens, metrics, marker='o', label=label, color=color, linewidth=2, markersize=7)

    # plot the open-source model horizontal lines
    for model_name, metric in open_source_lines.items():
        color = color_map.get(model_name, None)
        plt.axhline(y=metric, linestyle='--', color=color, label=model_name, linewidth=2)

    # set the chart attributes
    plt.xlabel('Tokens (B)', fontsize=14)
    plt.ylabel(args.metric.replace('_', '@'), fontsize=14)
    plt.title(f'{args.task} Performance vs Training Tokens', fontsize=16, weight='bold')
    plt.legend(
        title='Models',
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        fontsize=11,
        title_fontsize=12,
        frameon=False,
        borderaxespad=0.
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.output, format='pdf', bbox_inches='tight', facecolor='#f7f7f7')
    print(f'Saved plot to {args.output}')

if __name__ == '__main__':
    main()