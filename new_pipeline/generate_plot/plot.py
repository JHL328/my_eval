"""
This script generates a performance plot from a JSON results file.
It's designed to visualize the progress of different model configurations (or "groups") over varying numbers of samples/steps,
and compare them against fixed-performance open-source models.

Usage Example:
python plot.py --passk /path/to/your/results/passk.json --output /path/to/save/plot.pdf --metric "pass@16" --task "YourBenchmarkName"

Arguments:
    --passk (str): Path to the JSON file containing experiment results.
                   The JSON structure is typically:
                   {
                       "model_name_or_group_step": {
                           "metric_name_1": value_1,
                           "metric_name_2": value_2,
                           ...
                       },
                       ...
                   }
    --output (str): Path where the generated plot (PDF format) will be saved.
    --metric (str): The specific metric name (e.g., "pass@1", "pass@16") to plot from the results.
    --task (str): The name of the benchmark or task (e.g., "BBH", "MMLU") for which the plot is generated.
                  This will be used in the plot title.
"""
import json
import argparse
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Define which model groups to include in the plot
# Only models matching these patterns will be processed
ALLOWED_MODEL_GROUPS = [
    # 't60-m30-r10',
    # 't70-m30', 
    # 't35-m30-g35',
    # 't40-m30-o0-r4-p3-a3-g10-ma10',
    # 't20-m25-o0-r7-p7-a7-g9-ma25',
    # 't60-m30-o0-g0-r0-p0-ma10',
    # 't70-m10-o0-g0-r0-p0-ma20',
    # 't30-m15-o10-r5-p5-g20-ma15',
    # 't50-m30-o20-r0-p0-g0-ma0',
    # 'social_candy_0',
    # 'lonely_cone_0',
    # 'rolling_inverse_0',
    # 'mountainous_extension_0',
    # "weary_alias_0",
    # "enthusiastic_minimalism_0",
    # "courageous_congruence_0",
    # "all_third_sine_",
    # "baseline_crooked_rice_",
    # "math_lasting_mannerism_",
    # "nvidia_cheerful_gong_",
    # "regmix_evasive_gateway_",
    # "web_double_teenage_monochrome_",
    # "web_high_fashionable_multiplication_",
    ##########################
    ###### 7B base models ####
    ##########################
    # "all_fuchsia_ipaddress_",
    # "baseline_congruent_cocoa_",
    # "math_acrylic_beethoven_",
    # "nvidia_cynical_hydrogenfuel_",
    # "regmix_holistic_plane_",
    # Add more model groups here as needed
    ##########################
    ##### coding mixes #######
    ##########################
    # "code_dialogues_substantial_remoulade_",
    # "code_thinking_imperial_shannon_",
    # "math_grateful_refrain_",
]

def get_canonical_color_key(raw_group):
    # special case for social_candy_0
    if raw_group.startswith('social_candy_0'):
        return 'social_candy_0'
    # parse parameters
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
    # support multiple letter parameters (e.g., ma)
    parts = re.findall(r'([a-z]+)(\d+)', raw_group)
    # only keep non-zero parameters, t and mm are always kept
    items = []
    for k, v in parts:
        k = k.lower()
        if k in mapping:
            if k in ['t', 'm'] or int(v) != 0:
                items.append(f"{mapping[k]}{v}")
        else:
            items.append(f"{k}{v}")
    return '-'.join(items)

# group name to label, can add more groups here
def group_to_label(group):
    if group == 'social_candy_0':
        return 'test-mix'
    if group == 'lonely_cone_0':
        return 'final-mix'
    if group == 'rolling_inverse_0':
        return 'final-math-gpt'
    if group == 'mountainous_extension_0':
        return 'final-math-rewrite'
    if group == 'weary_alias_0':
        return 'data_v2'
    if group == 'enthusiastic_minimalism_0':
        return 'Web-High'
    if group == 'courageous_congruence_0':
        return 'Web-Medium'
    if group == 'all_third_sine_':
        return 'mix_all'
    if group == 'baseline_crooked_rice_':
        return 'mix_baseline'
    if group == 'math_lasting_mannerism_':
        return 'mix_math'
    if group == 'nvidia_cheerful_gong_':
        return 'mix_nvidia'
    if group == 'regmix_evasive_gateway_':
        return 'mix_regmix'
    if group == 'web_double_teenage_monochrome_':
        return 'mix_web_double'
    if group == 'web_high_fashionable_multiplication_':
        return 'mix_web_high'
    if group == 'all_fuchsia_ipaddress_':
        return 'mix_all_7B'
    if group == 'baseline_congruent_cocoa_':
        return 'mix_baseline_7B'
    if group == 'math_acrylic_beethoven_':
        return 'mix_math_7B'
    if group == 'nvidia_cynical_hydrogenfuel_':
        return 'mix_nvidia_7B'
    if group == 'regmix_holistic_plane_':
        return 'mix_regmix_7B'
    if group == 'code_dialogues_substantial_remoulade_':
        return 'mix_code_dialogues'
    if group == 'code_thinking_imperial_shannon_':
        return 'mix_code_thinking'
    if group == 'math_grateful_refrain_':
        return 'mix_math_WSD'
    
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

# can add more colors here
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
    'test-mix': '#17becf',
    'lonely_cone_0': '#ffd700',  # Gold
    'final-mix': '#ffd700',      # Gold
    'Llama-3.2-3B': '#e41a1c',
    'Qwen3-1.7B-Base': '#377eb8',
    'rolling_inverse_0': '#9b59b6',  # Purple
    'final-math-gpt': '#9b59b6',     # Purple
    'mountainous_extension_0': '#e67e22',  # Orange
    'final-math-rewrite': '#e67e22',       # Orange
    'weary_alias_0': '#45b1e8',  # Blue
    'data_v2': '#45b1e8',       # Blue
    'Web-High': '#1f77b4',
    'Web-Medium': '#1fff77', 
    'mix_all': '#1f77b4',
    'mix_baseline': '#ff7f0e',
    'mix_math': '#2ca02c',
    'mix_nvidia': '#d62728',
    'mix_regmix': '#9467bd',
    'mix_web_double': '#e377c2',
    'mix_web_high': '#7f7f7f',
    'mix_all_7B': '#1f77b4',
    'mix_baseline_7B': '#ff7f0e',
    'mix_math_7B': '#2ca02c',
    'mix_nvidia_7B': '#d62728',
    'mix_regmix_7B': '#9467bd',
    'mix_code_dialogues': '#b2df8a',
    'mix_code_thinking': '#cab2d6',
    'mix_math_WSD': '#fb9a99',
}

parser = argparse.ArgumentParser()
parser.add_argument('--passk', type=str, required=True, help='Path to passk.json')
parser.add_argument('--output', type=str, required=True, help='Output PDF path')
parser.add_argument('--metric', type=str, required=True, help='Metric name in passk.json, e.g., pass@16')
parser.add_argument('--task', type=str, required=True, help='Benchmark or task name for the plot title, e.g., BBH, MMLU')
args = parser.parse_args()

with open(args.passk, 'r') as f:
    data = json.load(f)

group_lines = defaultdict(list)  # group -> list of (step, metric)
open_source_lines = {}           # model_name -> metric

for model_name, result in data.items():
    # special case for social_candy_0_XXXXX
    if model_name.startswith('social_candy_0_'):
        group = 'social_candy_0'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for lonely_cone_0_XXXXX
    if model_name.startswith('lonely_cone_0_'):
        group = 'lonely_cone_0'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for rolling_inverse_0_XXXXX
    if model_name.startswith('rolling_inverse_0_'):
        group = 'rolling_inverse_0'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for mountainous_extension_0_XXXXX
    if model_name.startswith('mountainous_extension_0_'):
        group = 'mountainous_extension_0'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for weary_alias_0_XXXXX
    if model_name.startswith('weary_alias_0_'):
        group = 'weary_alias_0'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for enthusiastic_minimalism_0_XXXXX
    if model_name.startswith('enthusiastic_minimalism_0_'):
        group = 'enthusiastic_minimalism_0'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for courageous_congruence_0_XXXXX
    if model_name.startswith('courageous_congruence_0_'):
        group = 'courageous_congruence_0'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue

    # special case for all_third_sine_XXXXX
    if model_name.startswith('all_third_sine_'):
        group = 'all_third_sine_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for baseline_crooked_rice_XXXXX
    if model_name.startswith('baseline_crooked_rice_'):
        group = 'baseline_crooked_rice_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for math_lasting_mannerism_XXXXX
    if model_name.startswith('math_lasting_mannerism_'):
        group = 'math_lasting_mannerism_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for nvidia_cheerful_gong_XXXXX
    if model_name.startswith('nvidia_cheerful_gong_'):
        group = 'nvidia_cheerful_gong_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for regmix_evasive_gateway_XXXXX
    if model_name.startswith('regmix_evasive_gateway_'):
        group = 'regmix_evasive_gateway_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue

    # special case for web_double_teenage_monochrome_XXXXX
    if model_name.startswith('web_double_teenage_monochrome_'):
        group = 'web_double_teenage_monochrome_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for web_high_fashionable_multiplication_XXXXX
    if model_name.startswith('web_high_fashionable_multiplication_'):
        group = 'web_high_fashionable_multiplication_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for all_fuchsia_ipaddress_XXXXX
    if model_name.startswith('all_fuchsia_ipaddress_'):
        group = 'all_fuchsia_ipaddress_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for baseline_congruent_cocoa_XXXXX
    if model_name.startswith('baseline_congruent_cocoa_'):
        group = 'baseline_congruent_cocoa_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for math_acrylic_beethoven_XXXXX
    if model_name.startswith('math_acrylic_beethoven_'):
        group = 'math_acrylic_beethoven_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for nvidia_cynical_hydrogenfuel_XXXXX
    if model_name.startswith('nvidia_cynical_hydrogenfuel_'):
        group = 'nvidia_cynical_hydrogenfuel_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for regmix_holistic_plane_XXXXX
    if model_name.startswith('regmix_holistic_plane_'):
        group = 'regmix_holistic_plane_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue

    # special case for code_dialogues_substantial_remoulade_XXXXX
    if model_name.startswith('code_dialogues_substantial_remoulade_'):
        group = 'code_dialogues_substantial_remoulade_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for code_thinking_imperial_shannon_XXXXX
    if model_name.startswith('code_thinking_imperial_shannon_'):
        group = 'code_thinking_imperial_shannon_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    
    # special case for math_grateful_refrain_XXXXX
    if model_name.startswith('math_grateful_refrain_'):
        group = 'math_grateful_refrain_'
        # Check if this group is allowed
        if group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(model_name.split('_')[-1])
        if args.metric in result:
            group_lines[group].append((step, result[args.metric]))
        continue
    # other groups keep the original -digit ending regex logic
    match = re.match(r'([a-z0-9_\-]+)-(\d+)$', model_name)
    if match and args.metric in result:
        raw_group = match.group(1)
        # Check if this group is allowed
        if raw_group not in ALLOWED_MODEL_GROUPS:
            continue
        step = int(match.group(2))
        group = get_canonical_color_key(raw_group)
        group_lines[group].append((step, result[args.metric]))
    elif args.metric in result:
        # only keep Llama-3.2-3B and Qwen3-1.7B-Base for open source models
        if model_name not in ["Llama-3.2-3B", "Qwen3-1.7B-Base"]:
            continue
        open_source_lines[model_name] = result[args.metric]

plt.figure(figsize=(16, 7), facecolor='#f7f7f7')

# plot group curves
for group, points in group_lines.items():
    if len(points) < 2:
        continue
    points = sorted(points)
    steps, metrics = zip(*points)
    label = group_to_label(group)
    color = color_map.get(group)
    if color is None:
        # Fall back to the human readable label so groups with aliases reuse the intended palette.
        color = color_map.get(label)
    plt.plot(steps, metrics, marker='o', label=label, color=color, linewidth=2, markersize=7)

# assign fixed colors to open source models
for model_name, metric in open_source_lines.items():
    color = color_map.get(model_name, None)
    plt.axhline(y=metric, linestyle='--', color=color, label=model_name, linewidth=2)

plt.xlabel('Number of Samples', fontsize=14)
plt.ylabel(args.metric.replace('_', '@'), fontsize=14)
plt.title(f'{args.task} Tokenmix Ablation Progress', fontsize=16, weight='bold')
plt.legend(
    title='Experiment',
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    fontsize=11,
    title_fontsize=12,
    frameon=False,
    borderaxespad=0.
)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig(args.output, format='pdf', bbox_inches='tight', facecolor='#f7f7f7')
print(f'Saved plot to {args.output}')
