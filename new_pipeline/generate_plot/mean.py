import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 优先级从高到低
METRIC_PRIORITY = [
    'pass@16', 'acc,none', 'f1', 'mc2_acc', 'exact_match,remove_whitespace'
]

# benchmark: (result_path, [可用指标])
BENCHMARKS = [
    ('agieval', '/mnt/sharefs/users/haolong.jia/result/agieval/result.json'),
    ('arc_challenge', '/mnt/sharefs/users/haolong.jia/result/arc_challenge/result.json'),
    ('arc_easy', '/mnt/sharefs/users/haolong.jia/result/arc_easy/result.json'),
    ('bbh', '/mnt/sharefs/users/haolong.jia/result/bbh_pass16/passk.json'),
    ('commonsense_qa', '/mnt/sharefs/users/haolong.jia/result/commonsense_qa/result.json'),
    ('drop', '/mnt/sharefs/users/haolong.jia/result/drop/result.json'),
    ('gsm8k', '/mnt/sharefs/users/haolong.jia/result/gsm8k_pass16/passk.json'),
    ('gpqa', '/mnt/sharefs/users/haolong.jia/result/gpqa_pass32/passk.json'),
    ('hellaswag', '/mnt/sharefs/users/haolong.jia/result/hellaswag/result.json'),
    ('math500', '/mnt/sharefs/users/haolong.jia/result/math500_pass64/passk.json'),
    ('mmlu_pro', '/mnt/sharefs/users/haolong.jia/result/mmlu_pro_pass16/passk.json'),
    ('nq_open', '/mnt/sharefs/users/haolong.jia/result/nq_open/result.json'),
    ('openbookqa', '/mnt/sharefs/users/haolong.jia/result/openbookqa/result.json'),
    ('piqa', '/mnt/sharefs/users/haolong.jia/result/piqa/result.json'),
    ('social_iqa', '/mnt/sharefs/users/haolong.jia/result/social_iqa/result.json'),
    ('triviaqa', '/mnt/sharefs/users/haolong.jia/result/triviaqa/result.json'),
    ('truthfulqa', '/mnt/sharefs/users/haolong.jia/result/truthfulqa/result.json'),
    ('winogrande', '/mnt/sharefs/users/haolong.jia/result/winogrande/result.json'),
]

# Qwen3-1.7B 的 model_name 关键字
QWEN3_KEY = 'qwen3-1.7b'

# 只允许的指标
ALLOWED_METRICS = set([
    'pass@16', 'acc,none', 'f1', 'mc2_acc', 'exact_match,remove_whitespace'
])

# 解析 checkpoint/step
def extract_step(model_name):
    # 支持 group-step 格式，如 t70-m30-7343
    # 提取最后一个-后面的数字
    parts = model_name.split('-')
    if parts:
        last_part = parts[-1]
        if last_part.isdigit():
            return int(last_part)
    return None

# 检查是否需要忽略的模型
def should_ignore_model(model_name):
    # 忽略特定的模型
    ignore_patterns = ['haibt']
    for pattern in ignore_patterns:
        if pattern in model_name.lower():
            return True
    return False

def select_metric(metrics):
    """从 metrics 列表中选出优先级最高的指标"""
    for m in METRIC_PRIORITY:
        if m in metrics:
            return m
    return None

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_metric_scores(data, metric):
    scores = []  # (step, model_name, value)
    qwen3_score = None
    for model_name, v in data.items():
        if not isinstance(v, dict) or metric not in v:
            continue
        step = extract_step(model_name)
        value = v[metric]
        scores.append((step, model_name, value))
        if QWEN3_KEY in model_name.lower():
            qwen3_score = value
    return scores, qwen3_score

def filter_metric(scores):
    # scores: (step, model_name, value)
    if not scores:
        return False, 'no scores', None, None
    # 只考虑有step的
    values = [v for s, m, v in scores]
    if not values:
        return False, 'no values', None, None
    best_raw_perf = max(values)
    if best_raw_perf < 0.1:
        return False, f'best raw performance {best_raw_perf:.3f} < 0.1', None, best_raw_perf
    # 按 step 分组
    step_to_vals = defaultdict(list)
    for s, m, v in scores:
        step_to_vals[s].append(v)
    mean_abs_diffs = []
    for step, vals in step_to_vals.items():
        mean = np.mean(vals)
        mean_abs_diff = np.mean(np.abs(np.array(vals) - mean))
        mean_abs_diffs.append(mean_abs_diff)
    if not mean_abs_diffs:
        return False, 'no mean abs diff', None, best_raw_perf
    max_mad = max(mean_abs_diffs)
    if max_mad < 0.02:
        return False, f'max mean abs diff {max_mad:.3f} < 0.02', max_mad, best_raw_perf
    return True, '', max_mad, best_raw_perf

def main():
    processed = set()
    filtered = set()
    out_dir = '/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/plot'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mean_abs_diff_by_step.pdf')
    plt.figure(figsize=(14, 8))
    
    for bench_name, path in BENCHMARKS:
        if not os.path.exists(path):
            print(f"[MISS] {bench_name}: File not found: {path}")
            continue
        processed.add(bench_name)
        data = load_json(path)
        
        # 收集所有可用的指标
        all_metrics = set()
        for v in data.values():
            if isinstance(v, dict):
                all_metrics.update(m.strip() for m in v.keys())
        
        # 从允许的指标中选择第一个可用的
        metric = None
        for m in METRIC_PRIORITY:
            if m in all_metrics:
                metric = m
                break
        
        if not metric:
            print(f"[FILTERED] {bench_name}: No suitable metric found. Available: {sorted(list(all_metrics))}")
            filtered.add(bench_name)
            continue
        
        print(f"[INFO] {bench_name}: Using metric '{metric}'")
        
        # 阈值过滤逻辑 - 收集所有分数
        scores = []
        for model_name, v in data.items():
            if not isinstance(v, dict) or metric not in v:
                continue
            if QWEN3_KEY in model_name.lower():
                continue
            if should_ignore_model(model_name):
                continue
            step = extract_step(model_name)
            if step is None:
                continue
            value = v[metric]
            scores.append(value)
        
        if not scores:
            print(f"[FILTERED] {bench_name}: No scores found for metric '{metric}'")
            filtered.add(bench_name)
            continue
        
        # 检查 best_raw_perf
        best_raw_perf = max(scores)
        if best_raw_perf < 0.1:
            print(f"[FILTERED] {bench_name}: best_raw_perf={best_raw_perf:.4f} < 0.1 for metric '{metric}'")
            filtered.add(bench_name)
            continue
        
        # 计算原始 mean_abs_diff
        step_to_scores = defaultdict(list)
        for model_name, v in data.items():
            if not isinstance(v, dict) or metric not in v:
                continue
            if QWEN3_KEY in model_name.lower():
                continue
            if should_ignore_model(model_name):
                continue
            step = extract_step(model_name)
            if step is None:
                continue
            value = v[metric]
            step_to_scores[step].append(value)
        
        mean_abs_diffs = []
        for step, vals in step_to_scores.items():
            if len(vals) >= 2:
                mean = np.mean(vals)
                mean_abs_diff = np.mean(np.abs(np.array(vals) - mean))
                mean_abs_diffs.append(mean_abs_diff)
        
        if mean_abs_diffs and max(mean_abs_diffs) < 0.02:
            print(f"[FILTERED] {bench_name}: max_mean_abs_diff={max(mean_abs_diffs):.4f} < 0.02 for metric '{metric}'")
            filtered.add(bench_name)
            continue
        
        # 找Qwen3分数进行归一化
        qwen3_score = None
        for model_name, v in data.items():
            if QWEN3_KEY in model_name.lower() and metric in v:
                qwen3_score = v[metric]
                break
        
        if not qwen3_score:
            qwen3_present = any(QWEN3_KEY in k.lower() for k in data.keys())
            print(f"[FILTERED] {bench_name}: No Qwen3 score for {metric}. Qwen3 present: {qwen3_present}")
            filtered.add(bench_name)
            continue
        
        # 聚合同step下所有group的分数（归一化）
        step_to_norm_scores = defaultdict(list)
        for model_name, v in data.items():
            if not isinstance(v, dict) or metric not in v:
                continue
            if QWEN3_KEY in model_name.lower():
                continue
            if should_ignore_model(model_name):
                continue
            step = extract_step(model_name)
            if step is None:
                continue
            value = v[metric]
            norm_value = value / qwen3_score if qwen3_score else 0
            step_to_norm_scores[step].append(norm_value)
        
        # 计算归一化后的mean_abs_diff
        steps = []
        mad_scores = []
        for step in sorted(step_to_norm_scores.keys()):
            group_scores = step_to_norm_scores[step]
            if len(group_scores) < 2:
                continue
            mean = np.mean(group_scores)
            mad = np.mean(np.abs(np.array(group_scores) - mean))
            steps.append(step)
            mad_scores.append(mad)
        
        if not steps:
            print(f"[FILTERED] {bench_name}: No step has >=2 groups after normalization")
            filtered.add(bench_name)
            continue
        
        # 画图
        plt.plot(steps, mad_scores, marker='o', label=bench_name.upper(), linewidth=2)
    
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Mean Absolute Difference from Mean', fontsize=12)
    plt.title('Mean Absolute Difference Across Checkpoints for Each Benchmark\n(Normalized by Qwen3, Averaged Across Splits)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f'\nSaved plot to {out_path}')
    
    # 统计信息
    missed = set(x[0] for x in BENCHMARKS) - processed
    if missed:
        print(f"\n[MISS] The following benchmarks were not processed (file not found): {sorted(missed)}")
    if filtered:
        print(f"\n[FILTERED] The following benchmarks were filtered (see above for reasons): {sorted(filtered)}")
    print(f"\n[SUCCESS] Successfully plotted {len(processed) - len(filtered)} benchmarks.")

if __name__ == '__main__':
    main()





