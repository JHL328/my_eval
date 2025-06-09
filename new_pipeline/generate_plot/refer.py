import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from generate_plots import pass_at_k
from model_dirs import model_dirs

def get_checkpoint_number(path):
    """Extract checkpoint number from path."""
    last_part = path.rstrip('/').split('/')[-1]
    last_part = last_part.split('_')[-1].split('.')[0]
    if last_part.isdigit():
        return int(last_part)
    else:
        print(f"Error: {path} is not a valid path")
        return 0

def get_qwen3_baseline(task, k, split=None):
    """Get Qwen3 baseline performance for normalization."""
    model_path = 'Qwen/Qwen3-1.7B-Base'
    result_dir = f'./results/{task}'
    model_tag = '_'.join(model_path.split('/')[-3:])
    
    if split is not None:
        csv_path = os.path.join(result_dir, model_tag, f'{split}.csv')
        if not os.path.exists(csv_path):
            print(f"File {csv_path} does not exist for Qwen3 baseline")
            return None
        arr = np.loadtxt(csv_path, skiprows=1, delimiter=",")
    else:
        model_path = os.path.join(result_dir, model_tag)
        if not os.path.exists(model_path):
            print(f"Directory {model_path} does not exist for Qwen3 baseline")
            return None
        csv_paths = sorted(glob.glob(os.path.join(model_path, '*.csv')))
        if not csv_paths:
            print(f"No CSV files found in {model_path} for Qwen3 baseline")
            return None
        arrays = [np.loadtxt(p, skiprows=1, delimiter=",") for p in csv_paths]
        arr = np.concatenate(arrays, axis=0)
    
    return pass_at_k(arr, k)

def get_normalized_performance(task, k, split=None):
    """Get normalized performance (relative to Qwen3) for all checkpoints."""
    qwen3_score = get_qwen3_baseline(task, k, split)
    if qwen3_score is None or qwen3_score == 0:
        print(f'Skipping {task}: qwen score is 0 or other issues')
        return None
    
    result_dir = f'./results/{task}'
    checkpoint_scores = defaultdict(list)
    
    for model_path, short_name in model_dirs.items():
        if 'tokenmix_ablation' in model_path:
            model_tag = '_'.join(model_path.split('/')[-3:])
            checkpoint_num = get_checkpoint_number(model_path)
            
            if split is not None:
                csv_path = os.path.join(result_dir, model_tag, f'{split}.csv')
                if not os.path.exists(csv_path):
                    continue
                arr = np.loadtxt(csv_path, skiprows=1, delimiter=",")
            else:
                model_path = os.path.join(result_dir, model_tag)
                if not os.path.exists(model_path):
                    continue
                csv_paths = sorted(glob.glob(os.path.join(model_path, '*.csv')))
                if not csv_paths:
                    continue
                arrays = [np.loadtxt(p, skiprows=1, delimiter=",") for p in csv_paths]
                arr = np.concatenate(arrays, axis=0)
            
            score = pass_at_k(arr, k)
            normalized_score = score / qwen3_score
            checkpoint_scores[checkpoint_num].append(normalized_score)
    
    return checkpoint_scores

def get_raw_performance(task, k, split=None):
    """Get raw performance (not normalized) for all checkpoints."""
    result_dir = f'./results/{task}'
    checkpoint_scores = defaultdict(list)
    
    for model_path, short_name in model_dirs.items():
        if 'tokenmix_ablation' in model_path:
            model_tag = '_'.join(model_path.split('/')[-3:])
            checkpoint_num = get_checkpoint_number(model_path)
            
            if split is not None:
                csv_path = os.path.join(result_dir, model_tag, f'{split}.csv')
                if not os.path.exists(csv_path):
                    print(f'{csv_path} not found')
                    continue
                arr = np.loadtxt(csv_path, skiprows=1, delimiter=",")
            else:
                model_path = os.path.join(result_dir, model_tag)
                if not os.path.exists(model_path):
                    print(f'{model_path} not found')
                    continue
                csv_paths = sorted(glob.glob(os.path.join(model_path, '*.csv')))
                if not csv_paths:
                    print(f'{csv_paths} no files found')
                    continue
                arrays = [np.loadtxt(p, skiprows=1, delimiter=",") for p in csv_paths]
                arr = np.concatenate(arrays, axis=0)
            
            score = pass_at_k(arr, k)
            checkpoint_scores[checkpoint_num].append(score)
    
    return checkpoint_scores

def get_benchmark_metrics(scores):
    """Calculate mean absolute difference from mean and min/max ratio for a list of scores."""
    if not scores:
        print('No scores to calculate metrics')
        return None, None
    mean = np.mean(scores)
    mean_abs_diff = np.mean(np.abs(np.array(scores) - mean))
    max_val = max(scores)
    min_val = min(scores)
    min_max_ratio = min_val / max_val if max_val > 0 else 0
    return mean_abs_diff, min_max_ratio

def get_best_checkpoint_performance(checkpoint_scores):
    """Get the best performance across all checkpoints."""
    if not checkpoint_scores:
        print('No checkpoint scores to get best performance')
        return 0
    all_scores = []
    for scores in checkpoint_scores.values():
        all_scores.extend(scores)
    return max(all_scores) if all_scores else 0

def collect_benchmark_metrics(task, k, split=None):
    """Collect metrics for a single benchmark (or split of a benchmark).
    Returns (checkpoints, mean_abs_diffs, min_max_ratios) if successful, None otherwise."""
    # Get raw performance for filtering
    raw_scores = get_raw_performance(task, k, split)
    if not raw_scores:
        print(f'{task}-{split if split else ""}: no raw scores')
        return None
    
    # Check if best raw performance is at least 10%
    best_raw_perf = get_best_checkpoint_performance(raw_scores)
    if best_raw_perf < 0.1:
        print(f'Skipping {task}-{split if split else ""}: best raw performance {best_raw_perf:.3f} < 0.1')
        return None
    
    # Calculate raw mean absolute differences for filtering
    raw_mean_abs_diffs = []
    for cp in sorted(raw_scores.keys()):
        scores = raw_scores[cp]
        mean_abs_diff, _ = get_benchmark_metrics(scores)
        raw_mean_abs_diffs.append(mean_abs_diff)
    
    # Check if highest raw mean absolute difference is at least 0.02
    if max(raw_mean_abs_diffs) < 0.02:
        print(f'Skipping {task}-{split if split else ""}: max raw mean absolute difference {max(raw_mean_abs_diffs):.3f} < 0.02')
        return None
    
    # Get normalized performance for metrics
    checkpoint_scores = get_normalized_performance(task, k, split)
    if not checkpoint_scores:
        print(f'{task}-{split if split else ""}: no normalized scores')
        return None
    
    # Calculate metrics for each checkpoint
    checkpoints = sorted(checkpoint_scores.keys())
    mean_abs_diffs = []
    min_max_ratios = []
    for cp in checkpoints:
        scores = checkpoint_scores[cp]
        mean_abs_diff, min_max_ratio = get_benchmark_metrics(scores)
        mean_abs_diffs.append(mean_abs_diff)
        min_max_ratios.append(min_max_ratio)
    
    return checkpoints, mean_abs_diffs, min_max_ratios

def average_metrics_across_splits(split_metrics):
    """Average metrics across splits for each checkpoint.
    Returns (all_checkpoints, avg_mean_abs_diffs, avg_min_max_ratios)."""
    all_checkpoints = set()
    for checkpoints, _, _ in split_metrics.values():
        all_checkpoints.update(checkpoints)
    all_checkpoints = sorted(all_checkpoints)
    
    avg_mean_abs_diffs = []
    avg_min_max_ratios = []
    
    for cp in all_checkpoints:
        cp_diffs = []
        cp_ratios = []
        for checkpoints, diffs, ratios in split_metrics.values():
            if cp in checkpoints:
                idx = checkpoints.index(cp)
                cp_diffs.append(diffs[idx])
                cp_ratios.append(ratios[idx])
        if cp_diffs:  # Only include if we have data for this checkpoint
            avg_mean_abs_diffs.append(np.mean(cp_diffs))
            avg_min_max_ratios.append(np.mean(cp_ratios))
    
    return all_checkpoints, avg_mean_abs_diffs, avg_min_max_ratios

def plot_benchmark_metrics(benchmark_data, plot_type='individual'):
    """Plot metrics for benchmarks.
    plot_type can be 'individual' (one line per benchmark) or 'average' (single line for all benchmarks)."""
    if not benchmark_data:
        print("No benchmarks passed the filtering criteria!")
        return
    
    if plot_type == 'individual':
        # Plot mean absolute difference for all benchmarks
        plt.figure(figsize=(14, 8))
        for name, (checkpoints, mean_abs_diffs, _) in benchmark_data.items():
            plt.plot(checkpoints, mean_abs_diffs, marker='o', label=name.upper())
        plt.xlabel('Number of Samples')
        plt.ylabel('Mean Absolute Difference from Mean')
        plt.title('Mean Absolute Difference Across Checkpoints for Each Benchmark\n(Normalized by Qwen3, Averaged Across Splits)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/mean_abs_diff_all_benchmarks.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # Plot min/max ratio for all benchmarks
        plt.figure(figsize=(14, 8))
        for name, (checkpoints, _, min_max_ratios) in benchmark_data.items():
            plt.plot(checkpoints, min_max_ratios, marker='o', label=name.upper())
        plt.xlabel('Number of Samples')
        plt.ylabel('Min/Max Ratio')
        plt.title('Min/Max Ratio Across Checkpoints for Each Benchmark\n(Normalized by Qwen3, Averaged Across Splits)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig('plots/minmax_ratio_all_benchmarks.pdf', format='pdf', bbox_inches='tight')
        plt.close()
    
    else:  # plot_type == 'average'
        # Average across all benchmarks
        all_checkpoints = set()
        for checkpoints, _, _ in benchmark_data.values():
            all_checkpoints.update(checkpoints)
        all_checkpoints = sorted(all_checkpoints)
        
        # For each checkpoint, collect metrics from all benchmarks
        checkpoint_to_diffs = {cp: [] for cp in all_checkpoints}
        checkpoint_to_ratios = {cp: [] for cp in all_checkpoints}
        
        for checkpoints, mean_abs_diffs, min_max_ratios in benchmark_data.values():
            for cp, diff, ratio in zip(checkpoints, mean_abs_diffs, min_max_ratios):
                checkpoint_to_diffs[cp].append(diff)
                checkpoint_to_ratios[cp].append(ratio)
        
        # Calculate final averages
        avg_diffs = [np.mean(checkpoint_to_diffs[cp]) if checkpoint_to_diffs[cp] else np.nan for cp in all_checkpoints]
        avg_ratios = [np.mean(checkpoint_to_ratios[cp]) if checkpoint_to_ratios[cp] else np.nan for cp in all_checkpoints]
        
        print(f"\nFinal averages computed across {len(benchmark_data)} benchmarks")
        
        # Plot average mean absolute difference
        plt.figure(figsize=(10, 6))
        plt.plot(all_checkpoints, avg_diffs, 'b-o', linewidth=2)
        plt.xlabel('Number of Samples')
        plt.ylabel('Average Mean Absolute Difference from Mean')
        plt.title('Average Mean Absolute Difference Across All Benchmarks\n(Normalized by Qwen3)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/mean_abs_diff_average_all_benchmarks.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        
        # Plot average min/max ratio
        plt.figure(figsize=(10, 6))
        plt.plot(all_checkpoints, avg_ratios, 'r-o', linewidth=2)
        plt.xlabel('Number of Samples')
        plt.ylabel('Average Min/Max Ratio')
        plt.title('Average Min/Max Ratio Across All Benchmarks\n(Normalized by Qwen3)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('plots/minmax_ratio_average_all_benchmarks.pdf', format='pdf', bbox_inches='tight')
        plt.close()

def plot_benchmark_metrics_with_averages():
    """Plot mean absolute difference and min/max ratio across checkpoints for each benchmark,
    as well as averages across all benchmarks. For benchmarks with splits (kk, order),
    metrics are averaged across splits first."""
    tasks = ['kk', 'cd', 'sum', 'order']
    k = 64
    benchmark_data = {}
    
    # Collect all benchmarks (averaging across splits where applicable)
    for task in tasks:
        if task == 'kk':
            splits = ['2ppl', '3ppl', '4ppl', '5ppl', '6ppl', '7ppl', '8ppl']
            # Collect metrics for each split
            split_metrics = {}
            for split in splits:
                metrics = collect_benchmark_metrics(task, k, split)
                if metrics:
                    split_metrics[split] = metrics
            
            if split_metrics:
                # Average metrics across splits
                benchmark_data[task] = average_metrics_across_splits(split_metrics)
                print(f'Added kk with {len(split_metrics)} splits')
            
        elif task == 'order':
            splits = [6, 9, 12, 15, 18, 24, 30]
            # Collect metrics for each split
            split_metrics = {}
            for split in splits:
                metrics = collect_benchmark_metrics(task, k, str(split))
                if metrics:
                    split_metrics[split] = metrics
            
            if split_metrics:
                # Average metrics across splits
                benchmark_data[task] = average_metrics_across_splits(split_metrics)
                print(f'Added order with {len(split_metrics)} splits')
            
        else:  # cd and sum
            metrics = collect_benchmark_metrics(task, k)
            if metrics:
                benchmark_data[task] = metrics
                print(f'Added {task}')
    
    # Generate both individual and average plots
    plot_benchmark_metrics(benchmark_data, plot_type='individual')
    plot_benchmark_metrics(benchmark_data, plot_type='average')
    
    return benchmark_data

def main():
    # Generate all plots (individual benchmarks and averages)
    print('\nGenerating variance and max/min plots for all benchmarks and averages...')
    plot_benchmark_metrics_with_averages()

if __name__ == '__main__':
    main()