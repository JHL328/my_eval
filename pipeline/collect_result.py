#!/usr/bin/env python3
import os
import re
import json
import glob
import csv
from collections import defaultdict

"""
This script is used to collect the main metrics of the models.
usage:
    python collect_result.py -b /path/to/benchmark_dir -o /path/to/output.csv
"""

def extract_main_metrics(directory, main_benchmark):
    """
    only extract the main metric of the main benchmark, skip the subbenchmark and stderr
    """
    model_name = os.path.basename(directory)
    results = []
    main_metric_names = ['accuracy', 'score', 'overall', 'pass@k']
    # main_metric_names = ['exact_match', 'accuracy', 'score', 'overall']

    # try to find the json file
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('results_') and file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    if json_files:
        latest_json = max(json_files, key=os.path.getmtime)
        with open(latest_json, 'r') as f:
            data = json.load(f)

        # only keep the main metric, skip the subbenchmark and stderr
        metrics = data.get('results', {}).get(main_benchmark, {})
        for metric, value in metrics.items():
            # only keep the main metric, skip the stderr
            if any(metric == m or metric.startswith(m + ',') for m in main_metric_names):
                if not metric.endswith('stderr') and isinstance(value, (int, float)):
                    results.append((model_name, metric, value))
    return results

def model_sort_key(model_name):
    # Try to extract trailing _number for sorting
    m = re.match(r"(.+?)_(\d+)$", model_name)
    if m:
        prefix, num = m.groups()
        return (prefix, int(num))
    else:
        return (model_name, float('inf'))

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract main benchmark metrics')
    parser.add_argument('-b', '--benchmark_dir', help='Directory containing model results')
    parser.add_argument('-o', '--output', default='benchmark_summary.csv', 
                        help='Output file path (default: benchmark_summary.csv)')
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir
    if benchmark_dir is None:
        parser.error('You must specify -b or --benchmark_dir')
    main_benchmark = os.path.basename(benchmark_dir.rstrip('/'))

    # get all model directories
    model_dirs = [os.path.join(benchmark_dir, d) for d in os.listdir(benchmark_dir) 
                 if os.path.isdir(os.path.join(benchmark_dir, d))]

    # extract the metrics of all models
    all_results = []
    for model_dir in model_dirs:
        results = extract_main_metrics(model_dir, main_benchmark)
        all_results.extend(results)

    # sort: model (sort by the number in the model name), metric
    all_results.sort(key=lambda x: (model_sort_key(x[0]), x[1]))

    # write the results as csv
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Metric', 'Value'])
        for model_name, metric, value in all_results:
            writer.writerow([model_name, metric, f"{value:.4f}"])

    print(f"extracted {len(all_results)} metrics, results saved to {args.output}")

if __name__ == "__main__":
    main()