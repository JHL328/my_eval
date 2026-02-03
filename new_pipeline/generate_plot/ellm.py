import argparse
import csv
import json
import os
import sys


BENCHMARKS = {
    # output_dir paths follow submitter.sh / evaluate.py / evaluate_gsm8k.py
    "mmlu": "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu/passk.json",
    "mmlu_cot": "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu_flan_pass16/passk.json",
    "mmlu_pro": "/mnt/weka/shrd/k2m/haolong.jia/result/mmlu_pro_pass16/passk.json",
    "bbh": "/mnt/weka/shrd/k2m/haolong.jia/result/bbh_pass16/passk.json",
    "gsm8k": "/mnt/weka/shrd/k2m/haolong.jia/result/gsm8k_pass16/passk.json",
}

BENCHMARK_METRICS = {
    "mmlu": ["pass@1"],
    "mmlu_cot": ["pass@1"],
    "mmlu_pro": ["pass@1"],
    "bbh": ["pass@1"],
}

TARGET_MODEL_PATHS = [
    "/mnt/weka/shrd/k2m/haolong.jia/xllm/checkpoint/k2mobile780M_txt360v2.2_5T_jais64k_bsz16M_seq4k_lr9e-4_cosine_wd0.05_rope128/checkpoint_0300000",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Llama-3.2-3B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen2.5-1.5B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM2-1.7B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Llama-3.2-1B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Mistral-7B-v0.3",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen2.5-3B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen3-1.7B-Base",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen3-4B-Base",
]


def load_passk(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def get_model_name_map():
    new_pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(new_pipeline_dir)
    try:
        from model import get_model_map_by_type  # pylint: disable=import-error
    except Exception:
        return {}
    return get_model_map_by_type("base")


def build_target_models():
    model_map = get_model_name_map()
    targets = []
    for path in TARGET_MODEL_PATHS:
        name = model_map.get(path)
        if name is None:
            name = os.path.basename(path.rstrip("/"))
        targets.append((path, name))
    return targets


def collect_metrics():
    data = {}
    metrics = {}
    for bench, passk_path in BENCHMARKS.items():
        bench_data = load_passk(passk_path)
        data[bench] = bench_data
        if bench in BENCHMARK_METRICS:
            metrics[bench] = BENCHMARK_METRICS[bench]
            continue
        metric_keys = set()
        for _, result in bench_data.items():
            metric_keys.update(result.keys())
        metrics[bench] = sorted(metric_keys)
    return data, metrics


def write_csv(output_path, targets, data, metrics):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    columns = ["model_name", "model_path"]
    for bench in BENCHMARKS.keys():
        for metric in metrics.get(bench, []):
            columns.append(f"{bench}.{metric}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for model_path, model_name in targets:
            row = {"model_name": model_name, "model_path": model_path}
            for bench, bench_data in data.items():
                result = bench_data.get(model_name, {})
                for metric in metrics.get(bench, []):
                    row[f"{bench}.{metric}"] = result.get(metric, "")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/generate_plot/ellm_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    targets = build_target_models()
    data, metrics = collect_metrics()
    write_csv(args.output, targets, data, metrics)
    print(f"Saved CSV to {args.output}")


if __name__ == "__main__":
    main()