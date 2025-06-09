import os
import shutil
import json
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='drop', help='Task name for evaluation')
args = parser.parse_args()
task = args.task

# 不同 task 的主 metric 字段
TASK_METRIC = {
    'drop': ('f1,none', 'drop'),
    'arc_easy': ('acc_norm,none', 'arc_easy'),
    'arc_challenge': ('acc_norm,none', 'arc_challenge'),
    'hellaswag': ('acc_norm,none', 'hellaswag'),
    'piqa': ('acc_norm,none', 'piqa'),
    'winogrande': ('acc,none', 'winogrande'),
    'triviaqa': ('exact_match,remove_whitespace', 'triviaqa'),
    'nq_open': ('exact_match,remove_whitespace', 'nq_open'),
    'commonsense_qa': ('acc,none', 'commonsense_qa'),
    "agieval": ("acc,none", "agieval_en"),
    "openbookqa": ("acc_norm,none", "openbookqa"),
    "social_iqa": ("acc,none", "social_iqa"),
    "truthfulqa_mc1": ("acc,none", "truthfulqa_mc1"),
    # 可按需添加更多 task
}

metric_field, result_key = TASK_METRIC.get(task, ('f1,none', task))

drop_dir = f"/mnt/sharefs/users/haolong.jia/result/{task}"
summary = {}

for model in os.listdir(drop_dir):
    model_path = os.path.join(drop_dir, model)
    if not os.path.isdir(model_path):
        continue

    # 查找中间子目录
    subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    for subdir in subdirs:
        subdir_path = os.path.join(model_path, subdir)
        # 移动并重命名 results_*.json
        for result_file in glob(os.path.join(subdir_path, "results_*.json")):
            shutil.move(result_file, os.path.join(model_path, "result.json"))
        # 处理 sample 文件
        if task == "agieval":
            # 合并所有 samples_agieval_*.jsonl
            all_samples = []
            for sample_file in glob(os.path.join(subdir_path, "samples_agieval_*.jsonl")):
                with open(sample_file, "r") as f:
                    all_samples.extend(f.readlines())
                os.remove(sample_file)
            if all_samples:
                with open(os.path.join(model_path, "sample.jsonl"), "a") as f:
                    f.writelines(all_samples)
        else:
            for sample_file in glob(os.path.join(subdir_path, "samples_*.jsonl")):
                shutil.move(sample_file, os.path.join(model_path, "sample.jsonl"))
        # 删除中间子目录
        shutil.rmtree(subdir_path)

    # 提取 metric 分数
    result_json = os.path.join(model_path, "result.json")
    if os.path.exists(result_json):
        with open(result_json, "r") as f:
            data = json.load(f)
            try:
                if task == "agieval":
                    # 只提取 agieval 总分
                    metric = data["results"]["agieval"][metric_field]
                    summary[model] = {metric_field: metric}
                elif task == "truthfulqa":
                    # 汇总 truthfulqa 所有子任务
                    metrics = {}
                    for subtask, (metric_field_sub, result_key_sub) in {
                        "mc1_acc": ("acc,none", "truthfulqa_mc1"),
                        "mc2_acc": ("acc,none", "truthfulqa_mc2"),
                        "gen_bleu": ("bleu_max,none", "truthfulqa_gen"),
                        "gen_rouge1": ("rouge1_max,none", "truthfulqa_gen"),
                    }.items():
                        try:
                            metrics[subtask] = data["results"][result_key_sub][metric_field_sub]
                        except Exception as e:
                            print(f"Error reading {metric_field_sub} for {model} subtask {subtask}: {e}")
                    summary[model] = metrics
                else:
                    metric = data["results"][result_key][metric_field]
                    summary[model] = {metric_field: metric}
            except Exception as e:
                print(f"Error reading {metric_field} for {model}: {e}")

# 保存总的 result.json
with open(os.path.join(drop_dir, "result.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("处理完成！") 