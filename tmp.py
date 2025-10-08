#!/usr/bin/env python3
"""
检查并修复不完整的benchmark result.json文件
从各个模型子文件夹读取结果并合并到主result.json中
"""

import json
import os
import sys
import shutil
from typing import Dict, Optional, List, Tuple
from datetime import datetime

# 添加当前目录到sys.path以便导入model.py
sys.path.insert(0, '/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline')
from new_pipeline.model import Model_map


def load_model_maps() -> Dict[str, str]:
    """加载base model映射"""
    print(f"加载了 {len(Model_map)} 个base model定义")
    return Model_map


def extract_metric_from_subfolder(
    subfolder_path: str, 
    benchmark_name: str, 
    metric_key: str = "acc_norm,none"
) -> Optional[float]:
    """
    从模型子文件夹的result.json中提取指定metric
    
    Args:
        subfolder_path: 模型子文件夹路径
        benchmark_name: benchmark名称
        metric_key: 要提取的metric键名
    
    Returns:
        metric值，如果提取失败返回None
    """
    result_file = os.path.join(subfolder_path, "result.json")
    
    if not os.path.exists(result_file):
        return None
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # 尝试从results字典中提取
        if "results" in data and benchmark_name in data["results"]:
            benchmark_data = data["results"][benchmark_name]
            if metric_key in benchmark_data:
                return benchmark_data[metric_key]
        
        # 某些benchmark可能使用不同的键名，尝试其他可能的metric
        if "results" in data and benchmark_name in data["results"]:
            benchmark_data = data["results"][benchmark_name]
            # 尝试其他可能的metric键名
            fallback_keys = []
            if metric_key == "acc_norm,none":
                fallback_keys = ["acc,none", "exact_match,none", "em,none"]
            elif metric_key == "exact_match,remove_whitespace":
                fallback_keys = ["exact_match,none", "em,none", "acc_norm,none", "acc,none"]
            
            for key in fallback_keys:
                if key in benchmark_data:
                    print(f"  注意: 使用备用metric '{key}' 替代 '{metric_key}'")
                    return benchmark_data[key]
                    
    except Exception as e:
        print(f"  错误: 读取 {result_file} 失败: {e}")
    
    return None


def check_and_fix_benchmark(
    benchmark_path: str,
    model_map: Dict[str, str],
    dry_run: bool = False,
    backup: bool = True
) -> Tuple[int, int, List[str]]:
    """
    检查并修复单个benchmark的result.json文件
    
    Args:
        benchmark_path: benchmark文件夹路径
        model_map: 模型映射字典
        dry_run: 如果为True，只检查不修改
        backup: 是否备份原文件
    
    Returns:
        (已存在的模型数, 新增的模型数, 失败的模型列表)
    """
    benchmark_name = os.path.basename(benchmark_path)
    main_result_file = os.path.join(benchmark_path, "result.json")
    
    # 根据benchmark名称确定metric键
    if benchmark_name in ["triviaqa", "nq_open"]:
        metric_key = "exact_match,remove_whitespace"
    else:
        metric_key = "acc_norm,none"
    
    # 检查主result.json是否存在
    if not os.path.exists(main_result_file):
        print(f"  跳过: {benchmark_name} 没有result.json文件")
        return 0, 0, []
    
    # 读取现有的result.json
    try:
        with open(main_result_file, 'r') as f:
            main_results = json.load(f)
    except Exception as e:
        print(f"  错误: 无法读取 {main_result_file}: {e}")
        return 0, 0, []
    
    # 获取所有应该存在的模型名称
    all_model_names = set(model_map.values())
    existing_models = set(main_results.keys())
    missing_models = all_model_names - existing_models
    
    if not missing_models:
        print(f"  {benchmark_name}: 所有 {len(existing_models)} 个模型都已存在")
        return len(existing_models), 0, []
    
    print(f"  {benchmark_name}: 已有 {len(existing_models)} 个模型，缺失 {len(missing_models)} 个模型 (使用metric: {metric_key})")
    
    # 尝试从子文件夹读取缺失的模型结果
    added_count = 0
    failed_models = []
    
    for model_name in missing_models:
        subfolder_path = os.path.join(benchmark_path, model_name)
        
        if not os.path.exists(subfolder_path):
            # print(f"    - {model_name}: 子文件夹不存在")
            failed_models.append(model_name)
            continue
        
        # 提取metric
        metric_value = extract_metric_from_subfolder(subfolder_path, benchmark_name, metric_key)
        
        if metric_value is not None:
            if not dry_run:
                # 添加到主results中
                main_results[model_name] = {metric_key: metric_value}
            added_count += 1
            print(f"    + {model_name}: 成功提取 ({metric_key}={metric_value:.4f})")
        else:
            # print(f"    - {model_name}: 无法提取metric")
            failed_models.append(model_name)
    
    # 保存更新后的结果
    if added_count > 0 and not dry_run:
        # 备份原文件
        if backup:
            backup_file = main_result_file + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(main_result_file, backup_file)
            print(f"  备份创建: {backup_file}")
        
        # 保存更新后的文件
        with open(main_result_file, 'w') as f:
            json.dump(main_results, f, indent=2)
        print(f"  已更新: 添加了 {added_count} 个模型结果")
    
    return len(existing_models), added_count, failed_models


def main():
    """主函数"""
    print("=" * 80)
    print("开始检查和修复benchmark result.json文件")
    print("=" * 80)
    
    # 加载模型映射
    model_map = load_model_maps()
    
    # 设置结果目录
    result_dir = "/mnt/sharefs/users/haolong.jia/result"
    
    # 获取所有benchmark文件夹
    benchmarks = [
        d for d in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, d)) and not d.endswith('.csv')
    ]
    
    print(f"\n找到 {len(benchmarks)} 个benchmark文件夹")
    
    # 统计信息
    total_existing = 0
    total_added = 0
    total_failed = 0
    benchmarks_updated = []
    
    # 首先以dry_run模式运行，显示将要进行的更改
    print("\n" + "=" * 80)
    print("第一步: 检查模式 (不修改文件)")
    print("=" * 80)
    
    for benchmark in sorted(benchmarks):
        benchmark_path = os.path.join(result_dir, benchmark)
        existing, added, failed = check_and_fix_benchmark(
            benchmark_path, model_map, dry_run=True
        )
        
        if added > 0:
            benchmarks_updated.append(benchmark)
            total_existing += existing
            total_added += added
            total_failed += len(failed)
    
    # 显示统计
    print("\n" + "=" * 80)
    print("检查完成统计:")
    print(f"  需要更新的benchmark: {len(benchmarks_updated)} 个")
    print(f"  已存在的模型结果: {total_existing} 个")
    print(f"  可以添加的模型结果: {total_added} 个")
    print(f"  无法提取的模型: {total_failed} 个")
    print("=" * 80)
    
    if total_added == 0:
        print("\n没有需要更新的内容")
        return
    
    # 询问是否继续
    print("\n是否执行更新? (yes/no): ", end="")
    response = input().strip().lower()
    
    if response != 'yes' and response != 'y':
        print("已取消操作")
        return
    
    # 执行实际更新
    print("\n" + "=" * 80)
    print("⚠️第二步: 执行更新")
    print("=" * 80)
    
    for benchmark in benchmarks_updated:
        benchmark_path = os.path.join(result_dir, benchmark)
        check_and_fix_benchmark(benchmark_path, model_map, dry_run=False)
    
    print("\n" + "=" * 80)
    print("✅更新完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()