import json
import pandas as pd
import os
import re
from collections import defaultdict

# 设置JSON文件所在目录
json_dir = '/mnt/sharefs/users/haolong.jia/result-k2/iq/'  # 请修改为实际路径

# 需要提取的metrics
metrics_to_extract = [
    "AIME_2025",
    "iq_250_only_hard_math",
    "iq_250_procedural_math",
    "iq_250",
    "AIME_2024 pass@16",
    "AIME_2025 pass@16",
    "iq_250_only_hard_math pass@16",
    "iq_250_procedural_math pass@16",
    "iq_250 pass@16"
]

def parse_filename(filename):
    """解析文件名，提取model name和sample number"""
    # 匹配格式：output_{model_name}_samples_{number}.json
    # 也可能有fix-前缀
    pattern = r'^output_(fix-)?(.+?)_samples_(\d+(?:\.\d+)?).json$'
    match = re.match(pattern, filename)
    
    if match:
        has_fix = bool(match.group(1))
        model_name = match.group(2)
        sample_number = float(match.group(3))
        return has_fix, model_name, sample_number
    return None, None, None

# 收集所有JSON文件信息
file_info = []
for filename in os.listdir(json_dir):
    if filename.endswith('.json') and filename.startswith('output_'):
        has_fix, model_name, sample_number = parse_filename(filename)
        if model_name:
            file_info.append({
                'filename': filename,
                'has_fix': has_fix,
                'model_name': model_name,
                'sample_number': sample_number
            })

# 处理fix和重复问题
# 按model_name分组
model_groups = defaultdict(list)
for info in file_info:
    base_model = info['model_name']
    model_groups[base_model].append(info)

# 过滤和标记
filtered_files = []
for base_model, files in model_groups.items():
    # 检查是否有fix版本
    fix_files = [f for f in files if f['has_fix']]
    non_fix_files = [f for f in files if not f['has_fix']]
    
    if fix_files:
        # 如果有fix版本，只保留fix版本
        files_to_process = fix_files
    else:
        # 如果没有fix版本，保留所有非fix版本
        files_to_process = non_fix_files
    
    # 按sample_number排序
    files_to_process.sort(key=lambda x: x['sample_number'])
    
    # 添加序号标记
    ordinals = ['first', 'second', 'third', 'fourth', 'fifth']
    for i, file_info in enumerate(files_to_process):
        if i < len(ordinals):
            suffix = ordinals[i]
        else:
            suffix = f"{i+1}th"
        
        file_info['final_model_name'] = f"{base_model}-{suffix}"
        filtered_files.append(file_info)

# 读取JSON文件并提取数据
data_rows = []
for file_info in filtered_files:
    filepath = os.path.join(json_dir, file_info['filename'])
    
    try:
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        
        # 创建数据行
        row = {'Model Name': file_info['final_model_name']}
        
        # 提取指定的metrics
        for metric in metrics_to_extract:
            if metric in json_data:
                row[metric] = json_data[metric]
            else:
                row[metric] = None  # 如果metric不存在
                print(f"警告: {file_info['filename']} 中缺少 metric: {metric}")
        
        data_rows.append(row)
        
    except Exception as e:
        print(f"错误: 无法处理文件 {file_info['filename']}: {e}")

# 创建DataFrame并保存为CSV
df = pd.DataFrame(data_rows)

# 按Model Name排序
df = df.sort_values('Model Name')

# 保存CSV文件
output_path = '/mnt/sharefs/users/haolong.jia/result-k2/iq_results.csv'
df.to_csv(output_path, index=False)

print(f"\n处理完成！")
print(f"处理的JSON文件数: {len(filtered_files)}")
print(f"生成的CSV行数: {len(df)}")
print(f"结果已保存到: {output_path}")

# 显示一些统计信息
print("\n模型统计:")
# 统计first/second等的分布
suffix_counts = {}
for model_name in df['Model Name']:
    suffix = model_name.split('-')[-1]
    suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

for suffix, count in sorted(suffix_counts.items()):
    print(f"  {suffix}: {count} 个模型")

# 显示前几行
print("\nCSV前5行预览:")
print(df.head())

# 检查是否有缺失值
missing_counts = df.isnull().sum()
if missing_counts.any():
    print("\n警告: 以下列存在缺失值:")
    for col, count in missing_counts[missing_counts > 0].items():
        print(f"  {col}: {count} 个缺失值")