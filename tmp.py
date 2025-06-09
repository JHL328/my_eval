import os
from new_pipeline.model import Model_map

root = "/mnt/sharefs/users/haolong.jia/checkpoint"
all_ckpts = []

# 递归所有子目录，收集所有 checkpoint 路径（只收集叶子目录）
for dirpath, dirnames, filenames in os.walk(root):
    # 如果该目录下没有子目录，认为是 checkpoint 目录
    if not dirnames:
        all_ckpts.append(dirpath)

model_keys = set(Model_map.keys())

missing = [ckpt for ckpt in all_ckpts if ckpt not in model_keys]
print("Missing checkpoints:")
for m in missing:
    print(m)