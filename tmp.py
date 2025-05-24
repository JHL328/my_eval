import os
import re
import shutil

# base dir
base_dir = "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera"

# walk through the dir
for root, dirs, files in os.walk(base_dir):
    # find the dir that starts with tokenmix_ablation_
    if os.path.basename(root).startswith("tokenmix_ablation_"):
        # get the experiment name (e.g. lucid_fermat)
        exp_parts = os.path.basename(root).split("_")
        if len(exp_parts) >= 3:
            exp_name = exp_parts[2]
            
            # find the dir that is all digits
            for dirname in dirs:
                if dirname.isdigit():
                    old_path = os.path.join(root, dirname)
                    new_name = f"{exp_name}_{dirname}"
                    new_path = os.path.join(root, new_name)
                    
                    # rename
                    os.rename(old_path, new_path)
                    print(f"rename: {old_path} -> {new_path}")