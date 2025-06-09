import json
import re

with open('/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_cot_prompts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = {}
for subject, content in data.items():
    # 用正则去掉每个 answer 里的 "Let's think step by step."
    # 匹配 A: Let's think step by step. 及其后面可能的空格
    new_content = re.sub(
        r'(A:\s*)Let\'s think step by step\. ?',  # 匹配 A: Let's think step by step.（允许有空格）
        r'\1',  # 保留 A: 
        content
    )
    new_data[subject] = new_content

with open('/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/mmlu_prompts.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("Done")