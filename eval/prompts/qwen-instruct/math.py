import json

system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
few_shot_prompt = ""
N_FEWSHOT = 4
with open("./data/math/train.jsonl", "r") as f:
    for _ in range(N_FEWSHOT):
        line = f.readline()
        d = json.loads(line)
        # print(d)
        few_shot_prompt += d["problem"] + d["solution"] + "\n\n"
    # print(few_shot_prompt)

question_format = """{question}"""