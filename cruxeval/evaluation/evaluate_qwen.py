import subprocess
import os



from itertools import product

temperatures = [0.2, 0.8]
tasks = ["input", "output"]
cots = [True, False]
combinations = list(product(temperatures, tasks, cots))

for i, (temp, task, cot) in enumerate(combinations):
    run_name = f"Qwen32B_temp{temp}_cot{cot}_{task}"
    base_cmd = [
        "python", "evaluate_generations.py",
        "--generations_path", f"../model_generations/{run_name}/generations.json",
        "--scored_results_path", f"evaluation_results/${run_name}.json"
    ]
    cmd = base_cmd
    print(f"Running Evaluation {run_name}")
    subprocess.run(cmd)