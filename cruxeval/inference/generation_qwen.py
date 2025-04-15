import subprocess
import os

temperatures = [0.2, 0.8]
tasks = ["input", "output"]
cots = [True, False]

base_cmd = [
    "python", "main.py",
    "--model", "Qwen/Qwen2.5-32B-Instruct",
    "--use_auth_token",
    "--trust_remote_code",
    "--batch_size", "10",
    "--n_samples", "10",
    "--max_length_generation", "1024",
    "--precision", "bf16",
    "--limit", "800",
    "--save_generations",
    "--start", "0",
    "--end", "800",
    "--shuffle",
    "--tensor_parallel_size", "1"
]

gpu_list = list(range(8))  # GPUs 0-7
processes = []
os.makedirs("logs", exist_ok=True)

from itertools import product

combinations = list(product(temperatures, tasks, cots))
for i, (temp, task, cot) in enumerate(combinations):
    output_dir = f"model_generations_raw/Qwen32B_temp{temp}_cot{cot}_{task}"
    os.makedirs(output_dir, exist_ok=True)
    cmd = base_cmd + [
        "--tasks", f"{task}_prediction",
        "--temperature", str(temp),
        "--save_generations_path", f"{output_dir}/shared_0.json"
    ]
    if cot:
        cmd += [
            "--cot"
        ]
    gpu = gpu_list[i]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = f"logs/temp{temp}_{task}_cot{cot}.log"
    log_file = open(log_path, "w")
    print(f"Running with temperature={temp} task={task} cot={cot} running gpu {gpu}")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    processes.append((proc, log_file))

for proc, log_file in processes:
    proc.wait()
    log_file.close()
