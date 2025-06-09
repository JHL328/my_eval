根据你最新的需求和现有的 design，下面是**新版 main.py 的详细设计计划**，完全适配“通过 gpus_per_task 控制数据并行，不再手动切分数据”的思路。

---

# 新版 main.py 设计计划

## 1. 目标与原则

- **目标**：自动遍历所有模型和所有 benchmark，生成并提交 Slurm 任务，每个任务自动利用 gpus_per_task 张 GPU 进行数据并行（data parallel），无需手动切分数据。
- **原则**：
  - 只用数据并行（data parallel），TP=1。
  - 每个 Slurm job 申请 gpus_per_task 张 GPU，vllm/lm_eval 自动分配数据到每张卡。
  - 评测参数全部由 task.py/SUPPORTED_BENCHMARKS 统一管理。
  - 支持多节点（num_nodes），自动调度最大并发 job 数。

---

## 2. 主要结构与流程

### 2.1 配置与参数

- **模型列表**：从 model.py 的 Model_map 读取。
- **benchmark 配置**：从 task.py 的 SUPPORTED_BENCHMARKS 读取。
- **gpus_per_task**：命令行参数，决定每个 job 申请多少 GPU（即 data parallel size）。
- **num_nodes**：命令行参数，决定总共可用 GPU 数量。

### 2.2 任务队列生成

- 遍历 Model_map 和 SUPPORTED_BENCHMARKS，生成所有 (model, benchmark) 组合的任务队列。
- 每个任务包含：模型路径、benchmark 名、评测参数。

### 2.3 Job 脚本生成与提交

- 每个任务生成一个 Slurm 脚本，内容包括：
  - `#SBATCH --gres=gpu:{gpus_per_task}`
  - `--model_args pretrained=...,tensor_parallel_size=1,data_parallel_size={gpus_per_task},...`
  - 其他评测参数（如 tokens、temperature、top_p、n_fewshot 等）。
- 脚本中激活环境，调用 lm_eval/vllm，自动利用所有分配到的 GPU 进行数据并行。

### 2.4 Job 调度与监控

- 支持断点续跑（可选，后续实现）。
- 控制最大并发 job 数：`max_concurrent_jobs = total_gpus // gpus_per_task`。
- 跟踪每个 job 的状态，失败可重试或记录。

### 2.5 结果管理

- 每个模型-benchmark 的输出结果存储在唯一目录下。
- 可选：后处理/合并/统计等。

---

## 3. 主要类与方法

- **ModelEvaluator**
  - `__init__`: 初始化参数、生成任务队列、信号处理等。
  - `generate_task_queue`: 生成所有 (model, benchmark) 组合的任务。
  - `submit_job`: 生成并提交 Slurm 脚本，设置 TP=1, DP=gpus_per_task。
  - `check_job_status`: 查询 job 状态。
  - `run_evaluation`: 主调度循环，控制并发、提交、监控、收尾。

- **main**
  - 解析命令行参数，创建 ModelEvaluator，启动评测。

---

## 4. 关键注意事项

- **不再有数据切分逻辑**，所有数据并行都交给 vllm/lm_eval 的 data_parallel_size 实现。
- **TP 始终为 1**，确保每个模型副本只用一张卡。
- **gpus_per_task** 是唯一影响并行度的参数，建议与节点 GPU 数一致（如 8）。
- **任务队列和 job 脚本生成逻辑要简洁明了**，便于后续维护和扩展。

---

## 5. 可选扩展

- 支持 job 失败自动重试。
- 支持断点续跑（保存/恢复任务队列和状态）。
- 支持多种评测脚本（如 math_eval.py、lm_eval 等）。

---

**如你认可此设计计划，我将按此结构生成新版 main.py 代码。**  
如需补充特殊需求，请直接说明！
