计划

明确需求：核对 lm-evaluation-harness 中的 _mmlu.yaml 任务定义，确认调用方式（CLI 参数 / Python API）以及所需模型配置、输出路径约定。
设计脚本流程：参考 evaluate_likelihood.{sh,py}，确定 evaluate_harness.sh 的参数解析、环境激活、日志输出及对 Python 脚本的调用方式，并规划 Python 脚本的主要步骤（模型过滤、任务配置、Slurm 提交、结果汇总）。
编写与验证：先实现 Python 脚本中的核心函数，再填充 Shell 脚本；完成后检查命令行参数、输出目录、依赖路径（尤其是 _mmlu.yaml）是否正确，最后补充必要的日志与错误处理。


脚本结构设计

RL-eval/new_pipeline/evaluate_harness.sh

顶部 Slurm 配置块（job 名称、日志输出路径、资源需求）。
激活 harness-eval 环境的步骤。
参数解析函数或循环，仅提取 --task（后续可拓展）。
构造基于任务名的 .out/.err 路径。
调用 python -u new_pipeline/evaluate_harness.py "$@" > … 2> …。
结束状态检查与提示信息。
RL-eval/new_pipeline/evaluate_harness.py

parse_args()：处理 --task、--limit-models、--dry-run 等潜在参数。
load_model_map()：重用/引用 Model_map，必要时支持白名单过滤。
build_task_config(task)：根据任务名生成 harness 调用所需信息（指向 _mmlu.yaml、few-shot 数、其他 CLI flags）。
prepare_directories(task)：创建结果、日志、job 脚本目录并返回路径。
generate_sbatch_script(model_info, task_cfg, paths)：产出单模型 Slurm 脚本，包含 lm_eval 命令和 _mmlu.yaml 的配置加载方式。
submit_jobs(job_scripts)：提交 Slurm、收集 Job ID。
wait_for_jobs(job_ids)：轮询 squeue 等待任务完成。
collect_metrics(task_cfg, paths)：读取每个模型 result.json，抽取主要指标并写汇总文件。
main()：按顺序调用上述函数，打印进度日志与 summary。
每个函数职责清晰，可复用 evaluate_likelihood.py 的部分逻辑（尤其目录与 Slurm 相关），同时针对 _mmlu.yaml 添加专用的任务配置与 post-processing。完成实现后建议补充一个最小任务的 dry run 或本地 smoke test，确认 harness 能正确读取配置。