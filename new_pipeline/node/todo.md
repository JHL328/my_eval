目标：用 `evaluate_aime_2.py` 跑 AIME24/25（全模型，Avg@32），接入 `submitter.sh` + `model.py`，使用本地数据与 qwen-eval 环境。

实现逻辑/计划：
1) 明确当前输出结构（现有 `evaluate_math.py` / `evaluate_aime.py`）以对齐新脚本的输出。
2) 重构 `evaluate_aime_2.py`：
   - 读取本地 jsonl（AIME24/AIME25），不再使用 idx_start/idx_end。
   - 支持 `--task` 选择数据集（aime24/aime25）。
   - 统一设置：`max_model_len=16284`、`max_tokens=15260`、`n_samples=32`。
   - 逐模型循环（从 `model.py` 的 `model_map` 读取）。
   - 为每个模型输出 `result.csv`（题目 x 采样矩阵）、`metrics.txt`（含 avg@32 / pass@k）、`sample.jsonl`（含原题、预测与得分）。
   - 汇总写入 base_out 的 `result.json`（所有模型）。
3) 调整 `hf_utils.py`：允许 `get_llm` 接受显式 `max_model_len` 覆盖默认值。
4) 更新 `submitter.sh`：aime24/aime25 走新脚本，使用 qwen-eval 环境，移除旧 harness 入口。
5) 确认路径与参数一致后再逐步执行改动。
