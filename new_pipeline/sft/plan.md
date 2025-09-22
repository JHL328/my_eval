# Implementation Plan for MMLU Redux & GPQA Diamond Evaluation

## Overview
Implement MMLU Redux and GPQA Diamond evaluations using lighteval framework, following the existing 3-layer pipeline structure (submitter.sh → evaluate_*.sh → evaluate_*.py), using SFT models from `model.py`, with 0-shot evaluation and single generation per sample.

## Architecture Design

### 1. Pipeline Structure (Following GSM8K Pattern)
```
submitter.sh (entry point)
    ├── evaluate_lighteval.sh (SLURM batch script)
    │   └── evaluate_lighteval.py (orchestrator)
    │       └── Per-model SLURM jobs using lighteval
```

### 2. Key Requirements
- **Models**: Use `SFT_MODEL_MAP` from `/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/model.py`
- **Evaluation**: 0-shot, single generation per sample (no pass@k)
- **Framework**: lighteval with vLLM backend
- **Output Format**: JSON results + metrics.txt per model

## Implementation Steps

### Phase 1: Core Scripts Development

#### 1.1 Create `evaluate_lighteval.py` (Manager Script)
```python
# Features:
- Load SFT_MODEL_MAP from model.py
- Submit per-model SLURM jobs
- Configure lighteval tasks based on --task argument
- Monitor job completion
- Aggregate results
```

Key configurations:
- MMLU Redux: All 57 subjects via `lighteval|mmlu_redux_2:{subset}|0`
- GPQA Diamond: `lighteval|gpqa:diamond|0`
- Output directory structure: `/mnt/sharefs/users/haolong.jia/result/{task}_sft/{model_name}/`

#### 1.2 Create `evaluate_lighteval.sh` (SLURM Batch Script)
```bash
# SLURM configuration:
- Job name: eval_lighteval_manager_${SLURM_JOB_ID}
- Output logs: /mnt/weka/home/haolong.jia/eval/runs/
- Resources: 8 CPUs, 16GB RAM
- Calls evaluate_lighteval.py with --submit_jobs flag
```

#### 1.3 Update `submitter.sh`
Add new task entries:
```bash
elif [[ "$TASK_NAME" == "mmlu_redux" ]]; then
    CMD="sbatch evaluate_lighteval.sh --task mmlu_redux --type sft"
elif [[ "$TASK_NAME" == "gpqa_diamond" ]]; then
    CMD="sbatch evaluate_lighteval.sh --task gpqa_diamond --type sft"
```

### Phase 2: Task Configuration

#### 2.1 MMLU Redux Task Configuration
```python
MMLU_REDUX_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    # ... all 57 subjects
]

# Task list format for lighteval:
tasks = [f"lighteval|mmlu_redux_2:{subject}|0" for subject in MMLU_REDUX_SUBJECTS]
```

#### 2.2 GPQA Diamond Task Configuration
```python
GPQA_TASKS = ["lighteval|gpqa:diamond|0"]
```

### Phase 3: Per-Model Job Script Generation

Each model job will execute:
```bash
python -m lighteval vllm \
    --model-args="pretrained=$MODEL_PATH" \
    --tasks="$TASK_LIST" \
    --output-dir="$OUTPUT_DIR" \
    --save-details \
    --dataset-loading-processes=8 \
    --max-samples=-1
```

### Phase 4: Results Processing

#### 4.1 Output Files per Model
- `results.json`: Raw lighteval output
- `metrics.txt`: Formatted accuracy scores
- `slurm.out/err`: Job logs

#### 4.2 Aggregation
- Create summary JSON at base output directory
- Calculate average scores across subjects (for MMLU Redux)
- Format results for leaderboard submission

## Task-Specific Details

### MMLU Redux
- **Metrics**: Accuracy per subject, overall average
- **Output Path**: `/mnt/sharefs/users/haolong.jia/result/mmlu_redux_sft/`
- **Subjects**: 57 subjects from MMLU Redux 2.0
- **Evaluation**: 0-shot, single generation

### GPQA Diamond
- **Metrics**: Accuracy
- **Output Path**: `/mnt/sharefs/users/haolong.jia/result/gpqa_diamond_sft/`
- **Dataset**: GPQA Diamond subset
- **Evaluation**: 0-shot, single generation

## Validation Plan

1. **Dry Run**: Test with single model on subset of data
2. **Output Verification**: Ensure metrics.txt and results.json are created
3. **Accuracy Check**: Verify scores are reasonable (not 0% or 100%)
4. **Integration Test**: Run through full submitter.sh pipeline
5. **Multi-model Test**: Verify queue management with 2-3 models

## Environment Setup

```bash
# Conda environment (use harness-eval since lighteval install on this)
source /mnt/weka/home/haolong.jia/miniconda3/bin/activate harness-eval

# lighteval should already be installed in:
/mnt/weka/home/haolong.jia/eval/RL-eval/lighteval/
```

