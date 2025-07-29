# Changelog

## [2025-01-29] - Code Evaluation Pipeline Optimization

### Overview
Major refactoring of the code evaluation pipeline to separate GPU and CPU workloads, enabling better resource utilization and parallel processing.

### Architecture Changes

#### Previous Architecture
- Single job runs all 3 steps sequentially on GPU node
- All models processed in one batch after GPU generation
- GPU resources wasted during CPU-intensive sanitize/evaluate steps

#### New Architecture
- **Step 1 (Code Generation)**: GPU job only
- **Steps 2-3 (Sanitize/Evaluate)**: CPU-only jobs
- Each model gets independent CPU job for parallel processing
- GPU jobs automatically submit corresponding CPU jobs upon completion

### New Files Added

1. **`generate_code_gpu.sh`**
   - GPU job script for Step 1 only
   - Manages all model generation jobs
   - Configuration: 1 GPU, 32 CPUs, 300GB memory

2. **`sanitize_evaluate_single_model.sh`**
   - CPU job script for Steps 2-3 per model
   - Takes `--task` and `--model` parameters
   - Configuration: 0 GPU, 24 CPUs, 64GB memory
   - Output: `/mnt/sharefs/users/haolong.jia/result/{task}/logs/{model}_sanitize.out`

3. **`sanitize_evaluate_cpu.sh`** (deprecated)
   - Original batch CPU processing script
   - Replaced by single model approach

### Modified Files

1. **`evaluate_code.py`**
   - Added `--step` parameter: `all`, `generation`, `sanitize_evaluate`
   - Added `--model` parameter for single model processing
   - Parallel sanitize with ProcessPoolExecutor (16 workers)
   - Auto-detects models needing CPU jobs in generation mode
   - Fixed Model_map import path handling

2. **`submitter.sh`**
   - `mbpp`/`humaneval`: New two-stage pipeline
   - `mbpp_old`/`humaneval_old`: Legacy single-job pipeline
   - Updated task support documentation

### Key Features

1. **Resource Optimization**
   - GPU nodes only used for generation
   - CPU-only nodes handle sanitize/evaluate
   - Can utilize 4-5 idle CPU nodes simultaneously

2. **Parallel Processing**
   - Multiple models sanitized in parallel (up to 16)
   - Independent CPU jobs per model
   - No waiting for all GPU jobs to complete

3. **Progress Visibility**
   - Each model has independent log files
   - Real-time progress monitoring per model
   - Example: `tail -f /mnt/sharefs/users/haolong.jia/result/mbpp/logs/Llama-3.1-8B-Instruct_sanitize.out`

4. **Automatic Recovery**
   - Detects models with samples but no results
   - Automatically submits CPU jobs for incomplete models
   - Skips already completed models

### Usage

#### New Two-Stage Pipeline (Recommended)
```bash
bash submitter.sh --task mbpp
bash submitter.sh --task humaneval
```

#### Legacy Single-Job Pipeline
```bash
bash submitter.sh --task mbpp_old
bash submitter.sh --task humaneval_old
```

#### Manual Single Model Processing
```bash
# Run sanitize/evaluate for specific model
python evaluate_code.py --task mbpp --step sanitize_evaluate --model Llama-3.1-8B-Instruct
```

### Performance Improvements
- Step 2 (sanitize) speed: 3-5x faster through parallelization
- Total evaluation time: 30-50% reduction
- GPU utilization: Near 100% (no idle time)
- Better cluster resource utilization

### Technical Details

#### SLURM Configuration
- Fixed variable expansion issues by using command-line parameters
- Dynamic job naming: `--job-name=san_eval_{model_name}`
- Dynamic output paths: `--output={path}/{model}_sanitize.out`

#### Model Detection Logic
```python
# In generation mode:
if results.txt exists -> skip (completed)
elif samples.jsonl exists -> submit CPU job
else -> submit GPU job
```

### Known Limitations
- CPU job limit: `QOSMaxNodePerUserLimit` may queue jobs
- Maximum 16 parallel sanitize workers per CPU job
- Requires manual cleanup of incomplete runs before re-running