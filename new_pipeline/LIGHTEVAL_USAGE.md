# Lighteval Evaluation Pipeline Usage Guide

## Overview
This pipeline implements MMLU Redux and GPQA Diamond evaluations using the lighteval framework for SFT models.

## Architecture
```
submitter.sh
    ├── evaluate_lighteval.sh (SLURM manager)
    │   └── evaluate_lighteval.py (orchestrator)
    │       └── Per-model SLURM jobs (lighteval vllm)
```

## Tasks Supported

### 1. MMLU Redux
- **Task ID**: `mmlu_redux`
- **Description**: Evaluates models on all 57 MMLU Redux subjects
- **Output**: `/mnt/sharefs/users/haolong.jia/result/mmlu_redux_sft/`

### 2. GPQA Diamond
- **Task ID**: `gpqa_diamond`
- **Description**: Evaluates models on GPQA Diamond dataset
- **Output**: `/mnt/sharefs/users/haolong.jia/result/gpqa_diamond_sft/`

## Usage

### Run Full Evaluation (All SFT Models)
```bash
# MMLU Redux evaluation
bash submitter.sh --task mmlu_redux --type sft

# GPQA Diamond evaluation
bash submitter.sh --task gpqa_diamond --type sft
```

### Test Single Model
```bash
# Submit the test job
sbatch test_lighteval_single.sh

# Monitor the job
squeue -u $USER

# Check logs
tail -f /mnt/weka/home/haolong.jia/eval/runs/test_lighteval_*.out
```

### Direct Testing (Debug Mode)
```bash
# Test single model directly
python evaluate_lighteval.py \
    --task gpqa_diamond \
    --type sft \
    --model_path /path/to/model \
    --model_name test_model \
    --reforce
```

## Configuration

### Key Settings in `evaluate_lighteval.py`
- **Environment**: `harness-eval` conda environment
- **Backend**: vLLM with tensor_parallel_size=1
- **Chat Template**: Enabled for SFT models (--use-chat-template)
- **0-shot**: No few-shot examples used

### Resource Allocation
- **MMLU Redux**: 8 hours, 100GB RAM, 1 GPU
- **GPQA Diamond**: 4 hours, 100GB RAM, 1 GPU

## Output Structure
```
/mnt/sharefs/users/haolong.jia/result/{task}_sft/
├── summary.json              # Aggregated results for all models
├── {model_name}/
│   ├── results.json         # Raw lighteval output
│   ├── metrics.txt          # Processed metrics
│   ├── details/             # Per-sample predictions (if --save-details)
│   └── slurm.out/err       # Job logs
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure model path exists and contains valid Hugging Face format files
   - Check if tokenizer files are present

2. **OOM (Out of Memory)**
   - Reduce batch size in lighteval parameters
   - Use smaller dtype (float16 instead of auto)

3. **Results Not Generated**
   - Check slurm.err for errors
   - Verify lighteval is properly installed in harness-eval environment

### Checking Job Status
```bash
# View running jobs
squeue -u $USER

# Check specific job details
scontrol show job <JOB_ID>

# View job logs
tail -f /mnt/sharefs/users/haolong.jia/result/{task}_sft/{model}/slurm.out
```

## Models Evaluated
The pipeline uses `SFT_MODEL_MAP` from `model.py`, which includes:
- lonely_cone_0_27358
- awesome_kilby_27358
- brave_noether_27358
- confident_booth_27358
- driven_spectacle_27358
- steel_lamb_27358
- gullible_aperitif_27358
- resulting_eggs_27358
- electoral_lithography_27358
- near_habanera_27358

## Notes
- **Lighteval Compatibility**: The pipeline uses local model paths, which lighteval supports through its vLLM backend
- **Chat Template**: Applied automatically for SFT models to ensure proper formatting
- **Metrics**: Accuracy is the primary metric for both tasks