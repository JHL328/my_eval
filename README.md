# RL-eval

## Pre-requisite
- For aime and math, users need to create a conda env `qwen-eval`:
```
conda create -n qwen-eval python=3.10
conda activate qwen-eval
cd ./qwen2.5-math/evaluation/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers
```

- For bigcodebench and livecodebench, users need to create an independent conda env `bigcodebench-eval` and `livecodebench-eval`:
```
conda create -n qwen-eval python=3.10
conda activate qwen-eval
cd ./qwen2.5-math/evaluation/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers
```


- For other benchmarks, users need to create a conda env `harness-eval`:
```
conda create -n harness-eval python=3.10
conda activate harness-eval
cd ./lm-evaluation-harness
pip install -e .
pip install langdetect, immutabledict
```
> [!IMPORTANT]  
> Currently we are seeing OOM issues in MMLU evaluation with lm_eval using vllm as backend, this is [a known bug](https://github.com/EleutherAI/lm-evaluation-harness/issues/2490). To bypass this, users need to add `prompt_logprobs=1` to `SamplingParams` in function `_dummy_run` in vllm: `vllm/worker/model_runner.py`.

## Quick Start
update `MODEL_NAME_OR_PATH` and `BENCHMARKS_TO_RUN` in `main.py` and run
```
python main.py
```

Currently it supports multiple benchmarks: 
```
BENCHMARKS=["aime24", "math", "gpqa", "ifeval", "mmlu", "mmlu_pro", "mbpp", "mbpp_plus", "humaneval", "humaneval_plus"]
```
remove unnecessary benchmarks from `BENCHMARKS` as needed.