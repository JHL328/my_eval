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

- For other benchmarks, users need to create a conda env `harness-eval`:
```
conda create -n harness-eval python=3.10
conda activate harness-eval
cd ./lm-evaluation-harness
pip install -e .
pip install langdetect
```


## Quick Start
update `MODEL_NAME_OR_PATH` in `main.py` and run
```
python main.py
```

Currently it supports 5 benchmarks: 
```
BENCHMARKS=["aime24", "math", "gpqa", "ifeval", "mmlu", "mmlu_pro"]
```
remove unnecessary benchmarks from `BENCHMARKS` as needed.