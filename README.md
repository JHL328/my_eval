# RL-eval

## Pre-requisite
For AIME, users need to follow `./qwen2.5-math/README.md` to have all the dependencies installed in a conda env `qwen-eval`.
For other benchmarks, users need to follow `./lm-evaluation-harness/README.md` to have all the dependencies installed in a conda env `harness-eval`.


## Quick Start
```
python main.py

```

Currently it supports 5 benchmarks: 
```
BENCHMARKS=["AIME", "gpqa", "ifeval", "mmlu", "mmlu_pro"]
```
remove unnecessary benchmarks from `BENCHMARKS` as needed.