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
pip install vllm==0.6.6.post1 --no-build-isolation
pip install transformers
```

- There are three directories under code evaluation, and eval_plus includes HumanEval, HumanEval+, MBPP, MBPP+. Users need to create an independent conda env `bigcodebench-eval`, `livecodebench-eval`, and `evalplus-eval`:

BENCH = ["bigcodebench", "livecodebench", "evalplus"]

```
conda create -n {BENCH}-eval python=3.10
conda activate {BENCH}-eval
cd ./code_eval/{BENCH}
pip install -r requirements.txt 
pip install flash-attn --no-build-isolation

For both bigcodebench and livecodebench, the script (`test.sh`) accepts two parameters:

1. ROOT_DIR: The root directory for code_eval. It is also where output logs and files will be stored.
2. MODEL_DIR: The absolute path to the model directory.

Submit the job with your desired root directory and model path:

```bash
sbatch test.sh /path/to/root /absolute/path/to/model
```

For HumanEval(+) and MBPP(+), the script (`test_humaneval.sh`) or (`test_mbpp.sh`) accepts three parameters:

1. ROOT_DIR: The root directory for code_eval. It is also where output logs and files will be stored.
2. MODEL_DIR: The absolute path to the model directory.
3. MODEL: The abbreviation for the model. This parameter only affects the name of the output directory.

Submit the job with your desired root directory and model path:

```bash
sbatch test_humaneval.sh/test_mbpp.sh /path/to/root /absolute/path/to/model /abbreviation/for/model
```

- For other benchmarks, users need to create a conda env `harness-eval`:
```
conda create -n harness-eval python=3.10
conda activate harness-eval
cd ./lm-evaluation-harness
pip install -e .
pip install langdetect immutabledict
pip install vllm==0.6.6.post1 --no-build-isolation
```

> [!IMPORTANT]  
> Currently we are seeing OOM issues in MMLU evaluation with lm_eval using vllm as backend, this is [a known bug](https://github.com/EleutherAI/lm-evaluation-harness/issues/2490). To bypass this, users need to add `prompt_logprobs=1` to `SamplingParams` in function `_dummy_run` in vllm: `vllm/worker/model_runner.py`.

## Quick Start
update `MODEL_NAME_OR_PATH` and `BENCHMARKS_TO_RUN` in `main.py` and run
```
python main.py \
    -p deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --benchmarks aime24 \
    --prompt_type deepseek-distill-qwen \
    --model_size 32
```

Currently it supports multiple benchmarks: 
- aime24
- aime25
- math
- math500
- gpqa_diamond
- ifeval
- mmlu
- mmlu_pro
- mmlu_stem
- bigbench_extrahard
- bigbench_extrahard_verbal

Two other coding benchmarks are supported under `code_eval` folder: BigCodeBench and LiveCodeBench.

## Prompts
To evaluate qwen 2.5 series models, use `--prompt_type qwen25-math-cot`

To evaluate RLHF's SFT models, use `--prompt_type qwen25-math-no-sytem-think-boxed`
