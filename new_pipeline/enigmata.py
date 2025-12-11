#!/mnt/weka/home/haolong.jia/miniconda3/envs/qwen-eval/bin/python3.10
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=400G
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/weka/home/haolong.jia/eval/runs/enigmata_%j.out
#SBATCH --error=/mnt/weka/home/haolong.jia/eval/runs/enigmata_%j.err

"""
Minimal Enigmata batch runner:
- Read source parquet (with `prompt` column) from the shared location.
- Run a model (resolved from model.py) via vLLM to produce `output`.
- Save parquet under <data_root>/<model_name>/... with the same schema + output.
- Optionally call the official verifier to score the outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
from vllm import LLM, SamplingParams

# Ensure we can import from the current directory
# When running via sbatch, __file__ might point to a temporary spool location
# So we hardcode the known path or use CWD if it is correct
PROJECT_PIPELINE_DIR = Path("/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline")
if str(PROJECT_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_PIPELINE_DIR))

from model import get_model_map_by_type  # noqa: E402


DATA_ROOT = Path("/mnt/weka/shrd/k2m/haolong.jia/result/enigmata/Enigmata-Eval/")
EVAL_SCRIPT = Path("/mnt/weka/home/haolong.jia/eval/RL-eval/Enigmata/test_eval.py")

# Tasks that have shown non-zero accuracy on baseline models
TARGET_TASKS = [
    "FOLIO",
    "car_painting",
    "knights_and_knaves",
    "maze",
    "natural_language_navigation",
    "sudoku2"
]


# ----------------------------
# CLI helpers
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Enigmata generation + optional evaluation.")
    parser.add_argument(
        "--model",
        default="/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/math_grateful_refrain/checkpoint-27358",
        help="Model key/path/alias (resolved via model.py). Defaults to SFTmath_grateful_refrain.",
    )
    parser.add_argument("--model-type", choices=("base", "sft"), default="sft", help="Which model map to use.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for vLLM generation.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max sequence length for the model.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Torch dtype hint for vLLM.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel degree for vLLM.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU mem fraction for vLLM.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--run-eval", action="store_true", help="Run official Enigmata verifier after generation.")
    return parser.parse_args()


# ----------------------------
# Model resolution
# ----------------------------
def resolve_model(model_key: str, model_type: str) -> Dict[str, str]:
    model_map = get_model_map_by_type(model_type)

    if model_key in model_map:  # explicit path key
        return {"path": model_key, "name": model_map[model_key]}

    for path, alias in model_map.items():  # alias hit
        if model_key == alias:
            return {"path": path, "name": alias}

    candidate = Path(model_key)
    if candidate.exists():  # raw filesystem path
        return {"path": str(candidate), "name": candidate.name}

    raise ValueError(f"Unable to resolve model '{model_key}'. Update model.py or pass a valid path.")


# ----------------------------
# Core generation
# ----------------------------
def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=4096, # Increased max tokens for reasoning tasks
    )


def generate_outputs(llm: LLM, sampling_params: SamplingParams, prompts: List[str], batch_size: int) -> List[str]:
    outputs: List[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results = llm.generate(batch, sampling_params=sampling_params)
        for res in results:
            outputs.append(res.outputs[0].text if res.outputs else "")
    return outputs


def ensure_output_path(source_path: Path, model_name: str) -> Path:
    base_root = source_path.parent
    target = base_root / model_name / source_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_with_output(df, outputs: List[str], target_path: Path) -> None:
    if len(outputs) != len(df):
        raise RuntimeError(f"Generation count mismatch: df={len(df)} vs outputs={len(outputs)}")
    enriched = df.copy()
    enriched["output"] = outputs
    enriched.to_parquet(target_path, index=False)


# ----------------------------
# Evaluation bridge
# ----------------------------
def run_evaluation(parquet_path: Path) -> dict:
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot locate evaluator at {EVAL_SCRIPT}")
    
    # Run in the directory of the eval script so it can find 'verifiable_tasks'
    eval_cwd = EVAL_SCRIPT.parent
    cmd = [sys.executable, str(EVAL_SCRIPT), "--input", str(parquet_path)]
    
    # Capture output to parse results
    result = subprocess.run(cmd, check=True, cwd=eval_cwd, capture_output=True, text=True)
    
    # Print the full output to stdout as before
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Parse accuracy from the output
    acc = 0.0
    for line in result.stdout.splitlines():
        if "Overall Accuracy:" in line:
            try:
                acc = float(line.split(":")[1].strip())
            except:
                pass
    return {"accuracy": acc, "task": parquet_path.stem}


# ----------------------------
# Main entry
# ----------------------------
def main() -> None:
    args = parse_args()

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Source data root not found at {DATA_ROOT}")

    model_info = resolve_model(args.model, args.model_type)
    sampling_params = build_sampling_params(args)

    print(f"🚀 Initializing model: {model_info['name']}")
    print(f"📂 Model Path: {model_info['path']}")
    
    llm = LLM(
        model=model_info["path"],
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    
    tokenizer = llm.get_tokenizer()

    # Build task list from TARGET_TASKS
    task_files = []
    for task_name in TARGET_TASKS:
        fpath = DATA_ROOT / f"{task_name}_easy.parquet"
        if fpath.exists():
            task_files.append(fpath)
        else:
            print(f"⚠️  Warning: Target task file not found: {fpath.name}")

    print(f"🎯 Found {len(task_files)} valid tasks to process (filtered from whitelist).")
    
    final_results = []
    total_tasks = len(task_files)
    for idx, task_file in enumerate(task_files, 1):
        print(f"\n{'='*50}")
        print(f"🧩 Processing task [{idx}/{total_tasks}]: {task_file.name}")
        print(f"{'='*50}")
        
        try:
            df = pd.read_parquet(task_file)
            if "prompt" not in df.columns:
                print(f"Skipping {task_file.name}: 'prompt' column missing")
                continue

            prompts = df["prompt"].astype(str).tolist()
            
            # Apply chat template for SFT models
            if args.model_type == "sft":
                print("💬 Applying chat template for SFT model...")
                chat_prompts = []
                for p in prompts:
                    messages = [{"role": "user", "content": p}]
                    # We assume the tokenizer has a chat template configured
                    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    chat_prompts.append(formatted)
                prompts = chat_prompts

            generations = generate_outputs(llm, sampling_params, prompts, args.batch_size)

            # Print a few examples
            print(f"\n👀 Preview of first 2 generations:")
            for i in range(min(2, len(generations))):
                 print(f"--- Example {i+1} ---")
                 print(f"📥 Input: {prompts[i][:200]}..." if len(prompts[i]) > 200 else f"📥 Input: {prompts[i]}")
                 print(f"📤 Output: {generations[i][:200]}..." if len(generations[i]) > 200 else f"📤 Output: {generations[i]}")
            
            output_path = ensure_output_path(task_file, model_info["name"])
            save_with_output(df, generations, output_path)
            print(f"💾 Saved outputs to {output_path}")

            if args.run_eval:
                print("⚖️  Running Enigmata verifier...")
                eval_res = run_evaluation(output_path)
                final_results.append(eval_res)
            else:
                final_results.append({"task": task_file.stem, "accuracy": "N/A"})
                
        except Exception as e:
            print(f"❌ Error processing {task_file.name}: {e}")
            final_results.append({"task": task_file.stem, "accuracy": "Error"})
            continue

    # Print Final Summary Table
    print(f"\n{'#'*60}")
    print(f"🚀 FINAL EVALUATION SUMMARY: {model_info['name']}")
    print(f"{'#'*60}")
    
    # Calculate column widths
    task_width = max(len(r['task']) for r in final_results) + 2
    task_width = max(task_width, 20)  # Min width
    
    # Print Header
    print(f"| {'Task Name'.ljust(task_width)} | {'Accuracy'.center(10)} |")
    print(f"|{'-'*(task_width+2)}|{'-'*12}|")
    
    total_acc = 0
    valid_count = 0
    
    for res in final_results:
        acc_str = f"{res['accuracy']:.4f}" if isinstance(res['accuracy'], (int, float)) else str(res['accuracy'])
        print(f"| {res['task'].ljust(task_width)} | {acc_str.center(10)} |")
        
        if isinstance(res['accuracy'], (int, float)):
            total_acc += res['accuracy']
            valid_count += 1
            
    print(f"|{'-'*(task_width+2)}|{'-'*12}|")
    
    if valid_count > 0:
        avg_acc = total_acc / valid_count
        print(f"| {'AVERAGE'.ljust(task_width)} | {f'{avg_acc:.4f}'.center(10)} |")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
