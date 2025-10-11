#!/usr/bin/env python3
"""
Batch inference utility for Enigmata parquet datasets.

For every parquet under the provided data root we:
  1. Read the existing prompt/meta information.
  2. Generate completions with a VLLM-backed model.
  3. Append the model output as a new `output` column.
  4. Save the enriched parquet under `<data_root>/<model_name>/...`.

Optionally, the script can run the Enigmata verifier afterwards.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.append(str(Path(__file__).resolve().parent))
from model import Model_map, get_model_map_by_type  # noqa: E402


# #########################
# CLI argument parsing helpers
# #########################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLLM inference across Enigmata parquet files.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/sharefs/users/haolong.jia/result/enigmata/Enigmata-Eval"),
        help="Root directory that contains the source parquet files.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key or alias. Accepts a path from Model_map or the alias value.",
    )
    parser.add_argument(
        "--model-type",
        choices=("base", "sft"),
        default="base",
        help="Which model map to reference when resolving the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of prompts to send per VLLM batch.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length the model should accommodate.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype hint passed to VLLM (e.g. auto, float16, bfloat16).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="VLLM tensor parallelism degree.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory VLLM may use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per prompt.",
    )
    parser.add_argument(
        "--stop",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of stop strings to truncate generations.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many parquet files (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip parquet files that already have an output saved for this model.",
    )
    parser.add_argument(
        "--rope-scaling-type",
        type=str,
        default=None,
        help="Rope scaling strategy (e.g. linear, dynamic).",
    )
    parser.add_argument(
        "--rope-scaling-factor",
        type=float,
        default=None,
        help="Rope scaling factor passed to VLLM.",
    )
    parser.add_argument(
        "--rope-scaling-base",
        type=int,
        default=None,
        help="Base length for rope scaling (optional).",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="After generation, invoke Enigmata's test_eval.py on each output parquet.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Number of concurrent evaluations to launch when --run-eval is set.",
    )
    return parser.parse_args()


# #########################
# Model resolution utilities
# #########################

def resolve_model(model_key: str, model_type: str) -> Dict[str, str]:
    """
    Resolve the requested model against the registered map.

    Returns a dictionary with both the filesystem path and human-readable name.
    """
    model_map = get_model_map_by_type(model_type)

    if model_key in model_map:
        model_path = model_key
        model_name = model_map[model_key]
        return {"path": model_path, "name": model_name}

    for path, alias in model_map.items():
        if model_key == alias:
            return {"path": path, "name": alias}

    # Allow direct filesystem path if it exists
    candidate = Path(model_key)
    if candidate.exists():
        return {"path": str(candidate), "name": candidate.name}

    raise ValueError(f"Unable to resolve model '{model_key}'. Update model.py or provide a valid path.")


def build_rope_scaling(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    if args.rope_scaling_type is None:
        return None
    scaling: Dict[str, float] = {"type": args.rope_scaling_type}
    if args.rope_scaling_factor is not None:
        scaling["factor"] = args.rope_scaling_factor
    if args.rope_scaling_base is not None:
        scaling["base"] = args.rope_scaling_base
    return scaling


# #########################
# Core processing logic
# #########################

def iter_parquet_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.parquet"))


def ensure_output_path(base_root: Path, model_name: str, source_path: Path) -> Path:
    rel = source_path.relative_to(base_root)
    target = base_root / model_name / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def generate_outputs(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    batch_size: int,
) -> List[str]:
    outputs: List[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results = llm.generate(batch, sampling_params=sampling_params)
        for res in results:
            if not res.outputs:
                outputs.append("")
                continue
            # We only keep the top hypothesis.
            outputs.append(res.outputs[0].text)
    return outputs


def process_file(
    llm: LLM,
    sampling_params: SamplingParams,
    parquet_path: Path,
    output_path: Path,
    batch_size: int,
) -> None:
    df = pd.read_parquet(parquet_path)
    if "prompt" not in df.columns:
        raise KeyError(f"'prompt' column missing in {parquet_path}")

    prompts = df["prompt"].astype(str).tolist()
    generations = generate_outputs(llm, sampling_params, prompts, batch_size)
    if len(generations) != len(df):
        raise RuntimeError(
            f"Generation count mismatch for {parquet_path}: df={len(df)} vs outputs={len(generations)}"
        )

    df = df.copy()
    df["output"] = generations
    df.to_parquet(output_path, index=False)


def run_evaluation(parquet_path: Path) -> None:
    eval_script = Path(__file__).resolve().parent.parent / "Enigmata" / "test_eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Cannot locate evaluation script at {eval_script}")
    cmd = [sys.executable, str(eval_script), "--input", str(parquet_path)]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    model_info = resolve_model(args.model, args.model_type)
    rope_scaling = build_rope_scaling(args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=args.stop,
    )

    llm = LLM(
        model=model_info["path"],
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        rope_scaling=rope_scaling,
    )

    parquet_files = list(iter_parquet_files(data_root))
    if args.limit is not None:
        parquet_files = parquet_files[: args.limit]

    if not parquet_files:
        print(f"No parquet files found in {data_root}")
        return

    print(f"Resolved model: {model_info['name']} ({model_info['path']})")
    print(f"Total parquet files to process: {len(parquet_files)}")

    results: List[Path] = []

    for parquet_path in tqdm(parquet_files, desc="Processing parquet files"):
        output_path = ensure_output_path(data_root, model_info["name"], parquet_path)
        if args.skip_existing and output_path.exists():
            results.append(output_path)
            continue

        process_file(
            llm=llm,
            sampling_params=sampling_params,
            parquet_path=parquet_path,
            output_path=output_path,
            batch_size=args.batch_size,
        )
        results.append(output_path)

    if args.run_eval:
        print("Running evaluation...")
        for output_path in tqdm(results, desc="Evaluating outputs"):
            run_evaluation(output_path)

    # Persist a simple manifest for bookkeeping.
    manifest = {
        "model_name": model_info["name"],
        "model_path": model_info["path"],
        "data_root": str(data_root),
        "processed_files": [str(p) for p in results],
        "rope_scaling": rope_scaling,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "stop": args.stop,
        },
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
    }
    manifest_path = data_root / model_info["name"] / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
