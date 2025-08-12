import argparse
import os
import sys
from tqdm import tqdm
from vllm import LLM, SamplingParams
import json

# Set task-specific torch compile cache to avoid conflicts between humaneval and mbpp
def setup_cache_dir(dataset_name, model_name):
    cache_base = os.path.expanduser("~/.cache/vllm/torch_compile_cache")
    cache_dir = os.path.join(cache_base, f"{dataset_name}_{model_name.replace('/', '_')}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCH_COMPILE_CACHE_DIR"] = cache_dir
    print(f"[INFO] Using torch compile cache: {cache_dir}")
    return cache_dir

# Define EOS tokens based on EvalPlus practices
# Common EOS tokens relevant for code generation
EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#"
]
MBPP_EOS_TOKENS = EOS + ["\n###", "\nassert", "\n```"]
HUMANEVAL_EOS_TOKENS = EOS + ["\ndef", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]

def get_dataset_problems(dataset_name):
    """Loads dataset problems from evalplus.data."""
    if dataset_name == "humaneval":
        from evalplus.data import get_human_eval_plus
        print("[INFO] Loading humaneval(+) dataset")
        return get_human_eval_plus()
    elif dataset_name == "mbpp":
        from evalplus.data import get_mbpp_plus
        print("[INFO] Loading mbpp(+) dataset")
        return get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_stop_tokens_for_dataset(dataset_name):
    """Returns a list of stop tokens specific to the dataset."""
    if dataset_name == "humaneval":
        return HUMANEVAL_EOS_TOKENS
    elif dataset_name == "mbpp":
        return MBPP_EOS_TOKENS
    return EOS

def main():
    parser = argparse.ArgumentParser(description="Generate code samples using vLLM.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the HuggingFace model or VLLM-supported model identifier.")
    parser.add_argument('--dataset', type=str, required=True, choices=['humaneval', 'mbpp'], help="Dataset to generate samples for.")
    parser.add_argument('--n_samples', type=int, default=64, help="Number of samples to generate per problem.")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save generated samples.")
    parser.add_argument('--max_tokens', type=int, default=1024, help="Maximum number of tokens to generate for each sample.")
    # trust_remote_code might be needed for some models
    parser.add_argument('--trust_remote_code', action='store_true', help="Trust remote code for tokenizer/model loading.")
    # gpu_memory_utilization can be helpful for large models
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="GPU memory utilization for vLLM.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_problems = get_dataset_problems(args.dataset)
    stop_tokens = get_stop_tokens_for_dataset(args.dataset)

    # Setup task-specific cache directory
    model_name = os.path.basename(args.model_path)
    setup_cache_dir(args.dataset, model_name)

    print(f"[INFO] Loading model: {args.model_path} with TP={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto" # or bfloat16, float16
    )

    sampling_params = SamplingParams(
        n=args.n_samples, # Generate n_samples directly per prompt
        temperature=args.temperature,
        top_p=0.95, # A common value for top_p sampling
        max_tokens=args.max_tokens,
        stop=stop_tokens,
        # presence_penalty=0.0, # Default
        # frequency_penalty=0.0, # Default
    )
    if args.temperature == 0: # Greedy decoding
        # For greedy decoding, n_samples > 1 doesn't make sense as it will produce identical outputs.
        # However, EvalPlus might still expect n_samples files for pass@k. We'll generate one and duplicate if needed, or just generate one.
        # For now, let's adjust n to 1 if temp is 0 and print a warning, assuming user wants diverse samples if n_samples > 1.
        if args.n_samples > 1:
            print("[WARN] Temperature is 0 (greedy decoding), but n_samples > 1. Only one unique sample will be generated. "
                  "If multiple identical files are needed for pass@k, this script will generate only one.")
            # To strictly generate N identical files, one would replicate the single output N times.
            # For simplicity, we'll let vLLM generate n=1 if temp=0 and n_samples>1 to avoid identical computation, 
            # but this means fewer than n_samples files if user truly wanted identical copies from greedy.
            # A better approach for strict N files in greedy: generate 1, then copy it N-1 times.
            # For now, we let vLLM handle n=args.n_samples, it will likely return identical outputs for temp=0.
            pass # Let vLLM handle n=args.n_samples; it will return identical outputs for temp=0
        sampling_params.top_p = 1.0
        sampling_params.temperature = 0.0
        print("Using greedy decoding (temperature=0).")

    print(f"Generating {args.n_samples} samples per problem for {args.dataset}...")
    all_samples = []  # for collecting all samples
    for task_id, problem_data in tqdm(dataset_problems.items(), desc=f"Processing {args.dataset}"):
        prompt = problem_data["prompt"]
        task_file_prefix = task_id.replace('/', '_')
        
        # Single call to llm.generate to get all n_samples
        all_request_outputs = llm.generate([prompt], sampling_params) 

        if all_request_outputs and all_request_outputs[0].outputs:
            completions = all_request_outputs[0].outputs # List of CompletionOutput
            
            if len(completions) < args.n_samples:
                print(f"Warning: For task {task_id}, vLLM returned {len(completions)} samples, less than requested {args.n_samples}.")

            for i, completion_output in enumerate(completions):
                generated_text = completion_output.text
                
                # Basic post-processing: if ```python is found, take the content within the first block
                if "```python\n" in generated_text:
                    generated_text = generated_text.split("```python\n", 1)[1].split("\n```", 1)[0]
                elif "```" in generated_text: # Handle cases where only ``` is present
                     generated_text = generated_text.split("```",1)[1].split("\n```",1)[0]

                # Since we're using base models, we need to prepend the prompt to make a complete solution
                complete_solution = prompt + generated_text
                
                # add sample to all_samples
                all_samples.append({
                    "task_id": task_id,
                    "_identifier": f"{task_id}_{i}",
                    "solution": complete_solution  # Use "solution" instead of "completion"
                })
            # If vLLM returned fewer samples than requested, create placeholder samples for the remainder
            for i in range(len(completions), args.n_samples):
                print(f"Creating placeholder for missing sample {i} for task {task_id}")
                all_samples.append({
                    "task_id": task_id,
                    "_identifier": f"{task_id}_{i}",
                    "solution": prompt + "# Generation failed or not provided by vLLM"  # Include prompt for consistency
                })
        else:
            print(f"Warning: No output generated by vLLM for task {task_id}. Creating {args.n_samples} placeholder samples.")
            for i in range(args.n_samples):
                all_samples.append({
                    "task_id": task_id,
                    "_identifier": f"{task_id}_{i}",
                    "solution": prompt + "# Generation failed - no output from vLLM"  # Include prompt for consistency
                })

    # save all samples to samples.jsonl
    output_jsonl_path = os.path.join(args.output_dir, "samples.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✅ all samples are saved to {output_jsonl_path}")

if __name__ == "__main__":
    main()
