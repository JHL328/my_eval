from vllm import LLM, SamplingParams

# huggingface repo id
model_path = "/mnt/sharefs/users/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-40000"


# create LLM instance
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=1,
    enable_prefix_caching=True,
    # trust_remote_code=True,   # allow model to have custom code
    disable_custom_all_reduce=True
)

# sampling parameters (generate up to 50 tokens)
sampling_params = SamplingParams(max_tokens=50, temperature=0.7, top_p=0.95)

# run inference
prompts = [
    "What is the capital of France?",
    "Explain what a black hole is in simple terms."
]

outputs = llm.generate(prompts, sampling_params)

# print results
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output.outputs[0].text.strip()}")
    print("=" * 50)