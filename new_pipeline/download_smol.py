from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# huggingface repo id
model_path = "/mnt/sharefs/users/haolong.jia/RL-model/sft/confident_booth/checkpoint-5472"

# load tokenizer to apply chat template
tokenizer = AutoTokenizer.from_pretrained(model_path)

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
sampling_params = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.95)

# original prompts
original_prompts = [
    "What is the capital of France?",
    "Explain what a black hole is in simple terms."
]

# apply chat template to prompts
templated_prompts = []
for prompt in original_prompts:
    messages = [{"role": "user", "content": prompt}]
    templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    templated_prompts.append(templated_prompt)

# run inference with templated prompts
outputs = llm.generate(templated_prompts, sampling_params)

# print results with comparison
for original, templated, output in zip(original_prompts, templated_prompts, outputs):
    print("=" * 80)
    print(f"Original Prompt:\n{original}\n")
    print(f"After Chat Template:\n{templated}\n")
    print(f"All outputs:\n{output.outputs}")
    print(f"Model Output:\n{output.outputs[0].text.strip()}")
    print("=" * 80)
    print()