import os
import time
import random
from vllm import LLM
from transformers import AutoConfig, AutoTokenizer

def find_local_hf_model(model_name, cache_dir=None):
    """
    Find the local path to a Hugging Face model in the cache directory.
    If not found, raise an error instructing the user to download it first.
    Args:
        model_name (str): Model name in the form 'org/model'.
        cache_dir (str, optional): Path to the Hugging Face cache directory. Defaults to '~/.cache/huggingface/hub'.
    Returns:
        str: Path to the model snapshot directory.
    Raises:
        FileNotFoundError: If the model is not found locally.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/HF/hub")
    if len(model_name.split("/")) != 2:
        return model_name
    org, model = model_name.split("/")
    model_dir_prefix = f"models--{org}--{model}"
    model_dir = os.path.join(cache_dir, model_dir_prefix, "snapshots")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model '{model_name}' not found in local cache at {model_dir}. Please download the model first."
        )
    snapshots = os.listdir(model_dir)
    if not snapshots:
        raise FileNotFoundError(
            f"No snapshots found for model '{model_name}' in {model_dir}. Please download the model first."
        )
    # Sort and pick the latest snapshot (by name, which is usually the commit hash or timestamp)
    latest_snapshot = sorted(snapshots)[-1]
    snapshot_path = os.path.join(model_dir, latest_snapshot)
    return snapshot_path

def get_llm(model_name, n_gpu, max_model_len_override=None):
    model_name = find_local_hf_model(model_name)
    print('Using model:', model_name)


    config = AutoConfig.from_pretrained(model_name)

    max_model_len_config = (
        getattr(config, "max_model_len", None)
        or getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_positions", None)  # GPT-style
    )

    if max_model_len_override is not None:
        if max_model_len_config is not None and max_model_len_override > max_model_len_config:
            max_model_len = max_model_len_config
            print(
                "Requested max_model_len_override exceeds config; "
                f"capping to {max_model_len_config}"
            )
        else:
            max_model_len = max_model_len_override
    else:
        if max_model_len_config is None or max_model_len_config > 16384:
            max_model_len = 16384
        else:
            max_model_len = max_model_len_config

    print(f"Using max_model_len: {max_model_len}; max_model_len_config: {max_model_len_config}")


    max_retries = 10
    for attempt in range(max_retries):
        try:
            if model_name == '/lustrefs/users/haolong.jia/train/nanotron_checkpoints/test':
                llm = LLM(
                    model=model_name,
                    gpu_memory_utilization=0.95,
                    tokenizer="gpt2",
                    tensor_parallel_size=1,
                    enable_prefix_caching=True,
                    max_model_len=16384,
                )
            elif 'NVIDIA-Nemotron-Nano' in model_name:
                llm = LLM(
                    model=model_name,
                    gpu_memory_utilization=0.95,
                    tensor_parallel_size=n_gpu,
                    trust_remote_code=True,
                    mamba_ssm_cache_dtype='float32',
                    max_model_len=16384,
                )
            else:
                llm = LLM(
                    model=model_name,
                    gpu_memory_utilization=0.95,
                    tensor_parallel_size=n_gpu,
                    enable_prefix_caching=True,
                    max_model_len=16384,
                )
            break  # If successful, exit the loop
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.randint(2, 15))
            else:
                raise  # Re-raise the last exception if all retries fail
    return llm

def get_tokenizer(model_name):
    model_name = find_local_hf_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer