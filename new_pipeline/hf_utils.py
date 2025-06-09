import os

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