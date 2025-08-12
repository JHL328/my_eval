#!/usr/bin/env python3
"""
Centralized configuration for optimized sanitize and evaluate pipeline.
Based on performance testing results showing optimal performance with:
- 12 workers + 500 batch size achieved 20.54 samples/s (2.09x speedup)
"""

# Optimal configuration based on performance testing
# Test results from performance_report.txt:
# batch_12_workers_500_batch: 20.54 samples/s (best)
# batch_24_workers_200_batch: 20.40 samples/s (close second)
SANITIZE_CONFIG = {
    "n_workers": 12,
    "batch_size": 500,
    "show_progress": True
}

EVALUATE_CONFIG = {
    "parallel": 12,
    "batch_size": 500,
    "show_progress": True
}

# Alternative configurations for different scenarios
CONFIGS = {
    # Best overall performance (default)
    "optimal": {
        "sanitize": {"n_workers": 12, "batch_size": 500},
        "evaluate": {"parallel": 12, "batch_size": 500}
    },
    
    # For high memory systems (96 CPU cores)
    "high_memory": {
        "sanitize": {"n_workers": 24, "batch_size": 200},
        "evaluate": {"parallel": 24, "batch_size": 200}
    },
    
    # For low memory systems
    "low_memory": {
        "sanitize": {"n_workers": 8, "batch_size": 50},
        "evaluate": {"parallel": 8, "batch_size": 50}
    },
    
    # Conservative (stable but slower)
    "conservative": {
        "sanitize": {"n_workers": 8, "batch_size": 100},
        "evaluate": {"parallel": 8, "batch_size": 100}
    }
}

def get_optimal_config(task="mbpp", num_samples=None, config_name="optimal"):
    """
    Get optimal configuration based on task and sample count.
    
    Args:
        task: Task name (mbpp, humaneval, etc.)
        num_samples: Number of samples to process
        config_name: Configuration preset name
        
    Returns:
        dict: Configuration dictionary with sanitize and evaluate settings
    """
    # Use specified config or default to optimal
    if config_name in CONFIGS:
        config = CONFIGS[config_name].copy()
    else:
        config = CONFIGS["optimal"].copy()
    
    # Adjust for small datasets
    if num_samples and num_samples < 1000:
        # Use fewer workers for small datasets to reduce overhead
        config["sanitize"]["n_workers"] = min(8, config["sanitize"]["n_workers"])
        config["sanitize"]["batch_size"] = min(50, config["sanitize"]["batch_size"])
        config["evaluate"]["parallel"] = min(8, config["evaluate"]["parallel"])
        config["evaluate"]["batch_size"] = min(50, config["evaluate"]["batch_size"])
    
    # Task-specific adjustments
    if task == "humaneval":
        # HumanEval has fewer but more complex tests
        # Slightly smaller batches might be better
        config["evaluate"]["batch_size"] = min(300, config["evaluate"]["batch_size"])
    
    return config

def get_sanitize_command(samples_path, config=None, use_optimized=True):
    """
    Generate sanitize command with optimal parameters.
    
    Args:
        samples_path: Path to samples file
        config: Configuration dict or None for default
        use_optimized: Whether to use optimized version
        
    Returns:
        str: Command to run
    """
    if config is None:
        config = SANITIZE_CONFIG
    
    if use_optimized:
        cmd = f"python -m evalplus.sanitize_optimized {samples_path}"
        cmd += f" --n-workers {config['n_workers']}"
        cmd += f" --batch-size {config['batch_size']}"
        if not config.get('show_progress', True):
            cmd += " --no-progress"
    else:
        # Fallback to original version
        cmd = f"python -m evalplus.sanitize {samples_path}"
        cmd += f" --n_workers {config.get('n_workers', 48)}"
    
    return cmd

def get_evaluate_command(dataset, samples_path, config=None, use_optimized=True):
    """
    Generate evaluate command with optimal parameters.
    
    Args:
        dataset: Dataset name (humaneval, mbpp)
        samples_path: Path to samples file
        config: Configuration dict or None for default
        use_optimized: Whether to use optimized version
        
    Returns:
        str: Command to run
    """
    if config is None:
        config = EVALUATE_CONFIG
    
    if use_optimized:
        cmd = f"python -m evalplus.evaluate_optimized"
        cmd += f" --dataset {dataset}"
        cmd += f" --samples {samples_path}"
        cmd += f" --parallel {config['parallel']}"
        cmd += f" --batch-size {config['batch_size']}"
    else:
        # Fallback to original version
        cmd = f"python -m evalplus.evaluate"
        cmd += f" --dataset {dataset}"
        cmd += f" --samples {samples_path}"
        cmd += f" --parallel {config.get('parallel', 48)}"
    
    return cmd

# Performance estimates based on test results
def estimate_time(num_samples, operation="sanitize"):
    """
    Estimate processing time based on test results.
    
    Test results show:
    - Optimal config: 20.54 samples/second
    - Original config: 9.85 samples/second
    
    Args:
        num_samples: Number of samples to process
        operation: "sanitize" or "evaluate"
        
    Returns:
        dict: Time estimates in seconds
    """
    # Based on test results
    optimal_rate = 20.54  # samples per second
    original_rate = 9.85   # samples per second
    
    return {
        "optimal_seconds": num_samples / optimal_rate,
        "optimal_minutes": num_samples / optimal_rate / 60,
        "original_seconds": num_samples / original_rate,
        "original_minutes": num_samples / original_rate / 60,
        "speedup": original_rate / optimal_rate
    }

if __name__ == "__main__":
    # Example usage
    print("Optimal Configuration:")
    print("-" * 40)
    config = get_optimal_config()
    print(f"Sanitize: {config['sanitize']}")
    print(f"Evaluate: {config['evaluate']}")
    
    print("\nExample Commands:")
    print("-" * 40)
    print("Sanitize:", get_sanitize_command("samples.jsonl"))
    print("Evaluate:", get_evaluate_command("mbpp", "samples.jsonl"))
    
    print("\nTime Estimates for 24,192 samples:")
    print("-" * 40)
    estimates = estimate_time(24192)
    print(f"Optimal: {estimates['optimal_minutes']:.1f} minutes")
    print(f"Original: {estimates['original_minutes']:.1f} minutes")
    print(f"Speedup: {estimates['speedup']:.1f}x")