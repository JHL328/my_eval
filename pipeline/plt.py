import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from pathlib import Path
import seaborn as sns

def parse_pass_out_file(file_path):
    """Parse the pass.out file to extract model performance data"""
    models_data = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by model processing sections
    model_sections = re.split(r'Processing model: ([^(]+) \(task: mmlu_flan\)', content)
    
    for i in range(1, len(model_sections), 2):
        model_name = model_sections[i].strip()
        model_content = model_sections[i+1]
        
        # Extract pass@k metrics
        metrics_match = re.search(
            r'Model-level pass@k metrics:\s*\n\s*pass@1: ([0-9.]+)\s*\n\s*pass@2: ([0-9.]+)\s*\n\s*pass@4: ([0-9.]+)\s*\n\s*pass@8: ([0-9.]+)\s*\n\s*pass@16: ([0-9.]+)',
            model_content
        )
        
        if metrics_match:
            models_data[model_name] = {
                'pass@1': float(metrics_match.group(1)),
                'pass@2': float(metrics_match.group(2)),
                'pass@4': float(metrics_match.group(3)),
                'pass@8': float(metrics_match.group(4)),
                'pass@16': float(metrics_match.group(5))
            }
    
    return models_data

def categorize_models(models_data):
    """Categorize models into trained models and open-source baselines"""
    trained_models = {}
    opensource_models = {}
    
    # Define the trained model patterns
    trained_patterns = ['awesome_', 'brave_', 'confident_', 'near_']
    
    for model_name, metrics in models_data.items():
        is_trained = any(model_name.startswith(pattern) for pattern in trained_patterns)
        
        if is_trained:
            # Extract checkpoint number
            parts = model_name.split('_')
            if len(parts) >= 2:
                try:
                    checkpoint = int(parts[1])
                    model_type = parts[0]
                    
                    if checkpoint not in trained_models:
                        trained_models[checkpoint] = {}
                    trained_models[checkpoint][model_type] = metrics
                except ValueError:
                    continue
        else:
            opensource_models[model_name] = metrics
    
    return trained_models, opensource_models

def create_performance_plots(trained_models, opensource_models, output_dir='/mnt/sharefs/users/haolong.jia/result/mmlu_flan_cot_fewshot_plt'):
    """Create performance plots grouped by checkpoint steps"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Define data mix information for trained models
    data_mix_info = {
        'awesome': 'txt360(35%), megamath(30%), general_dataset(35%)',
        'brave': 'txt360(60%), megamath(30%), reasoning_dataset(10%)',
        'confident': 'txt360(70%), megamath(30%)',
        'near': 'txt360(50%), megamath(30%), opencoder(20%)'
    }
    
    # Create fixed color mapping for all models to ensure consistency
    all_model_types = ['awesome', 'brave', 'confident', 'near']
    baseline_models = ['Llama-3.2-1B', 'Llama-3.2-3B', 'Mistral-7B-v0.3', 'Qwen2.5-1.5B', 'Qwen2.5-3B']
    all_models = all_model_types + baseline_models
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    color_mapping = {model: colors[i % len(colors)] for i, model in enumerate(all_models)}
    
    k_values = [1, 2, 4, 8, 16]
    
    # Get available checkpoints (focusing on key ones from dataset mix table)
    key_checkpoints = [7343, 14686, 22029, 29372, 36715, 44058, 51401, 58744, 66087, 73430]
    available_checkpoints = [ckpt for ckpt in key_checkpoints if ckpt in trained_models]
    
    # Create a figure for each checkpoint
    for checkpoint in available_checkpoints:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        checkpoint_data = trained_models[checkpoint]
        
        # Plot trained models for this checkpoint
        for model_type in ['awesome', 'brave', 'confident', 'near']:
            if model_type in checkpoint_data:
                metrics = checkpoint_data[model_type]
                accuracies = [metrics[f'pass@{k}'] for k in k_values]
                # Create label with data mix information
                mix_info = data_mix_info.get(model_type, '')
                label = f'{model_type}_{checkpoint}\n{mix_info}'
                ax.plot(k_values, accuracies, marker='o', linewidth=2, markersize=6, 
                       label=label, color=color_mapping[model_type])
        
        # Add open-source baselines
        for baseline in baseline_models:
            if baseline in opensource_models:
                metrics = opensource_models[baseline]
                accuracies = [metrics[f'pass@{k}'] for k in k_values]
                ax.plot(k_values, accuracies, marker='s', linewidth=2, markersize=6,
                       linestyle='--', alpha=0.7, label=baseline, color=color_mapping[baseline])
        
        ax.set_xlabel('k (Number of Attempts)', fontsize=12)
        ax.set_ylabel('Pass@k Accuracy', fontsize=12)
        ax.set_title(f'MMLU-FLAN CoT Performance at Checkpoint {checkpoint}', fontsize=14, fontweight='bold')
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set y-axis to show the full range from 0 to 1
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mmlu_flan_performance_checkpoint_{checkpoint}.png', 
                   dpi=300, bbox_inches='tight')
        # plt.show()  # Comment out for headless environment
        plt.close()
    
    # Create a summary plot showing progression across checkpoints for each model type
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    model_types = ['awesome', 'brave', 'confident', 'near']
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        # Collect data across checkpoints for this model type
        checkpoints_with_data = []
        pass_at_1_data = []
        pass_at_4_data = []
        pass_at_16_data = []
        
        for checkpoint in sorted(available_checkpoints):
            if model_type in trained_models[checkpoint]:
                checkpoints_with_data.append(checkpoint)
                metrics = trained_models[checkpoint][model_type]
                pass_at_1_data.append(metrics['pass@1'])
                pass_at_4_data.append(metrics['pass@4'])
                pass_at_16_data.append(metrics['pass@16'])
        
        if checkpoints_with_data:
            # Use consistent colors for each model type and add data mix info to title
            model_color = color_mapping[model_type]
            ax.plot(checkpoints_with_data, pass_at_1_data, marker='o', label='pass@1', 
                   linewidth=2, color=model_color)
            ax.plot(checkpoints_with_data, pass_at_4_data, marker='s', label='pass@4', 
                   linewidth=2, color=model_color, alpha=0.8)
            ax.plot(checkpoints_with_data, pass_at_16_data, marker='^', label='pass@16', 
                   linewidth=2, color=model_color, alpha=0.6)
            
            ax.set_xlabel('Checkpoint Step', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            
            # Add data mix info to title
            mix_info = data_mix_info.get(model_type, '')
            title = f'{model_type.title()} Performance Progression\n{mix_info}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mmlu_flan_performance_progression.png', 
               dpi=300, bbox_inches='tight')
    # plt.show()  # Comment out for headless environment
    plt.close()
    
    # Create a comparison table
    create_performance_table(trained_models, opensource_models, output_dir)

def create_performance_table(trained_models, opensource_models, output_dir):
    """Create a performance comparison table"""
    # Create a comprehensive table for key checkpoint 7343
    checkpoint = 7343
    if checkpoint in trained_models:
        data = []
        
        # Add trained models
        for model_type in ['awesome', 'brave', 'confident', 'near']:
            if model_type in trained_models[checkpoint]:
                metrics = trained_models[checkpoint][model_type]
                data.append({
                    'Model': f'{model_type}_{checkpoint}',
                    'Type': 'Trained',
                    'pass@1': f"{metrics['pass@1']:.4f}",
                    'pass@2': f"{metrics['pass@2']:.4f}",
                    'pass@4': f"{metrics['pass@4']:.4f}",
                    'pass@8': f"{metrics['pass@8']:.4f}",
                    'pass@16': f"{metrics['pass@16']:.4f}"
                })
        
        # Add open-source baselines
        baseline_models = ['Llama-3.2-1B', 'Llama-3.2-3B', 'Mistral-7B-v0.3', 'Qwen2.5-1.5B', 'Qwen2.5-3B']
        for baseline in baseline_models:
            if baseline in opensource_models:
                metrics = opensource_models[baseline]
                data.append({
                    'Model': baseline,
                    'Type': 'Open-Source',
                    'pass@1': f"{metrics['pass@1']:.4f}",
                    'pass@2': f"{metrics['pass@2']:.4f}",
                    'pass@4': f"{metrics['pass@4']:.4f}",
                    'pass@8': f"{metrics['pass@8']:.4f}",
                    'pass@16': f"{metrics['pass@16']:.4f}"
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(f'{output_dir}/performance_table_checkpoint_{checkpoint}.csv', index=False)
        print(f"\nPerformance Table for Checkpoint {checkpoint}:")
        print(df.to_string(index=False))

def main():
    # Parse the data
    print("Parsing performance data from pass.out file...")
    models_data = parse_pass_out_file('/mnt/weka/home/haolong.jia/eval/runs/pass.out')
    print(f"Found {len(models_data)} models")
    
    # Categorize models
    trained_models, opensource_models = categorize_models(models_data)
    print(f"Found {len(trained_models)} checkpoint groups and {len(opensource_models)} open-source models")
    
    # Create plots
    print("Creating performance plots...")
    create_performance_plots(trained_models, opensource_models)
    
    print("All plots have been created and saved!")

if __name__ == "__main__":
    main()
