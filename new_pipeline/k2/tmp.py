import pandas as pd
import re

def format_model_name(model_name):
    """Format model name by adding colons between parameter names and values"""
    # Remove trailing spaces and handle the underscore suffix
    model_name = model_name.strip()
    
    # Split by underscore to get base and suffix
    if '_' in model_name:
        base, suffix = model_name.rsplit('_', 1)
    else:
        base = model_name
        suffix = ''
    
    # Split base by '-'
    parts = base.split('-')
    formatted_parts = []
    
    # First part is 'final', keep it as is
    formatted_parts.append(parts[0])
    
    # Known parameter names
    param_names = {'a', 'bp', 'g', 'm', 'mm', 'oc', 'p', 'qa', 'r1', 'r'}
    
    # Process the rest of the parts
    i = 1
    while i < len(parts):
        if parts[i] in param_names and i + 1 < len(parts):
            # This is a parameter name followed by its value
            formatted_parts.append(f"{parts[i]}:{parts[i+1]}")
            i += 2
        else:
            # This shouldn't happen with well-formed names
            formatted_parts.append(parts[i])
            i += 1
    
    # Join with '-' and add suffix
    return f"{'-'.join(formatted_parts)}-{suffix}" if suffix else '-'.join(formatted_parts)

def process_midtraining_results(input_file='/mnt/sharefs/users/haolong.jia/result-k2/midtraining_results.csv', output_file='/mnt/sharefs/users/haolong.jia/result-k2/midtraining_final_results_formatted.csv'):
    """Process the midtraining results CSV file to extract and format final models"""
    
    print(f"📖 Reading {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file, sep='\t')
    
    # Get the actual column names (they might have extra spaces)
    model_col = df.columns[0]
    gsm8k_col = df.columns[1]
    mmlu_col = df.columns[2]
    bbh_col = df.columns[3]
    humaneval_col = df.columns[4]
    
    print(f"📊 Found {len(df)} total models")
    
    # Filter for final models
    final_mask = df[model_col].str.contains('final-', na=False)
    final_df = df[final_mask].copy()
    
    print(f"🎯 Found {len(final_df)} final models")
    
    # Format the model names and clean the data
    results = []
    for idx, row in final_df.iterrows():
        model_name = str(row[model_col]).strip()
        formatted_name = format_model_name(model_name)
        
        results.append({
            'Model Name': formatted_name,
            'gsm8k': float(row[gsm8k_col]),
            'mmlu': float(row[mmlu_col]),
            'bbh': float(row[bbh_col]),
            'humaneval': float(row[humaneval_col])
        })
        
        print(f"  ✓ {model_name} -> {formatted_name}")
    
    # Create formatted DataFrame
    formatted_df = pd.DataFrame(results)
    
    # Ensure column order
    formatted_df = formatted_df[['Model Name', 'gsm8k', 'mmlu', 'bbh', 'humaneval']]
    
    # Save to CSV
    formatted_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Successfully saved to {output_file}")
    print(f"📊 Total final models: {len(formatted_df)}")
    
    print("\n📋 Results preview:")
    print(formatted_df.to_string())
    
    print("\n📈 Performance summary:")
    summary = formatted_df[['gsm8k', 'mmlu', 'bbh', 'humaneval']].describe().round(2)
    print(summary)
    
    return formatted_df

# Run the processing
if __name__ == "__main__":
    # Process the file
    df = process_midtraining_results()
    
    # Optional: Create a sorted version by model name
    df_sorted = df.sort_values('Model Name')
    df_sorted.to_csv('/mnt/sharefs/users/haolong.jia/result-k2/midtraining_final_results_sorted.csv', index=False)
    print("\n📝 Also saved sorted version to midtraining_final_results_sorted.csv")