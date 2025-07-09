import os

def generate_k2_model_map():
    """
    Scans the k2+_midtraining_mix directory and generates a model_map dictionary.
    The result is written to k2_model.py.
    """
    base_dir = "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/k2+_midtraining_mix"
    output_file = "/mnt/weka/home/haolong.jia/eval/RL-eval/new_pipeline/k2/k2_model.py"
    
    model_map = {}

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory not found at {base_dir}")
        return

    # scan all experiment folders
    for experiment_folder in sorted(os.listdir(base_dir)):
        experiment_path = os.path.join(base_dir, experiment_folder)
        
        if not os.path.isdir(experiment_path):
            continue

        # scan all hf_format folders
        hf_format_path = os.path.join(experiment_path, "hf_format")
        
        if not os.path.isdir(hf_format_path):
            continue

        # scan all sample/checkpoint folders
        for checkpoint_folder in sorted(os.listdir(hf_format_path)):
            checkpoint_path = os.path.join(hf_format_path, checkpoint_folder)
            
            if not os.path.isdir(checkpoint_path):
                continue
            
            # generate model name
            name_part1 = experiment_folder.replace(':', '-')
            
            try:
                # extract the number from "samples_3773124.0"
                name_part2 = checkpoint_folder.split('_')[1].split('.')[0]
            except IndexError:
                # skip the folder that does not follow the naming rule
                continue
            
            model_name = f"{name_part1}-{name_part2}"
            
            # fill the dictionary
            model_map[checkpoint_path] = model_name

    # write the result to k2_model.py
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("model_map = {\n")
            # sort the dictionary by key to ensure the output order is stable
            for path, name in sorted(model_map.items()):
                f.write(f'    "{path}": "{name}",\n')
            f.write("}\n")
        print(f"🎉 Successfully generated model map and saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")

if __name__ == "__main__":
    generate_k2_model_map()
