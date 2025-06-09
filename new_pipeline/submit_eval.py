from subprocess import Popen
from optparse import OptionParser
import os
# import numpy as np
import json
import time
from model_dirs import model_dirs
import sys
import uuid
import glob

def parse_args():
    parser = OptionParser()

    parser.add_option("--tasks", type="str", dest="tasks")

    (options, args) = parser.parse_args()

    return options

def create_job_script(scripts_dir, exp_name, logs_dir, command_args):
    script_path = os.path.join(scripts_dir, f'job_{exp_name}.sh')
    # SBATCH --reservation=gpu-exp
    script_content = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --job-name={exp_name}
#SBATCH --time=3:00:00
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH -o {logs_dir}%j_%x.out
#SBATCH -e {logs_dir}%j_%x.err

source /mnt/weka/home/mikhail.yurochkin/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export TRITON_CACHE_DIR="/tmp/triton-cache"

{command_args}
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path

def main(logs_dir = './logs/', results_dir = './results/'):

    
    options = parse_args()
    print(options)
    
        
    tasks = options.tasks
    tasks = tasks.split(',')
    
    try:
        os.makedirs(logs_dir)
    except:
        pass

    try:
        os.makedirs(results_dir)
    except:
        pass

    scripts_dir = './job_scripts/'
    os.makedirs(scripts_dir, exist_ok=True)

    for task in tasks:
        
        if task == 'kk':
            # splits = ['2ppl', '3ppl']
            splits = ['2ppl', '3ppl', '4ppl', '5ppl', '6ppl', '7ppl', '8ppl']
            logs_dir_kk = logs_dir + 'kk/'
            results_dir_kk = results_dir + 'kk/'
            try:
                os.makedirs(logs_dir_kk)
            except:
                pass
            try:
                os.makedirs(results_dir_kk)
            except:
                pass
            for model_name in model_dirs:
                model_tag = '_'.join(model_name.split('/')[-3:])
                results_dir_model = results_dir_kk + model_tag + '/'
                logs_dir_model = logs_dir_kk + model_tag + '/'
                try:
                    os.makedirs(results_dir_model)
                except:
                    pass
                try:
                    os.makedirs(logs_dir_model)
                except:
                    pass
                for split in splits:
                    
                    exp_name = f'kk-{model_tag}-{split}'
            
                    if not os.path.exists(f'{results_dir_model}{split}.csv'):
                        print(exp_name)
                        command_args = f"python evaluate_kk.py --model {model_name} --split {split} --results_dir {results_dir_model}"
                        job_script = create_job_script(scripts_dir, exp_name, logs_dir_model, command_args)
                        Popen(['sbatch', job_script])
                        time.sleep(0.25)
                    else:
                        print('ALREADY DONE', exp_name)
                    

        if task == 'cd':
            bs = 50
            logs_dir_cd = logs_dir + 'cd/'
            results_dir_cd = results_dir + 'cd/'
            try:
                os.makedirs(logs_dir_cd)
            except:
                pass
            try:
                os.makedirs(results_dir_cd)
            except:
                pass
            for model_name in model_dirs:
                model_tag = '_'.join(model_name.split('/')[-3:])
                results_dir_model = results_dir_cd + model_tag + '/'
                logs_dir_model = logs_dir_cd + model_tag + '/'
                try:
                    os.makedirs(results_dir_model)
                except:
                    pass
                idx_start = 0
                while idx_start < 1024:
                    idx_end = idx_start + bs
                    exp_name = f'cd-{model_tag}-{idx_start}_{idx_end}'
                    if not os.path.exists(f'{results_dir_model}{idx_start}-{idx_end}.csv'):
                        print(exp_name)
                        command_args = f"python evaluate_cd.py --model {model_name} --idx_start {idx_start} --idx_end {idx_end} --results_dir {results_dir_model}"
                        job_script = create_job_script(scripts_dir, exp_name, logs_dir_model, command_args)
                        Popen(['sbatch', job_script])
                        time.sleep(0.25)
                    else:
                        print('ALREADY DONE', exp_name)
                    idx_start = idx_end

        if task == 'sum':
            bs = 50
            logs_dir_sum = logs_dir + 'sum/'
            results_dir_sum = results_dir + 'sum/'
            try:
                os.makedirs(logs_dir_sum)
            except:
                pass
            try:
                os.makedirs(results_dir_sum)
            except:
                pass
            for model_name in model_dirs:
                model_tag = '_'.join(model_name.split('/')[-3:])
                results_dir_model = results_dir_sum + model_tag + '/'
                logs_dir_model = logs_dir_sum + model_tag + '/'
                try:
                    os.makedirs(results_dir_model)
                except:
                    pass
                idx_start = 0
                while idx_start < 475:
                    idx_end = idx_start + bs
                    exp_name = f'sum-{model_tag}-{idx_start}_{idx_end}'
                    if not os.path.exists(f'{results_dir_model}{idx_start}-{idx_end}.csv'):
                        print(exp_name)
                        command_args = f"python evaluate_sum.py --model {model_name} --idx_start {idx_start} --idx_end {idx_end} --results_dir {results_dir_model}"
                        job_script = create_job_script(scripts_dir, exp_name, logs_dir_model, command_args)
                        Popen(['sbatch', job_script])
                        time.sleep(0.25)
                    else:
                        print('ALREADY DONE', exp_name)
                    idx_start = idx_end

        if task == 'order':
            logs_dir_order = logs_dir + 'order/'
            results_dir_order = results_dir + 'order/'
            try:
                os.makedirs(logs_dir_order)
            except:
                pass
            try:
                os.makedirs(results_dir_order)
            except:
                pass
            for model_name in model_dirs:
                model_tag = '_'.join(model_name.split('/')[-3:])
                results_dir_model = results_dir_order + model_tag + '/'
                logs_dir_model = logs_dir_order + model_tag + '/'
                try:
                    os.makedirs(results_dir_model)
                except:
                    pass
                try:
                    os.makedirs(logs_dir_model)
                except:
                    pass
                for level in [6, 9, 12, 15, 18, 24, 30]:
                    exp_name = f'order-{model_tag}-{level}'
                    if not os.path.exists(f'{results_dir_model}{level}.csv'):
                        print(exp_name)
                        command_args = f"python evaluate_order.py --model {model_name} --level {level} --results_dir {results_dir_model}"
                        job_script = create_job_script(scripts_dir, exp_name, logs_dir_model, command_args)
                        Popen(['sbatch', job_script])
                        time.sleep(0.25)
                    else:
                        print('ALREADY DONE', exp_name)

        # srun -n 1 --mem 200G --gres gpu:1 --job-name 300_350:cd -o ./logs_eval/%j_%x.out -e ./logs_eval/%j_%x.err python evaluate_data.py --idx_start 300 --idx_end 350 --results_dir ./results_eval_cd_llama_reason/
        
if __name__ == '__main__':
    main()
