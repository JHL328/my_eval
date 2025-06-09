import argparse
import os
import json
import ast
import re
import traceback
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter, defaultdict
from vllm import LLM, SamplingParams
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.eval import untrusted_check, PASS, FAIL
from evalplus.gen.util import trusted_exec  # Use official trusted_exec for expected outputs
import numpy as np
from tqdm import tqdm
import csv

# define stop words
EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#"
]

def get_stop_words(dataset="humaneval"):
    """get stop words list"""
    stop_words = EOS.copy()
    
    # add specific stop words according to dataset
    if dataset.lower() == "humaneval":
        stop_words += ["\ndef", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    elif dataset.lower() == "mbpp":
        stop_words += ['\n"""', "\nassert"]
    
    return stop_words

def build_prompt(prompt):
    """Base模型直接返回原始prompt"""
    return prompt

def syntax_check(code, verbose=False):
    """检查代码语法是否正确"""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False

def remove_unindented_lines(code, protect_before, exceptions, trim_tails):
    """移除不合适缩进的行"""
    lines = code.splitlines()
    cut_idx = []
    cut_enabled = False
    for i, line in enumerate(lines):
        if not cut_enabled and line.startswith(protect_before):
            cut_enabled = True
            continue
        if line.strip() == "":
            continue
        if any(line.startswith(e) for e in exceptions):
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            cut_idx.append(i)

        if any(line.rstrip().startswith(t) for t in trim_tails):
            # cut off everything behind
            cut_idx.extend(list(range(i, len(lines))))
            break

    return "\n".join([line for i, line in enumerate(lines) if i not in cut_idx])

def to_four_space_indents(old_code):
    """转换为4空格缩进"""
    new_code = ""
    for line in old_code.splitlines():
        lspace = len(line) - len(line.lstrip())
        if lspace == 3:
            new_code += " "
        new_code += line + "\n"
    return new_code

def sanitize_code(old_code: str, entry_point: str, rm_prefix_lines: Optional[str] = None, eofs: List = None):
    """清理LLM生成的代码"""
    new_code = old_code
    if rm_prefix_lines is not None:
        new_code = "\n".join([
            line for line in old_code.splitlines() 
            if not line.startswith(rm_prefix_lines)
        ])

    new_code = "\n" + new_code
    def_left = "def " + entry_point

    # 处理markdown格式（如果存在）
    new_code = new_code.replace("\n```python\n", "\n```\n")
    for chunk in new_code.split("\n```\n"):
        if def_left in chunk:
            new_code = chunk
            break

    chunks = [chunk for chunk in re.split(f"{def_left}\s*\(", new_code)]
    # 找到包含return的函数体
    bodies = [chunk for chunk in chunks[1:] if "    return " in chunk.split("\ndef")[0]]
    def_left = def_left + "("
    new_code = def_left + def_left.join(bodies) if len(bodies) > 0 else ""
    new_code = to_four_space_indents(new_code)

    # 处理EOF标记
    for eof in eofs or []:
        new_code = new_code.split(eof)[0]

    # 移除不合适的行
    new_code = remove_unindented_lines(
        new_code,
        protect_before=def_left,
        exceptions=["def ", "import ", "from "],
        trim_tails=['"""', "if", "print"],
    )
    new_code = chunks[0] + new_code

    # 保留语法正确的函数
    parts = new_code.split("\ndef ")
    includes = [parts[0]]
    for fn in new_code.split("\ndef ")[1:]:
        if (fn.strip().startswith(entry_point + " ") or 
            fn.strip().startswith(entry_point + "(") or 
            syntax_check("\ndef " + fn)):
            includes.append(fn)
    new_code = "\ndef ".join(includes)
    return new_code.strip()

def generate_responses_vllm(model_path, tasks, output_dir, n=1, temperature=0.6, max_tokens=512, tensor_parallel_size=1, suffix="", dataset="humaneval"):
    """使用vllm生成代码响应"""
    # 创建两个子目录：response 和 sanitized_response
    raw_response_dir = os.path.join(output_dir, "response")
    sanitized_response_dir = os.path.join(output_dir, "sanitized_response")
    os.makedirs(raw_response_dir, exist_ok=True)
    os.makedirs(sanitized_response_dir, exist_ok=True)
    
    # 初始化VLLM模型
    llm = LLM(
        model=model_path, 
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        gpu_memory_utilization=0.98,
        max_model_len=max_tokens,
        trust_remote_code=True
    )
    
    # 获取停止词
    stop_words = get_stop_words(dataset=dataset)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature, 
        n=n, 
        max_tokens=max_tokens,
        top_p=0.95 if temperature > 0 else 1.0,
        stop=stop_words
    )
    
    raw_responses = []  # 保存 sanitize 之前的响应
    sanitized_responses = []  # 保存 sanitize 之后的响应
    print(f"Generating responses for {len(tasks)} tasks...")
    print(f"Stop words: {stop_words}")
    
    # 批量处理：准备所有prompts和任务映射
    prompts = []
    task_mapping = []  # 用于映射prompt到对应的task
    
    for i, task in enumerate(tasks):
        prompt = build_prompt(task["prompt"])
        prompts.append(prompt)
        task_mapping.append(task)
    
    print(f"Processing {len(prompts)} prompts in batch...")
    
    try:
        # 批量生成所有responses
        batch_outputs = llm.generate(prompts, sampling_params)
        
        # 处理生成结果
        for i, (task, outputs) in enumerate(zip(task_mapping, batch_outputs)):
            print(f"Processing results for task {i+1}/{len(tasks)}: {task['task_id']}")
            
            if len(outputs.outputs) == 0:
                print(f"Warning: No outputs generated for {task['task_id']}, creating placeholder")
                # 为没有输出的任务创建占位符
                for j in range(n):
                    placeholder_code = task["prompt"] + "\n    pass"
                    raw_responses.append({
                        "task_id": task["task_id"],
                        "completion": placeholder_code,
                        "_identifier": f"{task['task_id']}_{j}"
                    })
                    sanitized_responses.append({
                        "task_id": task["task_id"],
                        "completion": placeholder_code,
                        "_identifier": f"{task['task_id']}_{j}"
                    })
                continue
            
            for j, completion in enumerate(outputs.outputs):
                try:
                    # 原始生成的代码
                    raw_completion = completion.text.replace("\t", "    ")
                    
                    # 构建完整的函数定义：原始prompt + 生成的completion
                    full_code = task["prompt"] + raw_completion
                    
                    # 清理代码
                    try:
                        sanitized_completion = sanitize_code(
                            full_code, 
                            task["entry_point"],
                            eofs=stop_words
                        )
                    except Exception as e:
                        print(f"Sanitization failed for {task['task_id']} sample {j}: {e}")
                        # 如果sanitize失败，至少保证有完整的函数定义
                        sanitized_completion = full_code
                    
                    # 保存原始响应（sanitize之前）
                    raw_responses.append({
                        "task_id": task["task_id"],
                        "completion": full_code,
                        "_identifier": f"{task['task_id']}_{j}"
                    })
                    
                    # 保存净化后的响应（sanitize之后）
                    sanitized_responses.append({
                        "task_id": task["task_id"],
                        "completion": sanitized_completion,
                        "_identifier": f"{task['task_id']}_{j}"
                    })
                    
                except Exception as e:
                    print(f"Error processing completion {j} for {task['task_id']}: {e}")
                    # 为失败的completion创建占位符
                    placeholder_code = task["prompt"] + "\n    pass"
                    raw_responses.append({
                        "task_id": task["task_id"],
                        "completion": placeholder_code,
                        "_identifier": f"{task['task_id']}_{j}"
                    })
                    sanitized_responses.append({
                        "task_id": task["task_id"],
                        "completion": placeholder_code,
                        "_identifier": f"{task['task_id']}_{j}"
                    })
    
    except Exception as e:
        print(f"Batch generation failed: {e}")
        print("Falling back to individual task processing...")
        
        # 如果批量处理失败，回退到逐个处理
        for i, task in enumerate(tasks):
            print(f"Processing task {i+1}/{len(tasks)}: {task['task_id']}")
            
            try:
                prompt = build_prompt(task["prompt"])
                outputs = llm.generate([prompt], sampling_params)
                
                for j, completion in enumerate(outputs[0].outputs):
                    try:
                        raw_completion = completion.text.replace("\t", "    ")
                        full_code = task["prompt"] + raw_completion
                        
                        sanitized_completion = sanitize_code(
                            full_code, 
                            task["entry_point"],
                            eofs=stop_words
                        )
                        
                        # 保存原始响应
                        raw_responses.append({
                            "task_id": task["task_id"],
                            "completion": full_code,
                            "_identifier": f"{task['task_id']}_{j}"
                        })
                        
                        # 保存净化后的响应
                        sanitized_responses.append({
                            "task_id": task["task_id"],
                            "completion": sanitized_completion,
                            "_identifier": f"{task['task_id']}_{j}"
                        })
                        
                    except Exception as e:
                        print(f"Error processing {task['task_id']} sample {j}: {e}")
                        placeholder_code = task["prompt"] + "\n    pass"
                        raw_responses.append({
                            "task_id": task["task_id"],
                            "completion": placeholder_code,
                            "_identifier": f"{task['task_id']}_{j}"
                        })
                        sanitized_responses.append({
                            "task_id": task["task_id"],
                            "completion": placeholder_code,
                            "_identifier": f"{task['task_id']}_{j}"
                        })
            
            except Exception as e:
                print(f"Failed to generate for task {task['task_id']}: {e}")
                # 为失败的任务创建占位符响应
                for j in range(n):
                    placeholder_code = task["prompt"] + "\n    pass"
                    raw_responses.append({
                        "task_id": task["task_id"],
                        "completion": placeholder_code,
                        "_identifier": f"{task['task_id']}_{j}"
                    })
                    sanitized_responses.append({
                        "task_id": task["task_id"],
                        "completion": placeholder_code,
                        "_identifier": f"{task['task_id']}_{j}"
                    })
    
    # 验证所有任务都有样本
    task_ids_in_responses = set(r["task_id"] for r in sanitized_responses)
    expected_task_ids = set(task["task_id"] for task in tasks)
    missing_tasks = expected_task_ids - task_ids_in_responses
    
    if missing_tasks:
        print(f"Warning: Missing tasks in responses: {missing_tasks}")
        # 为缺失的任务添加占位符响应
        for task_id in missing_tasks:
            task = next(t for t in tasks if t["task_id"] == task_id)
            for j in range(n):
                placeholder_code = task["prompt"] + "\n    pass"
                raw_responses.append({
                    "task_id": task_id,
                    "completion": placeholder_code,
                    "_identifier": f"{task_id}_{j}"
                })
                sanitized_responses.append({
                    "task_id": task_id,
                    "completion": placeholder_code,
                    "_identifier": f"{task_id}_{j}"
                })
    
    # 验证每个任务都有足够的样本
    task_sample_counts = Counter(r["task_id"] for r in sanitized_responses)
    for task_id, count in task_sample_counts.items():
        if count < n:
            print(f"Warning: Task {task_id} has only {count} samples, expected {n}")
            task = next(t for t in tasks if t["task_id"] == task_id)
            # 补充缺失的样本
            for j in range(count, n):
                placeholder_code = task["prompt"] + "\n    pass"
                raw_responses.append({
                    "task_id": task_id,
                    "completion": placeholder_code,
                    "_identifier": f"{task_id}_{j}"
                })
                sanitized_responses.append({
                    "task_id": task_id,
                    "completion": placeholder_code,
                    "_identifier": f"{task_id}_{j}"
                })
    
    # 保存原始响应文件（sanitize之前）
    raw_response_file = os.path.join(raw_response_dir, f"response_{suffix}.jsonl")
    with open(raw_response_file, "w") as f:
        for r in raw_responses:
            # 只保存evalplus需要的字段
            output_sample = {
                "task_id": r["task_id"],
                "completion": r["completion"]
            }
            f.write(json.dumps(output_sample) + "\n")
    
    # 保存净化后的响应文件（sanitize之后）
    sanitized_response_file = os.path.join(sanitized_response_dir, f"response_{suffix}.jsonl")
    with open(sanitized_response_file, "w") as f:
        for r in sanitized_responses:
            # 只保存evalplus需要的字段
            output_sample = {
                "task_id": r["task_id"],
                "completion": r["completion"]
            }
            f.write(json.dumps(output_sample) + "\n")
    
    print(f"Saved {len(raw_responses)} raw responses to {raw_response_file}")
    print(f"Saved {len(sanitized_responses)} sanitized responses to {sanitized_response_file}")
    print(f"Tasks covered: {len(task_ids_in_responses)}/{len(expected_task_ids)}")
    print(f"Average samples per task: {len(sanitized_responses) / len(expected_task_ids):.1f}")
    
    return sanitized_responses  # 返回净化后的响应用于评估

def check_correctness_single(task_id, solution, problem, expected_output, dataset_name, min_time_limit, gt_time_limit_factor, fast_check, base_only_flag):
    """检查单个解决方案的正确性 using untrusted_check"""
    try:
        # Base tests (always run)
        base_status_str, base_details = untrusted_check(
            dataset_name,  # Positional
            solution,      # Positional
            problem["base_input"],  # Positional
            problem["entry_point"], # Positional
            expected=expected_output["base"],
            atol=problem.get("atol", 1e-6),
            ref_time=expected_output["base_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
        base_passed_count = sum(1 for passed in base_details if passed) if base_details else 0
        
        plus_status_str = FAIL # Default for plus status
        plus_passed_count = 0
        plus_total_count = len(problem["plus_input"]) if problem.get("plus_input") else 0

        if not base_only_flag:
            if problem.get("plus_input"):
                plus_status_str, plus_details = untrusted_check(
                    dataset_name,  # Positional
                    solution,      # Positional
                    problem["plus_input"],  # Positional
                    problem["entry_point"], # Positional
                    expected=expected_output["plus"],
                    atol=problem.get("atol", 1e-6),
                    ref_time=expected_output["plus_time"],
                    fast_check=fast_check,
                    min_time_limit=min_time_limit,
                    gt_time_limit_factor=gt_time_limit_factor,
                )
                plus_passed_count = sum(1 for passed in plus_details if passed) if plus_details else 0
            elif not problem.get("plus_input"): # No plus inputs (and not base_only mode)
                 plus_status_str = PASS 
                 plus_passed_count = 0 
        else: # base_only_flag is True, so plus tests are effectively skipped or marked as FAIL
            plus_status_str = FAIL # To ensure it doesn't count towards plus_pass_at_k
            plus_passed_count = 0

        return {
            "task_id": task_id,
            "base_status": base_status_str,
            "plus_status": plus_status_str,
            "base_passed": base_passed_count,
            "base_total": len(problem["base_input"]),
            "plus_passed": plus_passed_count,
            "plus_total": plus_total_count
        }
        
    except Exception as e:
        print(f"Error checking {task_id} with untrusted_check: {e}")
        # traceback.print_exc() # for more details if needed
        return {
            "task_id": task_id,
            "base_status": FAIL,
            "plus_status": FAIL,
            "base_passed": 0,
            "base_total": len(problem["base_input"]),
            "plus_passed": 0,
            "plus_total": len(problem["plus_input"]) if problem.get("plus_input") else 0
        }

def get_expected_outputs(problems, dataset_name="humaneval"):
    """获取期望输出 - 使用官方 trusted_exec"""
    print("Computing expected outputs...")
    expected_outputs = {}
    
    # Handle MBPP's special output_not_none tasks
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    tasks_only_output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS if dataset_name == "mbpp" else []
    
    for task_id, problem in problems.items():
        print(f"Computing expected output for {task_id}")
        
        # 执行标准解决方案获取期望输出
        canonical_solution = problem["prompt"] + problem["canonical_solution"]
        
        # Use official trusted_exec with proper parameters
        base_expected, base_time = trusted_exec(
            canonical_solution,
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none
        )
        
        plus_expected, plus_time = trusted_exec(
            canonical_solution,
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none
        )
        
        expected_outputs[task_id] = {
            "base": base_expected,
            "base_time": base_time,
            "plus": plus_expected,
            "plus_time": plus_time
        }
    
    return expected_outputs

def evaluate_responses(responses, problems, expected_outputs, base_only_flag, dataset_name, min_time_limit, gt_time_limit_factor, fast_check):
    """评估生成的响应"""
    print("Evaluating responses...")
    
    # 按task_id分组响应
    grouped_responses = defaultdict(list)
    for resp in responses:
        grouped_responses[resp["task_id"]].append(resp)
    
    results = []
    total = sum(len(v) for v in grouped_responses.values())
    pbar = tqdm(total=total, desc="Evaluating samples")
    
    for task_id, task_responses in grouped_responses.items():
        problem = problems[task_id]
        expected = expected_outputs[task_id]
        
        for i, resp in enumerate(task_responses):
            result = check_correctness_single(
                task_id, 
                resp["completion"], 
                problem, 
                expected, 
                dataset_name, 
                min_time_limit, 
                gt_time_limit_factor, 
                fast_check, 
                base_only_flag
            )
            result["completion_id"] = i
            result["solution"] = resp["completion"]
            results.append(result)
            pbar.update(1)
    pbar.close()
    
    return results

def calculate_pass_at_k(results, k_values=[1, 8, 16, 32, 64]):
    """calculate pass@k"""
    # devide results by task_id
    grouped_results = defaultdict(list)
    for result in results:
        grouped_results[result["task_id"]].append(result)
    
    def estimate_pass_at_k(n, c, k):
        """estimate pass@k"""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    base_pass_at_k = {}
    plus_pass_at_k = {}
    
    for k in k_values:
        base_correct = []
        plus_correct = []
        total_samples = []
        
        for task_id, task_results in grouped_results.items():
            n = len(task_results)
            base_passed = sum(1 for r in task_results if r["base_status"] == PASS)
            plus_passed = sum(1 for r in task_results if r["base_status"] == PASS and r["plus_status"] == PASS)
            
            total_samples.append(n)
            base_correct.append(base_passed)
            plus_correct.append(plus_passed)
        
        if min(total_samples) >= k:
            base_pass_at_k[f"pass@{k}"] = np.mean([
                estimate_pass_at_k(n, c, k) for n, c in zip(total_samples, base_correct)
            ])
            plus_pass_at_k[f"pass@{k}"] = np.mean([
                estimate_pass_at_k(n, c, k) for n, c in zip(total_samples, plus_correct)
            ])
    
    return base_pass_at_k, plus_pass_at_k

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--base_only", action="store_true", default=False, help="Only evaluate base tests")
    # 切分参数：起止索引
    parser.add_argument("--idx_start", type=int, default=0, help="Start index of tasks (inclusive)")
    parser.add_argument("--idx_end", type=int, default=None, help="End index of tasks (exclusive)")
    # New arguments for untrusted_check
    parser.add_argument("--min_time_limit", type=float, default=1.0, help="Minimum time limit for untrusted_check (seconds)")
    parser.add_argument("--gt_time_limit_factor", type=float, default=4.0, help="Ground truth time limit factor for untrusted_check")
    parser.add_argument("--fast_check", action="store_true", default=True, help="Enable fast_check in untrusted_check (stops at first failure). Default is True, similar to evalplus default behavior without --test-details.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载HumanEval任务
    print(f"Loading {args.dataset.upper()} tasks...")
    if args.dataset == "humaneval":
        problems = get_human_eval_plus()
    else:
        problems = get_mbpp_plus()
    tasks = list(problems.values())
    print(f"Loaded {len(tasks)} tasks")
    print(f"here is the first task prompt: {tasks[0]['prompt']}")

    # === 按索引切分任务 ===
    idx_start = args.idx_start
    idx_end = args.idx_end if args.idx_end is not None else len(tasks)
    tasks = tasks[idx_start:idx_end]
    print(f"Evaluating tasks from idx {idx_start} to {idx_end-1} (total {len(tasks)})")
    suffix = f"{idx_start}_{idx_end}"

    # 2. 使用vllm生成代码
    responses = generate_responses_vllm(
        args.model_path, tasks, args.output_dir,  # 传递主输出目录
        n=args.n_samples, temperature=args.temperature, max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size, suffix=suffix, dataset=args.dataset
    )

    # 3. 获取期望输出
    expected_outputs = get_expected_outputs(problems, dataset_name=args.dataset)
    first_task_id = list(expected_outputs.keys())[0]
    print(f"first task_id: {first_task_id}, expected output: {expected_outputs[first_task_id]}")

    # 4. 评估响应
    evaluation_results = evaluate_responses(
        responses, 
        problems, 
        expected_outputs, 
        args.base_only,
        dataset_name=args.dataset,
        min_time_limit=args.min_time_limit,
        gt_time_limit_factor=args.gt_time_limit_factor,
        fast_check=args.fast_check
    )
    print(f"here is the first evaluation result: {evaluation_results[0]}")
    
    # 5. 计算pass@k指标
    base_pass_at_k, plus_pass_at_k = calculate_pass_at_k(evaluation_results)

    # === 保存 base_result.csv 和 plus_result.csv ===
    # 按task_id分组
    grouped_results = defaultdict(list)
    for result in evaluation_results:
        grouped_results[result["task_id"]].append(result)
    # 确保每个task的completion按completion_id排序
    for v in grouped_results.values():
        v.sort(key=lambda x: x["completion_id"])
    task_ids = sorted(grouped_results.keys())
    n_samples = args.n_samples
    # base_result.csv
    base_matrix = []
    for task_id in task_ids:
        row = [1 if r["base_status"] == PASS else 0 for r in grouped_results[task_id]]
        row += [0] * (n_samples - len(row))
        base_matrix.append(row[:n_samples])
    base_csv_path = os.path.join(args.output_dir, f"base_result_{suffix}.csv")
    with open(base_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id"] + [f"sample_{i}" for i in range(n_samples)])
        for task_id, row in zip(task_ids, base_matrix):
            writer.writerow([task_id] + row)
    # plus_result.csv
    plus_matrix = []
    for task_id in task_ids:
        row = [1 if (r["base_status"] == PASS and r["plus_status"] == PASS) else 0 for r in grouped_results[task_id]]
        row += [0] * (n_samples - len(row))
        plus_matrix.append(row[:n_samples])
    plus_csv_path = os.path.join(args.output_dir, f"plus_result_{suffix}.csv")
    with open(plus_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id"] + [f"sample_{i}" for i in range(n_samples)])
        for task_id, row in zip(task_ids, plus_matrix):
            writer.writerow([task_id] + row)
    # === metrics.txt 只写 robust pass@k ===
    ks = [1, 8, 16, 32, 64]
    metrics_txt_path = os.path.join(args.output_dir, f"metrics_{suffix}.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("Base pass@k (robust):\n")
        for k in ks:
            if f"pass@{k}" in base_pass_at_k:
                f.write(f"pass@{k}: {base_pass_at_k[f'pass@{k}']:.4f}\n")
        f.write("\nPlus pass@k (robust):\n")
        for k in ks:
            if f"pass@{k}" in plus_pass_at_k:
                f.write(f"pass@{k}: {plus_pass_at_k[f'pass@{k}']:.4f}\n")
    print(f"Base/Plus result csv and metrics saved to {args.output_dir}")

    # 6. 保存结果
    results = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "base_pass_at_k": base_pass_at_k,
        "plus_pass_at_k": plus_pass_at_k,
        "evaluation_results": evaluation_results
    }

    # 保存详细结果
    results_file = os.path.join(args.output_dir, f"evaluation_results_{suffix}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # 打印结果
    print("\n" + "="*50)
    print(f"Evaluation Results for {args.model_name} on {args.dataset.upper()}")
    print("="*50)
    
    print(f"\n{args.dataset.upper()} (base tests):")
    for k, v in base_pass_at_k.items():
        print(f"  {k}: {v:.3f}")
    
    if not args.base_only:
        print(f"\n{args.dataset.upper()}+ (base + extra tests):")
        for k, v in plus_pass_at_k.items():
            print(f"  {k}: {v:.3f}")

    # 保存简洁的结果摘要
    summary_file = os.path.join(args.output_dir, f"summary_{suffix}.txt")
    with open(summary_file, "w") as f:
        f.write(f"Dataset: {args.dataset.upper()}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Samples: {args.n_samples}\n")
        f.write(f"Temperature: {args.temperature}\n\n")
        
        f.write(f"{args.dataset.upper()} (base tests):\n")
        for k, v in base_pass_at_k.items():
            f.write(f"  {k}: {v:.3f}\n")
        
        if not args.base_only:
            f.write(f"\n{args.dataset.upper()}+ (base + extra tests):\n")
            for k, v in plus_pass_at_k.items():
                f.write(f"  {k}: {v:.3f}\n")

    print(f"\nDetailed results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 