# 评估脚本修改计划

## 一、Math-Verify Integration Plan for GSM8K and Math500 Evaluation

### 📋 Overview
Replace current evaluation methods (regex-based for GSM8K, sympy-based for Math500) with math_verify library for more robust mathematical equivalence checking.

### 🎯 Objectives
1. Integrate math_verify into GSM8K evaluation pipeline
2. Create new Math500 evaluation script using math_verify
3. Maintain all existing generation settings (temperature, top_p, few-shot prompts)
4. Keep output format compatibility (result.csv, passk.json)

### 📁 Files to Modify/Create

#### 1. **evaluate_gsm8k.py** (Modify)
- ✅ Already imported `from math_verify import parse, verify`
- Replace `parse_answer()` function logic with math_verify
- Update evaluation logic in `run_single_model_evaluation()`:
  - Parse gold answers using `parse()`
  - Parse model predictions using `parse()`  
  - Use `verify()` for mathematical equivalence checking

#### 2. **evaluate_math500.py** (New File)
Create a simplified evaluation script specifically for Math500:
- Import vLLM for generation (same as GSM8K)
- Use same sampling parameters:
  - temperature=0.6
  - top_p=0.95
  - n_sampling=64
  - max_tokens=4096
- Use 4-shot prompts from examples.py (math500 examples)
- Implement math_verify for answer checking
- Generate compatible output format (sample.jsonl, result.csv)

#### 3. **evaluate_gsm8k.sh** (Modify if needed)
Add support for calling the new evaluate_math500.py when task is math500

### 🔧 Implementation Details

#### Math500 Script Key Components:
1. Load Math500 data from appropriate path
2. Generate 4-shot prompt using examples["math500"] from examples.py
3. Use vLLM with exact same parameters as original math_eval.py
4. Parse answers with math_verify.parse()
5. Verify with math_verify.verify()
6. Calculate pass@k metrics (k=[1,2,4,8,16,32,64])
7. Save outputs in compatible format

### Key Configuration Preservation
- **GSM8K**: 8-shot, n_sampling=16, temp=0.6, top_p=0.95
- **Math500**: 4-shot, n_sampling=64, temp=0.6, top_p=0.95
- Both use same prompt format from original scripts
- Maintain SLURM job settings (GPU, memory, time limits)

### ⚠️ Important Considerations
- Handle edge cases where math_verify might fail to parse
- Fall back to string comparison if parsing fails
- Ensure backward compatibility with existing results
- Keep all generation hyperparameters identical

### ✅ Benefits
- More robust mathematical equivalence checking
- Handles various mathematical notations better
- Reduces false negatives from formatting differences
- Unified evaluation approach for both datasets

## 二、BBH评估脚本修改计划

### 修改目标
修改evaluate_bbh_pass16.py以支持SFT模型的正确评估，解决当前存在的三个主要问题：
1. SFT模型prompt构建格式混乱
2. 需要支持0-shot评估（而非fewshot）
3. 需要自定义system prompt引导输出格式

## 修改方案详细说明

### 1. System Prompt设计
```python
system_prompt = """You are a helpful assistant that solves logical reasoning problems step by step.

When given a problem:
1. Think through the solution systematically
2. Show your reasoning process clearly  
3. End with a clear final answer using the format: "So the answer is [your answer]"

Remember to be precise and logical in your reasoning."""
```

**设计理由**：
- 简洁清晰，不过度约束模型
- 保留"So the answer is"格式，与原有答案提取逻辑兼容
- 强调step-by-step推理，符合BBH任务特点
- 避免特殊标记，减少输出格式的复杂性

### 2. evaluate_bbh_pass16.py具体修改

#### 修改build_prompt函数（第53-59行）✅
```python
def build_prompt(fewshot, example, model_type="base"):
    if model_type == "sft":
        # SFT模型使用0-shot，直接返回input
        return example['input']
    else:
        # Base模型保持原有fewshot格式
        return fewshot + '\n\nQ: ' + example['input'] + '\nA: Let\'s think step by step.'
```

#### 修改主函数中的prompt构建（第93-123行）✅
- 构建prompts时传入model_type参数
- SFT模型应用chat template时包含system prompt
- System prompt引导模型输出格式

### 3. 关键优势
1. **最小化改动**：只需修改prompt构建部分，答案提取逻辑保持不变
2. **兼容性好**：base和sft模型使用相同的答案提取函数
3. **0-shot实现**：SFT模型不使用fewshot examples
4. **格式统一**：输出格式与原有系统兼容

## 实施进度
- [x] 分析当前evaluate_bbh_pass16.py的问题
- [x] 制定修改方案以支持SFT模型0-shot评估
- [x] 设计合适的system prompt
- [x] 实现evaluate_bbh_pass16.py的修改
- [ ] 测试修改后的代码

## 测试计划
1. 选择一个小的BBH任务（如boolean_expressions）测试
2. 对比base和sft模型的输出格式
3. 验证答案提取的准确性
4. 检查pass@k指标计算是否正常

## 注意事项
- 所有修改都需要兼容base和sft两种模型类型
- 保持向后兼容，不影响现有base模型的评估
- 修改完成后需要进行充分测试再应用到其他评估脚本