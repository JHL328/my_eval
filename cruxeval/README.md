## Getting Started 

We evaluate the Qwen-32B-instruct on cruxeval dataset.


## Sample Code
```bash
cd cruxeval
# Install 
pip install -r requirements-inference.txt
pip install vllm  # update your vvlm package 

# Scoring the exsited generation using Qwen-32B-instruct
cd evaluation
python evaluate_qwen.py
```

## Run Inference and Evaluation
We evaluate the model under different configurations to measure performance across conditions:
- **Temperatures**: `[0.2, 0.8]`
- **Tasks**: `"input"` prediction and `"output"` prediction
- **CoT (Chain-of-Thought)**: `True` (Qwen-cot) and `False` (Qwen)


```bash
# Step 1: Generate responses with the Qwen-32B-Instruct model
cd inference

# Generate responses in parallel across different configurations
python generation_qwen.py  

# Combine the generated output files
python combine_generations.py  

# Step 2: Evaluate the generated responses
cd ../evaluation
python evaluate_qwen.py