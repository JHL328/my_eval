
SUPPORTED_BENCHMARKS = {
    # "aime24": {
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 2048
    # },
    # "math": {
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 32768
    # },
    # "math500_pass1": {
    #     "n_fewshot": 5,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 8192
    # },
    # "math500_pass64": {
    #     "n_fewshot": 5,
    #     "n_sampling": 64,
    #     "temperature": 0.6,
    #     "top_p": 0.95,
    #     "tokens": 8192
    # },
    # "humaneval": {
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 1024
    # },
    # "gpqa_diamond": {
    #     "n_fewshot": 5,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "gpqa_diamond_pass32": {
    #     "n_fewshot": 5,
    #     "n_sampling": 32,
    #     "temperature": 0.6,
    #     "top_p": 0.9,
    #     "tokens": 4096
    # },
    # "ifeval": {
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "mmlu": {
    #     "n_fewshot": 4,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    "mmlu_flan_cot_fewshot_pass16": {
        "n_fewshot": 4,
        "n_sampling": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "tokens": 4096
    },
    # "mmlu_pro_pass16": {
    #     "n_fewshot": 5,
    #     "n_sampling": 16,
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "tokens": 4096
    # },
    # "bbh_cot_fewshot_pass16":{
    #     "n_fewshot": 3,
    #     "n_sampling": 1,
    #     "temperature": 0.7,
    #     "top_p": 0.95,
    #     "tokens": 4096,
    # },
    # "bbh_cot_fewshot":{
    #     "n_fewshot": 3,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "drop":{
    #     "n_fewshot": 1,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "arc_easy":{
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "arc_challenge":{
    #     "n_fewshot": 25,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "hellaswag":{
    #     "n_fewshot": 10,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "piqa":{
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "winogrande":{
    #     "n_fewshot": 5,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "triviaqa":{
    #     "n_fewshot": 5,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 8192
    # },
    # "nq_open":{
    #     "n_fewshot": 5,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 8192
    # },
    # "agieval":{
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "commonsense_qa":{
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "openbookqa":{
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "social_iqa":{
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "truthfulqa":{
    #     "n_fewshot": 0,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "gsm8k_pass1":{
    #     "n_fewshot": 8,
    #     "n_sampling": 1,
    #     "temperature": 0,
    #     "top_p": 1,
    #     "tokens": 4096
    # },
    # "gsm8k_pass16":{
    #     "n_fewshot": 8,
    #     "n_sampling": 16,
    #     "temperature": 0.6,
    #     "top_p": 0.95,
    #     "tokens": 8192
    # }
}
