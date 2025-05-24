"""
This file is the all the model need to evaluated

"""

Model_list = [
    "/mnt/sharefs/users/haolong.jia/checkpoint/Llama-3.2-3B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Qwen2.5-1.5B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/SmolLM2-1.7B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Llama-3.2-1B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Mistral-7B-v0.3",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Qwen2.5-3B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Qwen3-1.7B-Base",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Qwen3-4B-Base",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_lucid_fermat/lucid_7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_lucid_fermat/lucid_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_lucid_fermat/lucid_22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_lucid_fermat/lucid_29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_lucid_fermat/lucid_35762",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_stoic_heyrovsky/stoic_7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_stoic_heyrovsky/stoic_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_stoic_heyrovsky/stoic_22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_stoic_heyrovsky/stoic_29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_stoic_heyrovsky/stoic_35762",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nervous_yonath/nervous_7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nervous_yonath/nervous_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nervous_yonath/nervous_22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nervous_yonath/nervous_29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nervous_yonath/nervous_35762",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nostalgic_curie/nostalgic_7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nostalgic_curie/nostalgic_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nostalgic_curie/nostalgic_22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nostalgic_curie/nostalgic_29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_nostalgic_curie/nostalgic_35762",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_hopeful_haibt/hopeful_7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_hopeful_haibt/hopeful_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_hopeful_haibt/hopeful_22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_hopeful_haibt/hopeful_29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokencount_ablation_hopeful_haibt/hopeful_35762",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_7343",
    "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/cpt_haibt/ai/hf_format/samples_10010624.0",
    "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/cpt_haibt/ai/hf_format/samples_5005312.0",
]

# model size mapping
Model_sizes = {
    "Llama-3.2-1B": 1.0,
    "Llama-3.2-3B": 3.0,
    "Qwen2.5-1.5B": 1.5,
    "Qwen2.5-3B": 3.0,
    "SmolLM2-1.7B": 1.7,
    "Mistral-7B-v0.3": 7.0,
    "Qwen3-1.7B-Base": 1.7,
    "Qwen3-4B-Base": 4.0,
}

def get_model_size(model_path):
    """
    automatically infer model size from model path
    
    Args:
        model_path: model path
        
    Returns:
        model size (in billion parameters)
    """
    model_name = model_path.split("/")[-1]
    
    # find model size from predefined mapping
    if model_name in Model_sizes:
        return Model_sizes[model_name]
    
    # try to parse model size from model name
    if "1b" in model_name.lower() or "1B" in model_name:
        return 1.0
    elif "3b" in model_name.lower() or "3B" in model_name:
        return 3.0
    elif "7b" in model_name.lower() or "7B" in model_name:
        return 7.0
    elif "1.5b" in model_name.lower() or "1.5B" in model_name:
        return 1.5
    elif "1.7b" in model_name.lower() or "1.7B" in model_name:
        return 1.7
    elif "4b" in model_name.lower() or "4B" in model_name:
        return 4.0
    
    # default value
    return 3.0






