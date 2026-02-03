"""
This file is the all the model need to evaluated

"""

import os
import time
from collections import deque

class ModelQueue:
    def __init__(self, model_map, output_dir, max_model=10, result_filename="result.json", fail_flag="fail.flag"):
        self.model_map = model_map
        self.output_dir = output_dir
        self.max_model = max_model
        self.result_filename = result_filename
        self.fail_flag = fail_flag
        self.model_queue = deque()
        self.running_models = set()
        self.completed = set()
        self.fail = set()
        self._init_queue()

    def _init_queue(self):
        for model_path, model_name in self.model_map.items():
            model_dir = os.path.join(self.output_dir, model_name)
            result_json = os.path.join(model_dir, self.result_filename)
            fail_flag_path = os.path.join(model_dir, self.fail_flag)
            if os.path.exists(result_json):
                self.completed.add(model_name)
            elif os.path.exists(fail_flag_path):
                self.fail.add(model_name)
            else:
                self.model_queue.append((model_path, model_name))

    def update_finished(self):
        finished = []
        failed = []
        for model_name in list(self.running_models):
            model_dir = os.path.join(self.output_dir, model_name)
            if os.path.exists(os.path.join(model_dir, self.result_filename)):
                self.running_models.remove(model_name)
                self.completed.add(model_name)
                finished.append(model_name)
            elif os.path.exists(os.path.join(model_dir, self.fail_flag)):
                self.running_models.remove(model_name)
                self.fail.add(model_name)
                failed.append(model_name)
        return finished, failed

    def can_submit(self):
        return len(self.running_models) < self.max_model and len(self.model_queue) > 0

    def submit_next(self):
        if self.can_submit():
            model_path, model_name = self.model_queue.popleft()
            self.running_models.add(model_name)
            return model_path, model_name
        return None, None

    def is_active(self):
        return bool(self.model_queue) or bool(self.running_models)

    def wait_for_slot(self, interval=60):
        while not self.can_submit() and self.is_active():
            self.update_finished()
            time.sleep(interval)

    def print_status(self):
        print(f"Completed: {sorted(self.completed)}")
        print(f"Running: {sorted(self.running_models)}")
        print(f"Failed: {sorted(self.fail)}")
        print(f"Queued: {[name for _, name in self.model_queue]}")


def get_model_map_by_type(model_type="base"):
    """
    Get the model map by type
    """
    if model_type == "sft":
        return SFT_MODEL_MAP
    else:
        return Model_map


SFT_MODEL_MAP = {
    ################################################################    
    ########## below are the 1p5B base with chat data models #######
    ################################################################    
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_ministerial_banjo/100135": "mix-bbq-all-sft-chat-Chat_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_ministerial_banjo/114440": "mix-bbq-all-sft-chat-Chat_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_ministerial_banjo/128745": "mix-bbq-all-sft-chat-Chat_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_ministerial_banjo/143051": "mix-bbq-all-sft-chat-Chat_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_with_allturns_diagonal_degree/100135": "mix-bbq-all-sft-chat-with-allturns-Chat_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_with_allturns_diagonal_degree/114440": "mix-bbq-all-sft-chat-with-allturns-Chat_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_with_allturns_diagonal_degree/128745": "mix-bbq-all-sft-chat-with-allturns-Chat_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_with_allturns_diagonal_degree/143051": "mix-bbq-all-sft-chat-with-allturns-Chat_143051",

    ##############################################
    ########## below are the SFT models ##########
    ##############################################
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/math_grateful_refrain/checkpoint-27358": "math_grateful_refrain_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/all_third_sine/checkpoint-27358": "all_third_sine_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/awesome_kilby/checkpoint-27358": "awesome_kilby_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/baseline_crooked_rice/checkpoint-27358": "baseline_crooked_rice_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/brave_noether/checkpoint-27358": "brave_noether_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/code_dialogues_substantial_remoulade/checkpoint-27358": "code_dialogues_substantial_remoulade_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/code_thinking_imperial_shannon/checkpoint-27358": "code_thinking_imperial_shannon_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/confident_booth/checkpoint-27358": "confident_booth_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/courageous_congruence_0/checkpoint-27358": "courageous_congruence_0_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/driven_spectacle/checkpoint-27358": "driven_spectacle_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/electoral_lithography/checkpoint-27358": "electoral_lithography_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/enthusiastic_minimalism_0/checkpoint-27358": "enthusiastic_minimalism_0_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/gullible_aperitif/checkpoint-27358": "gullible_aperitif_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/lonely_cone_0/checkpoint-27358": "lonely_cone_0_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/math_lasting_mannerism/checkpoint-27358": "math_lasting_mannerism_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/mountainous_extension_0/checkpoint-27358": "mountainous_extension_0_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/near_habanera/checkpoint-27358": "near_habanera_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/nvidia_cheerful_gong/checkpoint-27358": "nvidia_cheerful_gong_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/regmix_evasive_gateway/checkpoint-27358": "regmix_evasive_gateway_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/resulting_eggs/checkpoint-27358": "resulting_eggs_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/rolling_inverse_0/checkpoint-27358": "rolling_inverse_0_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/steel_lamb/checkpoint-27358": "steel_lamb_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/weary_alias_0/checkpoint-27358": "weary_alias_0_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/web_double_teenage_monochrome/checkpoint-27358": "web_double_teenage_monochrome_27358",
    # "/mnt/weka/shrd/k2m/haolong.jia/RL-model/sft/web_high_fashionable_multiplication/checkpoint-27358": "web_high_fashionable_multiplication_27358",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen3-1.7B": "Qwen3-1.7B-Instruct",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen3-8B": "Qwen3-8B-Instruct",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen3-1.7B-Base": "Qwen3-1.7B-Base-Chat",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen2.5-1.5B": "Qwen2.5-1.5B-Base-Chat",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
}


Model_map = {
    ################################################################
    ########## below are the 1p5B base with chat data models #######
    ################################################################
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_no_chat_temperate_datatable/100135": "mix-bbq-all-sft-no-chat_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_no_chat_temperate_datatable/114440": "mix-bbq-all-sft-no-chat_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_no_chat_temperate_datatable/128745": "mix-bbq-all-sft-no-chat_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_no_chat_temperate_datatable/143051": "mix-bbq-all-sft-no-chat_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_no_chat_no_think_loose_polynomial/100135": "mix-bbq-all-sft-no-chat-no-think_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_no_chat_no_think_loose_polynomial/114440": "mix-bbq-all-sft-no-chat-no-think_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_no_chat_no_think_loose_polynomial/128745": "mix-bbq-all-sft-no-chat-no-think_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_no_chat_no_think_loose_polynomial/143051": "mix-bbq-all-sft-no-chat-no-think_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_ministerial_banjo/100135": "mix-bbq-all-sft-chat_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_ministerial_banjo/114440": "mix-bbq-all-sft-chat_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_ministerial_banjo/128745": "mix-bbq-all-sft-chat_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_ministerial_banjo/143051": "mix-bbq-all-sft-chat_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_with_allturns_diagonal_degree/100135": "mix-bbq-all-sft-chat-with-allturns_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_with_allturns_diagonal_degree/114440": "mix-bbq-all-sft-chat-with-allturns_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_with_allturns_diagonal_degree/128745": "mix-bbq-all-sft-chat-with-allturns_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_sft_chat_with_allturns_diagonal_degree/143051": "mix-bbq-all-sft-chat-with-allturns_143051",
    


    #########################################################
    ########## below are the ELLM models for midtraining ####
    #########################################################
    # "/mnt/weka/shrd/k2m/runner/ellm/checkpoints/huggingface/checkpoint_0300000": "ellm_checkpoint_0300000",
    "/mnt/weka/shrd/k2m/haolong.jia/xllm/checkpoint/k2mobile780M_txt360v2.2_5T_jais64k_bsz16M_seq4k_lr9e-4_cosine_wd0.05_rope128/checkpoint_0300000": "xllm_wd0.05_rope128_0300000",

    #########################################################################
    ########## below are the 1p5b December models， new tokenizers ##########
    #########################################################################
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_nltk_booster_afraid_calculator/100135": "nltk_booster_afraid_calculator_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_nltk_booster_afraid_calculator/114440": "nltk_booster_afraid_calculator_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_nltk_booster_afraid_calculator/128745": "nltk_booster_afraid_calculator_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_nltk_booster_afraid_calculator/143051": "nltk_booster_afraid_calculator_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_solutions_with_reasoning_alternating_tomato/100135": "code_solutions_with_reasoning_alternating_tomato_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_solutions_with_reasoning_alternating_tomato/114440": "code_solutions_with_reasoning_alternating_tomato_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_solutions_with_reasoning_alternating_tomato/128745": "code_solutions_with_reasoning_alternating_tomato_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_solutions_with_reasoning_alternating_tomato/143051": "code_solutions_with_reasoning_alternating_tomato_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_solutions_no_reasoning_buoyant_sauce/100135": "code_solutions_no_reasoning_buoyant_sauce_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_solutions_no_reasoning_buoyant_sauce/114440": "code_solutions_no_reasoning_buoyant_sauce_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_solutions_no_reasoning_buoyant_sauce/128745": "code_solutions_no_reasoning_buoyant_sauce_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_solutions_no_reasoning_buoyant_sauce/143051": "code_solutions_no_reasoning_buoyant_sauce_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_mega_code_sensitive_cpu/100135": "mix-mega-code_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_mega_code_sensitive_cpu/114440": "mix-mega-code_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_mega_code_sensitive_cpu/128745": "mix-mega-code_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_mega_code_sensitive_cpu/143051": "mix-mega-code_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_prompts_all_knobs_weary_artificialneuron/100135": "mix-code-prompts-knobs_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_prompts_all_knobs_weary_artificialneuron/114440": "mix-code-prompts-knobs_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_prompts_all_knobs_weary_artificialneuron/128745": "mix-code-prompts-knobs_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_prompts_all_knobs_weary_artificialneuron/143051": "mix-code-prompts-knobs_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_vibe_test_nebulous_cookie/100135": "mix-vibe-test_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_vibe_test_nebulous_cookie/114440": "mix-vibe-test_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_vibe_test_nebulous_cookie/128745": "mix-vibe-test_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_vibe_test_nebulous_cookie/143051": "mix-vibe-test_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_bbq_all_legislative_html/100135": "mix-bbq-all-legislative-html_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_bbq_all_legislative_html/114440": "mix-bbq-all-legislative-html_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_bbq_all_legislative_html/128745": "mix-bbq-all-legislative-html_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_bbq_all_legislative_html/143051": "mix-bbq-all-legislative-html_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_math_supposed_calligraphy/100135": "mix-bbq-math-supposed-calligraphy_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_math_supposed_calligraphy/114440": "mix-bbq-math-supposed-calligraphy_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_math_supposed_calligraphy/128745": "mix-bbq-math-supposed-calligraphy_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_math_supposed_calligraphy/143051": "mix-bbq-math-supposed-calligraphy_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_sft_lacking_butterkase/100135": "mix-bbq-sft-lacking-butterkase_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_sft_lacking_butterkase/114440": "mix-bbq-sft-lacking-butterkase_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_sft_lacking_butterkase/128745": "mix-bbq-sft-lacking-butterkase_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/txt360-ablations/ckpt/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_sft_lacking_butterkase/143051": "mix-bbq-sft-lacking-butterkase_143051",
    
    ################################################################################
    ########## below are the 1p5b November models, only last 4 checkpoints ##########
    ################################################################################
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_dialogues_substantial_remoulade/100135": "code_dialogues_substantial_remoulade_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_dialogues_substantial_remoulade/114440": "code_dialogues_substantial_remoulade_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_dialogues_substantial_remoulade/128745": "code_dialogues_substantial_remoulade_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_dialogues_substantial_remoulade/143051": "code_dialogues_substantial_remoulade_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_thinking_imperial_shannon/100135": "code_thinking_imperial_shannon_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_thinking_imperial_shannon/114440": "code_thinking_imperial_shannon_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_thinking_imperial_shannon/128745": "code_thinking_imperial_shannon_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_code_thinking_imperial_shannon/143051": "code_thinking_imperial_shannon_143051",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_math_grateful_refrain/100135": "math_grateful_refrain_100135",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_math_grateful_refrain/114440": "math_grateful_refrain_114440",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_math_grateful_refrain/128745": "math_grateful_refrain_128745",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_math_grateful_refrain/143051": "math_grateful_refrain_143051",
    ###############################################
    ########## below are the 7b base models #######
    ###############################################
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_all_fuchsia_ipaddress/76488": "all_fuchsia_ipaddress_76488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_all_fuchsia_ipaddress/82862": "all_fuchsia_ipaddress_82862",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_all_fuchsia_ipaddress/121106": "all_fuchsia_ipaddress_121106",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_all_fuchsia_ipaddress/127480": "all_fuchsia_ipaddress_127480",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_all_fuchsia_ipaddress/152976": "all_fuchsia_ipaddress_152976",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_all_fuchsia_ipaddress/159350": "all_fuchsia_ipaddress_159350",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_all_fuchsia_ipaddress/165724": "all_fuchsia_ipaddress_165724",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_all_fuchsia_ipaddress/165856": "all_fuchsia_ipaddress_165856",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_baseline_congruent_cocoa/76488": "baseline_congruent_cocoa_76488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_baseline_congruent_cocoa/82862": "baseline_congruent_cocoa_82862",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_baseline_congruent_cocoa/121106": "baseline_congruent_cocoa_121106",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_baseline_congruent_cocoa/127480": "baseline_congruent_cocoa_127480",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_baseline_congruent_cocoa/152976": "baseline_congruent_cocoa_152976",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_baseline_congruent_cocoa/159350": "baseline_congruent_cocoa_159350",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_baseline_congruent_cocoa/165724": "baseline_congruent_cocoa_165724",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_baseline_congruent_cocoa/165856": "baseline_congruent_cocoa_165856",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_math_acrylic_beethoven/76488": "math_acrylic_beethoven_76488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_math_acrylic_beethoven/82862": "math_acrylic_beethoven_82862",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_math_acrylic_beethoven/121106": "math_acrylic_beethoven_121106",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_math_acrylic_beethoven/127480": "math_acrylic_beethoven_127480",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_math_acrylic_beethoven/152976": "math_acrylic_beethoven_152976",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_math_acrylic_beethoven/159350": "math_acrylic_beethoven_159350",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_math_acrylic_beethoven/165724": "math_acrylic_beethoven_165724",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_math_acrylic_beethoven/165856": "math_acrylic_beethoven_165856",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_nvidia_cynical_hydrogenfuel/76488": "nvidia_cynical_hydrogenfuel_76488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_nvidia_cynical_hydrogenfuel/82862": "nvidia_cynical_hydrogenfuel_82862",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_nvidia_cynical_hydrogenfuel/121106": "nvidia_cynical_hydrogenfuel_121106",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_nvidia_cynical_hydrogenfuel/127480": "nvidia_cynical_hydrogenfuel_127480",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_nvidia_cynical_hydrogenfuel/152976": "nvidia_cynical_hydrogenfuel_152976",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_nvidia_cynical_hydrogenfuel/159350": "nvidia_cynical_hydrogenfuel_159350",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_nvidia_cynical_hydrogenfuel/165724": "nvidia_cynical_hydrogenfuel_165724",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_nvidia_cynical_hydrogenfuel/165856": "nvidia_cynical_hydrogenfuel_165856",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_regmix_holistic_plane/76488": "regmix_holistic_plane_76488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_regmix_holistic_plane/82862": "regmix_holistic_plane_82862",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_regmix_holistic_plane/121106": "regmix_holistic_plane_121106",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_regmix_holistic_plane/127480": "regmix_holistic_plane_127480",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_regmix_holistic_plane/152976": "regmix_holistic_plane_152976",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_regmix_holistic_plane/159350": "regmix_holistic_plane_159350",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_regmix_holistic_plane/165724": "regmix_holistic_plane_165724",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_7B/tokenmix_ablation_7B_mix_regmix_holistic_plane/165856": "regmix_holistic_plane_165856",

    ############################################################################################
    ######## below are Octorber's models for new mixes(1p5B and only last 4 checkpoints) #######
    ############################################################################################
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_all_third_sine/124899": "all_third_sine_124899",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_all_third_sine/132246": "all_third_sine_132246",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_all_third_sine/139593": "all_third_sine_139593",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_all_third_sine/143051": "all_third_sine_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_baseline_crooked_rice/124899": "baseline_crooked_rice_124899",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_baseline_crooked_rice/132246": "baseline_crooked_rice_132246",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_baseline_crooked_rice/139593": "baseline_crooked_rice_139593",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_baseline_crooked_rice/143051": "baseline_crooked_rice_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_math_lasting_mannerism/124899": "math_lasting_mannerism_124899",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_math_lasting_mannerism/132246": "math_lasting_mannerism_132246",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_math_lasting_mannerism/139593": "math_lasting_mannerism_139593",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_math_lasting_mannerism/143051": "math_lasting_mannerism_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_nvidia_cheerful_gong/124899": "nvidia_cheerful_gong_124899",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_nvidia_cheerful_gong/132246": "nvidia_cheerful_gong_132246",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_nvidia_cheerful_gong/139593": "nvidia_cheerful_gong_139593",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_nvidia_cheerful_gong/143051": "nvidia_cheerful_gong_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_regmix_evasive_gateway/124899": "regmix_evasive_gateway_124899",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_regmix_evasive_gateway/132246": "regmix_evasive_gateway_132246",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_regmix_evasive_gateway/139593": "regmix_evasive_gateway_139593",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_regmix_evasive_gateway/143051": "regmix_evasive_gateway_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_web_double_teenage_monochrome/124899": "web_double_teenage_monochrome_124899",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_web_double_teenage_monochrome/132246": "web_double_teenage_monochrome_132246",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_web_double_teenage_monochrome/139593": "web_double_teenage_monochrome_139593",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_web_double_teenage_monochrome/143051": "web_double_teenage_monochrome_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_web_high_fashionable_multiplication/124899": "web_high_fashionable_multiplication_124899",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_web_high_fashionable_multiplication/132246": "web_high_fashionable_multiplication_132246",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_web_high_fashionable_multiplication/139593": "web_high_fashionable_multiplication_139593",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_1p5B/tokenmix_ablation_1p5B_mix_web_high_fashionable_multiplication/143051": "web_high_fashionable_multiplication_143051",
    ########################################################
    ######## below are the rewrite based models ##########
    #######################################################
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_courageous_congruence_0/courageous_congruence_0_73430": "courageous_congruence_0_73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_courageous_congruence_0/courageous_congruence_0_143051": "courageous_congruence_0_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_enthusiastic_minimalism_0/enthusiastic_minimalism_0_143051": "enthusiastic_minimalism_0_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_enthusiastic_minimalism_0/enthusiastic_minimalism_0_73430": "enthusiastic_minimalism_0_73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_mountainous_extension_0/mountainous_extension_0_73430": "mountainous_extension_0_73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_mountainous_extension_0/mountainous_extension_0_143051": "mountainous_extension_0_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_rolling_inverse_0/rolling_inverse_0_73430": "rolling_inverse_0_73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_rolling_inverse_0/rolling_inverse_0_143051": "rolling_inverse_0_143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_weary_alias_0/weary_alias_0_73430": "weary_alias_0_73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-rewrite/tokenmix_ablation_weary_alias_0/weary_alias_0_143051": "weary_alias_0_143051",

    # # ########################################################
    # # ######## below are the SmolLM3-3B Base series ##########
    # # # ########################################################
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-40000": "stage1-step-40000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-80000": "stage1-step-80000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-120000": "stage1-step-120000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-160000": "stage1-step-160000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-200000": "stage1-step-200000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-240000": "stage1-step-240000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-280000": "stage1-step-280000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-320000": "stage1-step-320000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-360000": "stage1-step-360000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-400000": "stage1-step-400000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-440000": "stage1-step-440000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-480000": "stage1-step-480000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-520000": "stage1-step-520000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-560000": "stage1-step-560000",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM3-3B-Base-series/stage1-step-600000": "stage1-step-600000",

    # # # #######################################################################
    # # # ######### below are the final 587bins with 600B tokens models #########
    # # # #######################################################################
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_7343": "lonely_cone_0_7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_14686": "lonely_cone_0_14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_22029": "lonely_cone_0_22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_29372": "lonely_cone_0_29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_36715": "lonely_cone_0_36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_44058": "lonely_cone_0_44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_51401": "lonely_cone_0_51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_58744": "lonely_cone_0_58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_66087": "lonely_cone_0_66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_73430": "lonely_cone_0_73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_80773": "lonely_cone_0_80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_88116": "lonely_cone_0_88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_95459": "lonely_cone_0_95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_102802": "lonely_cone_0_102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_110145": "lonely_cone_0_110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_117488": "lonely_cone_0_117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_124831": "lonely_cone_0_124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_132174": "lonely_cone_0_132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_139517": "lonely_cone_0_139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/tokenmix-checkpoints-588bins-2-17a/tokenmix_ablation_lonely_cone_0/lonely_cone_0_143051": "lonely_cone_0_143051",
    # # #######################################################
    # # ########## below are the final 587bins models ##########
    # # #######################################################
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_final4/tokenmix_ablation_usable_model_0/usable_model_0_71525": "usable_model_0_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_final4/tokenmix_ablation_solitary_instruction_1/solitary_instruction_1_71525": "solitary_instruction_1_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_final4/tokenmix_ablation_adorable_axis_2/adorable_axis_2_71525": "adorable_axis_2_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_final4/tokenmix_ablation_cooperative_matrix_3/cooperative_matrix_3_71525": "cooperative_matrix_3_71525",
    # # #####################################################
    # # ######## below are the iter2 587bins models #########
    # # #####################################################
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_abstracted_wozniak_1/abstracted_wozniak_1_71525": "abstracted_wozniak_1_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_adapting_legato_24/adapting_legato_24_71525": "adapting_legato_24_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_admiring_postmodernism_3/admiring_postmodernism_3_71525": "admiring_postmodernism_3_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_adoring_ratio_8/adoring_ratio_8_71525": "adoring_ratio_8_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_ashamed_havarti_12/ashamed_havarti_12_71525": "ashamed_havarti_12_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_certain_pattern_7/certain_pattern_7_71525": "certain_pattern_7_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_circular_inkstone_16/circular_inkstone_16_71525": "circular_inkstone_16_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_classical_eggs_32/classical_eggs_32_71525": "classical_eggs_32_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_colorful_waffle_22/colorful_waffle_22_71525": "colorful_waffle_22_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_courageous_monoid_6/courageous_monoid_6_71525": "courageous_monoid_6_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_dreadful_os_14/dreadful_os_14_71525": "dreadful_os_14_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_eminent_abscissa_11/eminent_abscissa_11_71525": "eminent_abscissa_11_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_equilateral_opera_10/equilateral_opera_10_71525": "equilateral_opera_10_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_futuristic_cube_13/futuristic_cube_13_71525": "futuristic_cube_13_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_generative_icecream_20/generative_icecream_20_71525": "generative_icecream_20_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_goodly_matrix_26/goodly_matrix_26_71525": "goodly_matrix_26_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_grumpy_c++_33/grumpy_c++_33_71525": "grumpy_c++_33_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_hurt_sandwich_9/hurt_sandwich_9_71525": "hurt_sandwich_9_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_lethargic_alpha_27/lethargic_alpha_27_71525": "lethargic_alpha_27_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_liberal_habanera_25/liberal_habanera_25_71525": "liberal_habanera_25_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_penitent_omelette_30/penitent_omelette_30_71525": "penitent_omelette_30_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_plastic_calligraphy_31/plastic_calligraphy_31_71525": "plastic_calligraphy_31_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_promoted_c++_23/promoted_c++_23_71525": "promoted_c++_23_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_reasonable_grid_15/reasonable_grid_15_71525": "reasonable_grid_15_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_remaining_zucchini_5/remaining_zucchini_5_71525": "remaining_zucchini_5_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_ridiculous_mezzoforte_2/ridiculous_mezzoforte_2_71525": "ridiculous_mezzoforte_2_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_sick_point_0/sick_point_0_71525": "sick_point_0_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_sincere_stew_17/sincere_stew_17_71525": "sincere_stew_17_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_sorry_marimba_18/sorry_marimba_18_71525": "sorry_marimba_18_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_spiritual_calligraphy_21/spiritual_calligraphy_21_71525": "spiritual_calligraphy_21_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_surrounding_mayonnaise_19/surrounding_mayonnaise_19_71525": "surrounding_mayonnaise_19_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_taxonomic_monograph_4/taxonomic_monograph_4_71525": "taxonomic_monograph_4_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_thirsty_opacity_23/thirsty_opacity_23_71525": "thirsty_opacity_23_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_unchanged_freddiemercury_28/unchanged_freddiemercury_28_71525": "unchanged_freddiemercury_28_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins_round2/tokenmix_ablation_zesty_triangle_29/zesty_triangle_29_71525": "zesty_triangle_29_71525",
    ###################################################
    ####### below are the open source models ###########
    ####################################################
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Llama-3.2-3B": "Llama-3.2-3B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/SmolLM2-1.7B": "SmolLM2-1.7B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Llama-3.2-1B": "Llama-3.2-1B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Mistral-7B-v0.3": "Mistral-7B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen2.5-3B": "Qwen2.5-3B",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen3-1.7B-Base": "Qwen3-1.7B-Base",
    "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/Qwen3-4B-Base": "Qwen3-4B-Base",
    ####################################################################
    #### below are the tokenmix ablation models with 587 bins ##########
    ####################################################################
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_abstract_determinant_16/abstract_determinant_16_71525": "abstract_determinant_16_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_accurate_method_12/accurate_method_12_71525": "accurate_method_12_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_alluring_calculator_13/alluring_calculator_13_71525": "alluring_calculator_13_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_appalling_duple_27/appalling_duple_27_71525": "appalling_duple_27_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_billowy_mousse_1/billowy_mousse_1_71525": "billowy_mousse_1_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_booming_bintree_26/booming_bintree_26_71525": "booming_bintree_26_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_brainy_normal_49/brainy_normal_49_71525": "brainy_normal_49_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_breezy_beer_3/breezy_beer_3_71525": "breezy_beer_3_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_cheerful_union_28/cheerful_union_28_71525": "cheerful_union_28_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_chocolate_brushstroke_67/chocolate_brushstroke_67_71525": "chocolate_brushstroke_67_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_classy_fractals_55/classy_fractals_55_71525": "classy_fractals_55_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_close_pretzel_63/close_pretzel_63_71525": "close_pretzel_63_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_complex_gouda_20/complex_gouda_20_71525": "complex_gouda_20_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_complicated_tetrad_60/complicated_tetrad_60_71525": "complicated_tetrad_60_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_coordinated_assembler_48/coordinated_assembler_48_71525": "coordinated_assembler_48_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_covering_polka_37/covering_polka_37_71525": "covering_polka_37_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_criminal_network_40/criminal_network_40_71525": "criminal_network_40_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_dainty_vorticism_47/dainty_vorticism_47_71525": "dainty_vorticism_47_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_dark_peripheral_10/dark_peripheral_10_71525": "dark_peripheral_10_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_decent_syrup_17/decent_syrup_17_71525": "decent_syrup_17_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_disagreeable_cookie_56/disagreeable_cookie_56_71525": "disagreeable_cookie_56_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_equable_resultant_52/equable_resultant_52_71525": "equable_resultant_52_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_fascinated_graffiti_21/fascinated_graffiti_21_71525": "fascinated_graffiti_21_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_futuristic_composition_59/futuristic_composition_59_71525": "futuristic_composition_59_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_goodly_synchromism_18/goodly_synchromism_18_71525": "goodly_synchromism_18_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_graceful_port_33/graceful_port_33_71525": "graceful_port_33_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_grand_canvas_51/grand_canvas_51_71525": "grand_canvas_51_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_harsh_concerto_15/harsh_concerto_15_71525": "harsh_concerto_15_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_holy_papyrus_7/holy_papyrus_7_71525": "holy_papyrus_7_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_imported_foobar_11/imported_foobar_11_71525": "imported_foobar_11_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_informed_aryabhata_46/informed_aryabhata_46_71525": "informed_aryabhata_46_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_integral_hierarchy_30/integral_hierarchy_30_71525": "integral_hierarchy_30_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_interested_wifi_14/interested_wifi_14_71525": "interested_wifi_14_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_involved_fauvism_34/involved_fauvism_34_71525": "involved_fauvism_34_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_last_trumpet_2/last_trumpet_2_71525": "last_trumpet_2_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_linear_ampersand_23/linear_ampersand_23_71525": "linear_ampersand_23_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_marxist_configuration_66/marxist_configuration_66_71525": "marxist_configuration_66_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_meaty_refrain_61/meaty_refrain_61_71525": "meaty_refrain_61_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_merciful_bytecode_9/merciful_bytecode_9_71525": "merciful_bytecode_9_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_novel_sine_64/novel_sine_64_71525": "novel_sine_64_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_obliged_codeline_38/obliged_codeline_38_71525": "obliged_codeline_38_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_occasional_emmentaler_62/occasional_emmentaler_62_71525": "occasional_emmentaler_62_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_oceanic_line_29/oceanic_line_29_71525": "oceanic_line_29_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_potent_deanmartin_4/potent_deanmartin_4_71525": "potent_deanmartin_4_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_psychological_painting_5/psychological_painting_5_71525": "psychological_painting_5_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_purple_syntaxerror_53/purple_syntaxerror_53_71525": "purple_syntaxerror_53_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_rectilinear_firewall_65/rectilinear_firewall_65_71525": "rectilinear_firewall_65_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_representative_cocoa_44/representative_cocoa_44_71525": "representative_cocoa_44_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_righteous_computer_45/righteous_computer_45_71525": "righteous_computer_45_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_rugged_polynomial_35/rugged_polynomial_35_71525": "rugged_polynomial_35_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_satisfying_utility_50/satisfying_utility_50_71525": "satisfying_utility_50_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_scenic_chants_8/scenic_chants_8_71525": "scenic_chants_8_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_scruffy_symbolism_6/scruffy_symbolism_6_71525": "scruffy_symbolism_6_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_sharing_radian_58/sharing_radian_58_71525": "sharing_radian_58_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_smooth_strauss_25/smooth_strauss_25_71525": "smooth_strauss_25_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_14686": "social_candy_0_14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_22029": "social_candy_0_22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_29372": "social_candy_0_29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_36715": "social_candy_0_36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_44058": "social_candy_0_44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_51401": "social_candy_0_51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_58744": "social_candy_0_58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_66087": "social_candy_0_66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_71525": "social_candy_0_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_social_candy_0/social_candy_0_7343": "social_candy_0_7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_sophisticated_singers_42/sophisticated_singers_42_71525": "sophisticated_singers_42_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_substantial_cubism_32/substantial_cubism_32_71525": "substantial_cubism_32_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_synchronic_statistic_43/synchronic_statistic_43_71525": "synchronic_statistic_43_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_touched_sfumato_24/touched_sfumato_24_71525": "touched_sfumato_24_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_unnecessary_codeline_39/unnecessary_codeline_39_71525": "unnecessary_codeline_39_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_untidy_dish_57/untidy_dish_57_71525": "untidy_dish_57_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_venomous_program_54/venomous_program_54_71525": "venomous_program_54_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_vibrant_slider_19/vibrant_slider_19_71525": "vibrant_slider_19_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_vivid_burger_41/vivid_burger_41_71525": "vivid_burger_41_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_weekly_polynomial_36/weekly_polynomial_36_71525": "weekly_polynomial_36_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_weird_eminem_22/weird_eminem_22_71525": "weird_eminem_22_71525",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint_587bins/tokenmix_ablation_willing_violin_31/willing_violin_31_71525": "willing_violin_31_71525",
    # # # #########################################################################
    # # # ######### below are the tokenmix ablation models with low bins ##########
    # # # #########################################################################
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_102802": "t35-m30-g35-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_110145": "t35-m30-g35-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_117488": "t35-m30-g35-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_124831": "t35-m30-g35-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_132174": "t35-m30-g35-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_139517": "t35-m30-g35-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_143051": "t35-m30-g35-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_14686": "t35-m30-g35-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_22029": "t35-m30-g35-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_29372": "t35-m30-g35-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_36715": "t35-m30-g35-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_44058": "t35-m30-g35-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_51401": "t35-m30-g35-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_58744": "t35-m30-g35-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_66087": "t35-m30-g35-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_7343": "t35-m30-g35-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_73430": "t35-m30-g35-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_80773": "t35-m30-g35-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_88116": "t35-m30-g35-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_95459": "t35-m30-g35-95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_102802": "t60-m30-r10-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_110145": "t60-m30-r10-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_117488": "t60-m30-r10-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_124831": "t60-m30-r10-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_132174": "t60-m30-r10-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_139517": "t60-m30-r10-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_143051": "t60-m30-r10-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_14686": "t60-m30-r10-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_22029": "t60-m30-r10-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_29372": "t60-m30-r10-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_36715": "t60-m30-r10-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_44058": "t60-m30-r10-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_51401": "t60-m30-r10-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_58744": "t60-m30-r10-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_66087": "t60-m30-r10-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_7343": "t60-m30-r10-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_73430": "t60-m30-r10-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_80773": "t60-m30-r10-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_88116": "t60-m30-r10-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_95459": "t60-m30-r10-95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_102802": "t70-m30-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_110145": "t70-m30-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_117488": "t70-m30-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_124831": "t70-m30-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_132174": "t70-m30-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_139517": "t70-m30-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_143051": "t70-m30-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_14686": "t70-m30-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_22029": "t70-m30-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_29372": "t70-m30-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_36715": "t70-m30-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_44058": "t70-m30-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_51401": "t70-m30-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_58744": "t70-m30-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_66087": "t70-m30-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_7343": "t70-m30-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_73430": "t70-m30-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_80773": "t70-m30-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_88116": "t70-m30-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_95459": "t70-m30-95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_7343": "t40-m30-o0-r4-p3-a3-g10-ma10-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_14686": "t40-m30-o0-r4-p3-a3-g10-ma10-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_22029": "t40-m30-o0-r4-p3-a3-g10-ma10-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_29372": "t40-m30-o0-r4-p3-a3-g10-ma10-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_36715": "t40-m30-o0-r4-p3-a3-g10-ma10-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_44058": "t40-m30-o0-r4-p3-a3-g10-ma10-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_51401": "t40-m30-o0-r4-p3-a3-g10-ma10-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_58744": "t40-m30-o0-r4-p3-a3-g10-ma10-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_66087": "t40-m30-o0-r4-p3-a3-g10-ma10-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_73430": "t40-m30-o0-r4-p3-a3-g10-ma10-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_80773": "t40-m30-o0-r4-p3-a3-g10-ma10-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_88116": "t40-m30-o0-r4-p3-a3-g10-ma10-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_95459": "t40-m30-o0-r4-p3-a3-g10-ma10-95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_102802": "t40-m30-o0-r4-p3-a3-g10-ma10-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_110145": "t40-m30-o0-r4-p3-a3-g10-ma10-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_117488": "t40-m30-o0-r4-p3-a3-g10-ma10-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_124831": "t40-m30-o0-r4-p3-a3-g10-ma10-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_132174": "t40-m30-o0-r4-p3-a3-g10-ma10-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_139517": "t40-m30-o0-r4-p3-a3-g10-ma10-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_143051": "t40-m30-o0-r4-p3-a3-g10-ma10-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_7343": "t20-m25-o0-r7-p7-a7-g9-ma25-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_14686": "t20-m25-o0-r7-p7-a7-g9-ma25-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_22029": "t20-m25-o0-r7-p7-a7-g9-ma25-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_29372": "t20-m25-o0-r7-p7-a7-g9-ma25-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_36715": "t20-m25-o0-r7-p7-a7-g9-ma25-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_44058": "t20-m25-o0-r7-p7-a7-g9-ma25-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_51401": "t20-m25-o0-r7-p7-a7-g9-ma25-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_58744": "t20-m25-o0-r7-p7-a7-g9-ma25-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_66087": "t20-m25-o0-r7-p7-a7-g9-ma25-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_73430": "t20-m25-o0-r7-p7-a7-g9-ma25-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_80773": "t20-m25-o0-r7-p7-a7-g9-ma25-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_88116": "t20-m25-o0-r7-p7-a7-g9-ma25-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_95459": "t20-m25-o0-r7-p7-a7-g9-ma25-95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_102802": "t20-m25-o0-r7-p7-a7-g9-ma25-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_110145": "t20-m25-o0-r7-p7-a7-g9-ma25-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_117488": "t20-m25-o0-r7-p7-a7-g9-ma25-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_124831": "t20-m25-o0-r7-p7-a7-g9-ma25-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_132174": "t20-m25-o0-r7-p7-a7-g9-ma25-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_139517": "t20-m25-o0-r7-p7-a7-g9-ma25-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_143051": "t20-m25-o0-r7-p7-a7-g9-ma25-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_7343": "t60-m30-o0-g0-r0-p0-ma10-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_14686": "t60-m30-o0-g0-r0-p0-ma10-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_22029": "t60-m30-o0-g0-r0-p0-ma10-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_29372": "t60-m30-o0-g0-r0-p0-ma10-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_36715": "t60-m30-o0-g0-r0-p0-ma10-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_44058": "t60-m30-o0-g0-r0-p0-ma10-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_51401": "t60-m30-o0-g0-r0-p0-ma10-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_58744": "t60-m30-o0-g0-r0-p0-ma10-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_66087": "t60-m30-o0-g0-r0-p0-ma10-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_73430": "t60-m30-o0-g0-r0-p0-ma10-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_80773": "t60-m30-o0-g0-r0-p0-ma10-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_88116": "t60-m30-o0-g0-r0-p0-ma10-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_95459": "t60-m30-o0-g0-r0-p0-ma10-95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_102802": "t60-m30-o0-g0-r0-p0-ma10-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_110145": "t60-m30-o0-g0-r0-p0-ma10-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_117488": "t60-m30-o0-g0-r0-p0-ma10-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_124831": "t60-m30-o0-g0-r0-p0-ma10-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_132174": "t60-m30-o0-g0-r0-p0-ma10-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_139517": "t60-m30-o0-g0-r0-p0-ma10-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_143051": "t60-m30-o0-g0-r0-p0-ma10-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_7343": "t70-m10-o0-g0-r0-p0-ma20-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_14686": "t70-m10-o0-g0-r0-p0-ma20-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_22029": "t70-m10-o0-g0-r0-p0-ma20-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_29372": "t70-m10-o0-g0-r0-p0-ma20-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_36715": "t70-m10-o0-g0-r0-p0-ma20-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_44058": "t70-m10-o0-g0-r0-p0-ma20-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_51401": "t70-m10-o0-g0-r0-p0-ma20-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_58744": "t70-m10-o0-g0-r0-p0-ma20-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_66087": "t70-m10-o0-g0-r0-p0-ma20-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_73430": "t70-m10-o0-g0-r0-p0-ma20-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_80773": "t70-m10-o0-g0-r0-p0-ma20-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_88116": "t70-m10-o0-g0-r0-p0-ma20-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_95459": "t70-m10-o0-g0-r0-p0-ma20-95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_102802": "t70-m10-o0-g0-r0-p0-ma20-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_110145": "t70-m10-o0-g0-r0-p0-ma20-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_117488": "t70-m10-o0-g0-r0-p0-ma20-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_124831": "t70-m10-o0-g0-r0-p0-ma20-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_132174": "t70-m10-o0-g0-r0-p0-ma20-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_139517": "t70-m10-o0-g0-r0-p0-ma20-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_143051": "t70-m10-o0-g0-r0-p0-ma20-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_7343": "t30-m15-o10-r5-p5-g20-ma15-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_14686": "t30-m15-o10-r5-p5-g20-ma15-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_22029": "t30-m15-o10-r5-p5-g20-ma15-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_29372": "t30-m15-o10-r5-p5-g20-ma15-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_36715": "t30-m15-o10-r5-p5-g20-ma15-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_44058": "t30-m15-o10-r5-p5-g20-ma15-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_51401": "t30-m15-o10-r5-p5-g20-ma15-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_58744": "t30-m15-o10-r5-p5-g20-ma15-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_66087": "t30-m15-o10-r5-p5-g20-ma15-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_73430": "t30-m15-o10-r5-p5-g20-ma15-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_80773": "t30-m15-o10-r5-p5-g20-ma15-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_88116": "t30-m15-o10-r5-p5-g20-ma15-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_95459": "t30-m15-o10-r5-p5-g20-ma15-95459",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_102802": "t30-m15-o10-r5-p5-g20-ma15-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_110145": "t30-m15-o10-r5-p5-g20-ma15-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_117488": "t30-m15-o10-r5-p5-g20-ma15-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_124831": "t30-m15-o10-r5-p5-g20-ma15-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_132174": "t30-m15-o10-r5-p5-g20-ma15-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_139517": "t30-m15-o10-r5-p5-g20-ma15-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_143051": "t30-m15-o10-r5-p5-g20-ma15-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_7343": "t50-m30-o20-r0-p0-g0-ma0-7343",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_102802": "t50-m30-o20-r0-p0-g0-ma0-102802",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_110145": "t50-m30-o20-r0-p0-g0-ma0-110145",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_117488": "t50-m30-o20-r0-p0-g0-ma0-117488",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_124831": "t50-m30-o20-r0-p0-g0-ma0-124831",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_132174": "t50-m30-o20-r0-p0-g0-ma0-132174",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_139517": "t50-m30-o20-r0-p0-g0-ma0-139517",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_143051": "t50-m30-o20-r0-p0-g0-ma0-143051",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_14686": "t50-m30-o20-r0-p0-g0-ma0-14686",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_22029": "t50-m30-o20-r0-p0-g0-ma0-22029",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_29372": "t50-m30-o20-r0-p0-g0-ma0-29372",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_36715": "t50-m30-o20-r0-p0-g0-ma0-36715",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_44058": "t50-m30-o20-r0-p0-g0-ma0-44058",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_51401": "t50-m30-o20-r0-p0-g0-ma0-51401",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_58744": "t50-m30-o20-r0-p0-g0-ma0-58744",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_66087": "t50-m30-o20-r0-p0-g0-ma0-66087",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_73430": "t50-m30-o20-r0-p0-g0-ma0-73430",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_80773": "t50-m30-o20-r0-p0-g0-ma0-80773",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_88116": "t50-m30-o20-r0-p0-g0-ma0-88116",
    # "/mnt/weka/shrd/k2m/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_95459": "t50-m30-o20-r0-p0-g0-ma0-95459",
}

