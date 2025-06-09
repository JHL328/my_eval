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

    def wait_for_slot(self, interval=30):
        while not self.can_submit() and self.is_active():
            self.update_finished()
            time.sleep(interval)

    def print_status(self):
        print(f"Completed: {sorted(self.completed)}")
        print(f"Running: {sorted(self.running_models)}")
        print(f"Failed: {sorted(self.fail)}")
        print(f"Queued: {[name for _, name in self.model_queue]}")


Model_map = {
    "/mnt/sharefs/users/haolong.jia/checkpoint/Llama-3.2-3B": "Llama-3.2-3B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/SmolLM2-1.7B": "SmolLM2-1.7B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Llama-3.2-1B": "Llama-3.2-1B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Mistral-7B-v0.3": "Mistral-7B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Qwen2.5-3B": "Qwen2.5-3B",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Qwen3-1.7B-Base": "Qwen3-1.7B-Base",
    "/mnt/sharefs/users/haolong.jia/checkpoint/Qwen3-4B-Base": "Qwen3-4B-Base",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_102802": "t35-m30-g35-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_110145": "t35-m30-g35-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_117488": "t35-m30-g35-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_124831": "t35-m30-g35-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_132174": "t35-m30-g35-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_139517": "t35-m30-g35-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_143051": "t35-m30-g35-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_14686": "t35-m30-g35-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_22029": "t35-m30-g35-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_29372": "t35-m30-g35-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_36715": "t35-m30-g35-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_44058": "t35-m30-g35-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_51401": "t35-m30-g35-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_58744": "t35-m30-g35-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_66087": "t35-m30-g35-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_7343": "t35-m30-g35-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_73430": "t35-m30-g35-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_80773": "t35-m30-g35-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_88116": "t35-m30-g35-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_awesome_kilby/awesome_kilby_95459": "t35-m30-g35-95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_102802": "t60-m30-r10-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_110145": "t60-m30-r10-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_117488": "t60-m30-r10-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_124831": "t60-m30-r10-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_132174": "t60-m30-r10-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_139517": "t60-m30-r10-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_143051": "t60-m30-r10-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_14686": "t60-m30-r10-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_22029": "t60-m30-r10-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_29372": "t60-m30-r10-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_36715": "t60-m30-r10-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_44058": "t60-m30-r10-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_51401": "t60-m30-r10-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_58744": "t60-m30-r10-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_66087": "t60-m30-r10-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_7343": "t60-m30-r10-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_73430": "t60-m30-r10-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_80773": "t60-m30-r10-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_88116": "t60-m30-r10-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_brave_noether/brave_noether_95459": "t60-m30-r10-95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_102802": "t70-m30-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_110145": "t70-m30-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_117488": "t70-m30-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_124831": "t70-m30-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_132174": "t70-m30-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_139517": "t70-m30-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_143051": "t70-m30-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_14686": "t70-m30-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_22029": "t70-m30-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_29372": "t70-m30-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_36715": "t70-m30-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_44058": "t70-m30-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_51401": "t70-m30-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_58744": "t70-m30-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_66087": "t70-m30-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_7343": "t70-m30-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_73430": "t70-m30-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_80773": "t70-m30-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_88116": "t70-m30-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_confident_booth/confident_booth_95459": "t70-m30-95459",
    "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/cpt_haibt/ai/hf_format/samples_10010624.0": "haibt-10mil-ai",
    "/mnt/sharefs/users/mikhail.yurochkin/checkpoints_to_eval/cpt_haibt/math/hf_format/samples_5005312.0": "haibt-5mil-math",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_7343": "t40-m30-o0-r4-p3-a3-g10-ma10-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_14686": "t40-m30-o0-r4-p3-a3-g10-ma10-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_22029": "t40-m30-o0-r4-p3-a3-g10-ma10-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_29372": "t40-m30-o0-r4-p3-a3-g10-ma10-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_36715": "t40-m30-o0-r4-p3-a3-g10-ma10-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_44058": "t40-m30-o0-r4-p3-a3-g10-ma10-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_51401": "t40-m30-o0-r4-p3-a3-g10-ma10-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_58744": "t40-m30-o0-r4-p3-a3-g10-ma10-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_66087": "t40-m30-o0-r4-p3-a3-g10-ma10-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_73430": "t40-m30-o0-r4-p3-a3-g10-ma10-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_80773": "t40-m30-o0-r4-p3-a3-g10-ma10-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_88116": "t40-m30-o0-r4-p3-a3-g10-ma10-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_95459": "t40-m30-o0-r4-p3-a3-g10-ma10-95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_102802": "t40-m30-o0-r4-p3-a3-g10-ma10-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_110145": "t40-m30-o0-r4-p3-a3-g10-ma10-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_117488": "t40-m30-o0-r4-p3-a3-g10-ma10-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_124831": "t40-m30-o0-r4-p3-a3-g10-ma10-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_132174": "t40-m30-o0-r4-p3-a3-g10-ma10-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_139517": "t40-m30-o0-r4-p3-a3-g10-ma10-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_driven_spectacle/driven_spectacle_143051": "t40-m30-o0-r4-p3-a3-g10-ma10-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_7343": "t20-m25-o0-r7-p7-a7-g9-ma25-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_14686": "t20-m25-o0-r7-p7-a7-g9-ma25-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_22029": "t20-m25-o0-r7-p7-a7-g9-ma25-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_29372": "t20-m25-o0-r7-p7-a7-g9-ma25-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_36715": "t20-m25-o0-r7-p7-a7-g9-ma25-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_44058": "t20-m25-o0-r7-p7-a7-g9-ma25-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_51401": "t20-m25-o0-r7-p7-a7-g9-ma25-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_58744": "t20-m25-o0-r7-p7-a7-g9-ma25-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_66087": "t20-m25-o0-r7-p7-a7-g9-ma25-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_73430": "t20-m25-o0-r7-p7-a7-g9-ma25-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_80773": "t20-m25-o0-r7-p7-a7-g9-ma25-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_88116": "t20-m25-o0-r7-p7-a7-g9-ma25-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_95459": "t20-m25-o0-r7-p7-a7-g9-ma25-95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_102802": "t20-m25-o0-r7-p7-a7-g9-ma25-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_110145": "t20-m25-o0-r7-p7-a7-g9-ma25-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_117488": "t20-m25-o0-r7-p7-a7-g9-ma25-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_124831": "t20-m25-o0-r7-p7-a7-g9-ma25-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_132174": "t20-m25-o0-r7-p7-a7-g9-ma25-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_139517": "t20-m25-o0-r7-p7-a7-g9-ma25-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_steel_lamb/steel_lamb_143051": "t20-m25-o0-r7-p7-a7-g9-ma25-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_7343": "t60-m30-o0-g0-r0-p0-ma10-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_14686": "t60-m30-o0-g0-r0-p0-ma10-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_22029": "t60-m30-o0-g0-r0-p0-ma10-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_29372": "t60-m30-o0-g0-r0-p0-ma10-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_36715": "t60-m30-o0-g0-r0-p0-ma10-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_44058": "t60-m30-o0-g0-r0-p0-ma10-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_51401": "t60-m30-o0-g0-r0-p0-ma10-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_58744": "t60-m30-o0-g0-r0-p0-ma10-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_66087": "t60-m30-o0-g0-r0-p0-ma10-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_73430": "t60-m30-o0-g0-r0-p0-ma10-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_80773": "t60-m30-o0-g0-r0-p0-ma10-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_88116": "t60-m30-o0-g0-r0-p0-ma10-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_95459": "t60-m30-o0-g0-r0-p0-ma10-95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_102802": "t60-m30-o0-g0-r0-p0-ma10-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_110145": "t60-m30-o0-g0-r0-p0-ma10-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_117488": "t60-m30-o0-g0-r0-p0-ma10-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_124831": "t60-m30-o0-g0-r0-p0-ma10-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_132174": "t60-m30-o0-g0-r0-p0-ma10-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_139517": "t60-m30-o0-g0-r0-p0-ma10-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_gullible_aperitif/gullible_aperitif_143051": "t60-m30-o0-g0-r0-p0-ma10-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_7343": "t70-m10-o0-g0-r0-p0-ma20-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_14686": "t70-m10-o0-g0-r0-p0-ma20-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_22029": "t70-m10-o0-g0-r0-p0-ma20-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_29372": "t70-m10-o0-g0-r0-p0-ma20-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_36715": "t70-m10-o0-g0-r0-p0-ma20-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_44058": "t70-m10-o0-g0-r0-p0-ma20-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_51401": "t70-m10-o0-g0-r0-p0-ma20-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_58744": "t70-m10-o0-g0-r0-p0-ma20-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_66087": "t70-m10-o0-g0-r0-p0-ma20-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_73430": "t70-m10-o0-g0-r0-p0-ma20-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_80773": "t70-m10-o0-g0-r0-p0-ma20-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_88116": "t70-m10-o0-g0-r0-p0-ma20-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_95459": "t70-m10-o0-g0-r0-p0-ma20-95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_102802": "t70-m10-o0-g0-r0-p0-ma20-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_110145": "t70-m10-o0-g0-r0-p0-ma20-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_117488": "t70-m10-o0-g0-r0-p0-ma20-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_124831": "t70-m10-o0-g0-r0-p0-ma20-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_132174": "t70-m10-o0-g0-r0-p0-ma20-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_139517": "t70-m10-o0-g0-r0-p0-ma20-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_resulting_eggs/resulting_eggs_143051": "t70-m10-o0-g0-r0-p0-ma20-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_7343": "t30-m15-o10-r5-p5-g20-ma15-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_14686": "t30-m15-o10-r5-p5-g20-ma15-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_22029": "t30-m15-o10-r5-p5-g20-ma15-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_29372": "t30-m15-o10-r5-p5-g20-ma15-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_36715": "t30-m15-o10-r5-p5-g20-ma15-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_44058": "t30-m15-o10-r5-p5-g20-ma15-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_51401": "t30-m15-o10-r5-p5-g20-ma15-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_58744": "t30-m15-o10-r5-p5-g20-ma15-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_66087": "t30-m15-o10-r5-p5-g20-ma15-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_73430": "t30-m15-o10-r5-p5-g20-ma15-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_80773": "t30-m15-o10-r5-p5-g20-ma15-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_88116": "t30-m15-o10-r5-p5-g20-ma15-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_95459": "t30-m15-o10-r5-p5-g20-ma15-95459",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_102802": "t30-m15-o10-r5-p5-g20-ma15-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_110145": "t30-m15-o10-r5-p5-g20-ma15-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_117488": "t30-m15-o10-r5-p5-g20-ma15-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_124831": "t30-m15-o10-r5-p5-g20-ma15-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_132174": "t30-m15-o10-r5-p5-g20-ma15-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_139517": "t30-m15-o10-r5-p5-g20-ma15-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_electoral_lithography/electoral_lithography_143051": "t30-m15-o10-r5-p5-g20-ma15-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_7343": "t50-m30-o20-r0-p0-g20-ma15-7343",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_102802": "t50-m30-o20-r0-p0-g20-ma15-102802",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_110145": "t50-m30-o20-r0-p0-g20-ma15-110145",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_117488": "t50-m30-o20-r0-p0-g20-ma15-117488",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_124831": "t50-m30-o20-r0-p0-g20-ma15-124831",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_132174": "t50-m30-o20-r0-p0-g20-ma15-132174",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_139517": "t50-m30-o20-r0-p0-g20-ma15-139517",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_143051": "t50-m30-o20-r0-p0-g20-ma15-143051",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_14686": "t50-m30-o20-r0-p0-g20-ma15-14686",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_22029": "t50-m30-o20-r0-p0-g20-ma15-22029",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_29372": "t50-m30-o20-r0-p0-g20-ma15-29372",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_36715": "t50-m30-o20-r0-p0-g20-ma15-36715",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_44058": "t50-m30-o20-r0-p0-g20-ma15-44058",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_51401": "t50-m30-o20-r0-p0-g20-ma15-51401",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_58744": "t50-m30-o20-r0-p0-g20-ma15-58744",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_66087": "t50-m30-o20-r0-p0-g20-ma15-66087",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_73430": "t50-m30-o20-r0-p0-g20-ma15-73430",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_80773": "t50-m30-o20-r0-p0-g20-ma15-80773",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_88116": "t50-m30-o20-r0-p0-g20-ma15-88116",
    "/mnt/sharefs/users/haolong.jia/checkpoint/tokenmix_ablation_near_habanera/near_habanera_95459": "t50-m30-o20-r0-p0-g20-ma15-95459",
}







