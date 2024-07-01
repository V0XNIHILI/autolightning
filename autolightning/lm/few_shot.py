from typing import Dict

from autolightning import AutoModule

import autolightning.lm.functional.prototypical as LFPrototypical


class PrototypicalLearner(AutoModule):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        few_shot_phases = ["meta_train", "meta_val", "meta_test"]

        for phase in few_shot_phases:
            self.log_phases[phase] = phase

    def shared_step(self, batch, batch_idx, phase: str):
        return LFPrototypical.shared_step(self, batch, batch_idx, phase)
    
    def config_model(self):
        model = super().config_model()

        if self.hparams.learner.get("cfg", {}).get("embedder_key", None):
            return dict(model.named_modules())[self.hparams.learner["cfg"]["embedder_key"]]
        
        return model
