from torch_mate.lightning import AutoModule

import torch_mate.lightning.lm.functional.supervised as LFSupervised

class SupervisedLearner(AutoModule):
    log_metrics = {"loss": "loss", "accuracy": "accuracy"}

    def shared_step(self, batch, batch_idx, phase: str):
        return LFSupervised.shared_step(self, batch, batch_idx, phase)
