from typing import Dict

from autolightning import AutoModule

import autolightning.lm.functional.supervised as LFSupervised

class SupervisedLearner(AutoModule):
    def shared_step(self, batch, batch_idx, phase: str):
        return LFSupervised.shared_step(self, batch, batch_idx, phase)
