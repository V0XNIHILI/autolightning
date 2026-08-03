from typing import Optional, Any

import torch.nn as nn

from torch_mate.utils import calc_binary_accuracy

from .supervised import Supervised, supervised_forward, supervised_shared_step
from ..types import AutoModuleKwargsNoCriterion, Unpack, Phase


class BinaryClassifierMixin:
    def __init__(
        self,
        criterion: Optional[nn.Module] = None,
        **kwargs: Unpack[AutoModuleKwargsNoCriterion],
    ):
        if criterion is None:
            criterion = nn.BCELoss()

        super().__init__(criterion=criterion, **kwargs)  # type: ignore

    def forward(self, *args, **kwargs):
        return supervised_forward(self.net, *args, **kwargs).flatten()

    def shared_step(self, phase: Phase, batch, batch_idx):
        output, targets = supervised_shared_step(phase, self, batch[0], batch[1])
        return output, targets.float()

    def configure_metrics(self):
        return {
            "accuracy": calc_binary_accuracy,
        }


class BinaryClassifier(BinaryClassifierMixin, Supervised):
    pass
