from typing import Optional, Any

import torch.nn as nn

from torch_mate.utils import calc_binary_accuracy

from .supervised import Supervised, supervised_forward
from ..types import AutoModuleKwargsNoCriterion, Unpack, Phase, NetType


def binary_classifier_shared_step(phase: Phase, module: NetType, inputs: Any, targets: Any):
    if isinstance(inputs, tuple):
        output = module(*inputs)
    else:
        # If inputs is a dict/list/tensor
        output = module(inputs)

    return (output, targets.float())


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
        return binary_classifier_shared_step(phase, self, batch[0], batch[1])

    def configure_metrics(self):
        return {
            "accuracy": calc_binary_accuracy,
        }


class BinaryClassifier(BinaryClassifierMixin, Supervised):
    pass
