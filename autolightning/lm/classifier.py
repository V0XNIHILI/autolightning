from typing import Optional
from functools import partial

import torch.nn as nn

from torch_mate.utils import calc_accuracy

from .supervised import Supervised
from ..types import AutoModuleKwargsNoCriterion, Unpack


class ClassifierMixin:
    def __init__(self, top_k: int = 1, criterion: Optional[nn.Module] = None, **kwargs: Unpack[AutoModuleKwargsNoCriterion]):
        if criterion == None:
            criterion = nn.CrossEntropyLoss()

        super().__init__(criterion=criterion, **kwargs)

        self.top_k = top_k

    def configure_metrics(self):
        return {
            "accuracy": partial(calc_accuracy, k=self.top_k),
        }
        

class Classifier(ClassifierMixin, Supervised):
    pass
