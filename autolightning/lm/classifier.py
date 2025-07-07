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

        self.top_k = top_k

        super().__init__(criterion=criterion, **kwargs) # type: ignore

    def configure_metrics(self):
        return {
            "accuracy": partial(calc_accuracy, k=self.top_k),
        }
        

class Classifier(ClassifierMixin, Supervised):
    pass
