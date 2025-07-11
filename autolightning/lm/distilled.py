from typing import Optional

import torch
import torch.nn as nn

from .. import AutoModule
from ..types import AutoModuleKwargsNoNet, Unpack
from ..utils import disable_grad


def distilled_forward(student: nn.Module, student_head : Optional[nn.Module] = None, student_regressor : Optional[nn.Module]= None, *args, **kwargs):
    outputs = student(*args, **kwargs)

    if student_head is not None:
        if student_regressor:
            features = student_regressor(outputs)
        else:
            features = outputs

        return student_head(outputs), features
    
    return outputs, None


def distilled_shared_step(module: nn.Module, teacher_only: nn.Module, targets, *args, **kwargs):
    outputs, possible_features = module(*args, **kwargs)

    with torch.no_grad():
        teacher_outputs = teacher_only.eval()(*args, **kwargs)

    if possible_features == None:
        return (outputs, teacher_outputs, targets)
    
    return (outputs, teacher_outputs, possible_features, targets)


class DistilledMixin:
    def __init__(self, student_net: nn.Module, teacher_net: nn.Module, student_head_net: Optional[nn.Module] = None, student_regressor_net: Optional[nn.Module] = None, **kwargs: Unpack[AutoModuleKwargsNoNet]):
        super().__init__(net=None, **kwargs)

        if student_head_net == None and student_regressor_net is not None:
            raise ValueError("Cannot use student_regressor without student_head")

        self.student_net = student_net
        self.teacher = teacher_net
        self.student_head = student_head_net

        disable_grad(self.teacher)


class Distilled(DistilledMixin, AutoModule):
    def forward(self, *args, **kwargs):
        return distilled_forward(self.student_net, self.teacher, self.student_head, self.student_regressor)
    
    def shared_step(self, phase: str, batch, batch_idx):
        return distilled_shared_step(self, self.teacher, batch[1], batch[0])
