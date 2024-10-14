from typing import Optional

import torch
import torch.nn as nn

from autolightning import AutoModule
from autolightning.types import Phase


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()

        self.margin = margin

    def forward(self, x1, x2, label):
        dist = torch.nn.functional.pairwise_distance(x1, x2)

        loss = (1 - label) * torch.pow(dist, 2) \
            + (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(loss)

        return loss


def multi_argument_forward(module: nn.Module, *args):
    return module(*args)


def siamese_forward(module: nn.Module, input1, input2):
    return multi_argument_forward(module, input1, input2)


def siamese_shared_step(module: AutoModule, batch):
    input1, input2, label = batch
    output1, output2 = module(input1, input2)

    return (output1, output2, label)


class SiameseMixin:
    def __init__(self, criterion: Optional[nn.Module] = None, **kwargs):
        if criterion == None:
            criterion = ContrastiveLoss()

        super().__init__(criterion=criterion, **kwargs)


class Siamese(SiameseMixin, AutoModule):
    def forward(self, input1, input2):
        return siamese_forward(self.net, input1, input2)
    
    def shared_step(self, phase: Phase, batch, batch_idx):
        return siamese_shared_step(self, batch)


def triplet_forward(module: nn.Module, anchor, positive, negative):
    return multi_argument_forward(module, positive, negative)


def triplet_shared_step(module: AutoModule, batch):
    anchor, positive, negative = batch
    output_anch, output_pos, output_neg = module(module, anchor, positive, negative)

    return (output_anch, output_pos, output_neg)


class TripletMixin:
    def __init__(self, criterion: Optional[nn.Module] = None, **kwargs):
        if criterion == None:
            criterion = nn.TripletMarginLoss()

        super().__init__(criterion=criterion, **kwargs)


class Triplet(TripletMixin, AutoModule):
    def forward(self, anchor, positive, negative):
        return triplet_forward(self.net, anchor, positive, negative)
    
    def shared_step(self, phase: Phase, batch, batch_idx):
        return triplet_shared_step(self, batch)
