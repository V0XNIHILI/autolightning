import torch
import torch.nn as nn
import torchvision

from autolightning import AutoModule
from autolightning.nn.barlow_twins_losses import BarlowTwinsLoss
from autolightning.types import Phase, AutoModuleKwargsNoCriterion, AutoModuleKwargs, Unpack


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


def siamese_shared_step(module: AutoModule, batch):
    input1, input2, label = batch
    output1, output2 = module(input1), module(input2)

    return (output1, output2, label)


class SiameseMixin:
    def __init__(self, margin: float = 1.0, **kwargs: Unpack[AutoModuleKwargs]):
        if kwargs.get("criterion", None) == None:
            super().__init__(criterion=ContrastiveLoss(margin), **kwargs)
        else:
            super().__init__(**kwargs)


class Siamese(SiameseMixin, AutoModule):
    def shared_step(self, phase: Phase, batch, batch_idx):
        return siamese_shared_step(self, batch)


def triplet_shared_step(module: AutoModule, batch):
    anchor, positive, negative, _, _ = batch

    return module(anchor), module(positive), module(negative)


class TripletMixin:
    def __init__(self, margin: float = 1.0, p: int = 2, swap: bool = False, **kwargs: Unpack[AutoModuleKwargs]):
        if kwargs.get("criterion", None) == None:
            super().__init__(**{**kwargs, **dict(criterion=nn.TripletMarginLoss(margin=margin, p=p, swap=swap))})
        else:
            super().__init__(**kwargs)


class Triplet(TripletMixin, AutoModule):
    def shared_step(self, phase: Phase, batch, batch_idx):
        return triplet_shared_step(self, batch)
