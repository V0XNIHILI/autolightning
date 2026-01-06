from typing import Any

from autolightning import AutoModule
from autolightning.types import Phase, NetType


def supervised_forward(module: NetType, *args, **kwargs):
    return module(*args, **kwargs)


def supervised_shared_step(phase: Phase, module: NetType, inputs: Any, targets: Any):
    output = module(inputs)

    return (output, targets)


class Supervised(AutoModule):
    def forward(self, *args, **kwargs):
        return supervised_forward(self.net, *args, **kwargs)

    def shared_step(self, phase: Phase, batch, batch_idx):
        return supervised_shared_step(phase, self, batch[0], batch[1])
