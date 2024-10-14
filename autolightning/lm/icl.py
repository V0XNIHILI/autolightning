from typing import Optional

import torch.nn as nn

from autolightning import AutoModule
from autolightning.lm.supervised import ClassifierMixin


def icl_forward(head_or_net: nn.Module, embedder: Optional[nn.Module] = None, *args, **kwargs):
    if embedder is not None:
        return head_or_net(embedder(*args, **kwargs))
    
    return head_or_net(*args, **kwargs)


class ICLMixin:
    def __init__(self, embedder: Optional[nn.Module] = None, **kwargs):
        super().__init__(**kwargs)

        self.embedder = embedder

class ICL(ICLMixin, ClassifierMixin, AutoModule):
    def forward(self, *args, **kwargs):
        return icl_forward(self.net, self.embedder, *args, **kwargs)

