from typing import Optional

import torch
import torch.nn as nn

from autolightning.lm.supervised import Supervised
from autolightning.lm.classifier import ClassifierMixin
from autolightning.types import Phase


def icl_forward(head_or_net: nn.Module, X_train, y_train, X_test, merge_data_strategy: str = "flatten", combine_batch_and_samples: bool = False, embedder: Optional[nn.Module] = None):
    if embedder is not None:
        batch = X_train.size(0)

        if combine_batch_and_samples:
            # Combine first and second dimension
            X_train = X_train.view(-1, *X_train.size()[2:]) # (batch * n_train_samples, ...)
            X_test = X_test.view(-1, *X_test.size()[2:]) # (batch * n_test_samples, ...)

        X_test = embedder(X_test) # (batch * n_test_samples, ...)
        X_train = embedder(X_train) # (batch * n_train_samples, ...)

        if combine_batch_and_samples:
            X_train = X_train.view(batch, -1, *X_train.size()[1:]) # (batch, n_train_samples, ...)
            X_test = X_test.view(batch, -1, *X_test.size()[1:]) # (batch, n_test_samples, ...)

    X = torch.cat([X_train, y_train, X_test], dim=1) # (batch, 2 * n_train_samples + n_test_samples, n_features)

    if merge_data_strategy == "flatten":
        X = X.view(X.size(0), -1)
    elif merge_data_strategy == "transpose":
        X = X.transpose(1, 2)
    else:
        raise ValueError(f"Unknown strategy: {merge_data_strategy}")
    
    return head_or_net(X)


def icl_shared_step(module: nn.Module, batch):
    (X_train, y_train), (X_test, y_test) = batch

    output = module(X_train, y_train, X_test)

    return (output, y_test)


class ICLMixin:
    def __init__(self, embedder: Optional[nn.Module] = None, merge_data_strategy: str = "flatten", combine_batch_and_samples: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.embedder = embedder
        self.merge_data_strategy = merge_data_strategy
        self.combine_batch_and_samples = combine_batch_and_samples


class ICL(ICLMixin, Supervised):
    def forward(self, X_train, y_train, X_test):
        return icl_forward(self.net, X_train, y_train, X_test, self.merge_data_strategy, self.combine_batch_and_samples, self.embedder)
    
    def shared_step(self, phase: Phase, batch, batch_idx):
        return icl_shared_step(self, batch)


class ICLClassifier(ClassifierMixin, ICL):
    pass
