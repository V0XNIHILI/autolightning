from typing import Optional, TypedDict

import torch
import torch.nn as nn

from autolightning.lm.supervised import Supervised
from autolightning.lm.classifier import ClassifierMixin
from autolightning.types import Phase, AutoModuleKwargs, Unpack


def icl_forward(
    head_or_net: nn.Module,
    X_train,
    y_train,
    X_test,
    sample_embedder: Optional[nn.Module] = None,
    merge_data_strategy: str = "flatten",
    combine_batch_and_samples: bool = False,
):
    batch = X_train.size(0)

    if sample_embedder is not None:
        if combine_batch_and_samples:
            # Combine first and second dimension
            X_train = X_train.view(-1, *X_train.size()[2:])  # (batch * n_train_samples, ...)
            X_test = X_test.view(-1, *X_test.size()[2:])  # (batch * n_test_samples, ...)

        X_test = sample_embedder(X_test)  # (batch * n_test_samples, ...)
        X_train = sample_embedder(X_train)  # (batch * n_train_samples, ...)

        if combine_batch_and_samples:
            X_train = X_train.view(batch, -1, *X_train.size()[1:])  # (batch, n_train_samples, ...)
            X_test = X_test.view(batch, -1, *X_test.size()[1:])  # (batch, n_test_samples, ...)

    if merge_data_strategy == "flatten":
        X_train_test = torch.cat([X_train, X_test], dim=1).view(
            batch, -1
        )  # (batch, (n_train_samples + n_test_samples) * total_n_features)

        # Flatten y_train separately to make sure that different embedding
        # size between y and X are handled correctly
        y_train = y_train.view(batch, -1)  # (batch, n_train_samples * n_label_features)

        X = torch.cat(
            [X_train_test, y_train], dim=1
        )  # (batch, (n_train_samples + n_test_samples) * total_n_features + n_train_samples * n_label_features)
    elif merge_data_strategy == "transpose":
        X = torch.cat(
            [X_train, y_train, X_test], dim=1
        )  # (batch, 2 * n_train_samples + n_test_samples, total_n_features)
        X = X.transpose(1, 2)
    else:
        raise ValueError(f"Unknown strategy: {merge_data_strategy}")

    return head_or_net(X)


def icl_shared_step(module: nn.Module, batch):
    (X_train, y_train), (X_test, y_test) = batch

    output = module(X_train, y_train, X_test)

    # y_test is of shape (batch, 1) here, but we need it to be of shape (batch,)
    y_test = y_test.flatten()

    return (output, y_test)


class ICLKwargs(TypedDict, total=False):
    sample_embedder: Optional[nn.Module]
    merge_data_strategy: str
    combine_batch_and_samples: bool


class ICLMixin:
    def __init__(
        self,
        sample_embedder: Optional[nn.Module] = None,
        merge_data_strategy: str = "flatten",
        combine_batch_and_samples: bool = False,
        **kwargs: Unpack[AutoModuleKwargs],
    ):
        super().__init__(**kwargs)

        self.sample_embedder = sample_embedder
        self.merge_data_strategy = merge_data_strategy
        self.combine_batch_and_samples = combine_batch_and_samples


class ICL(ICLMixin, Supervised):
    def forward(self, X_train, y_train, X_test):
        return icl_forward(
            self.net,
            X_train,
            y_train,
            X_test,
            self.sample_embedder,
            self.merge_data_strategy,
            self.combine_batch_and_samples,
        )

    def shared_step(self, phase: Phase, batch, batch_idx):
        return icl_shared_step(self, batch)


class ICLClassifierKwargs(ICLKwargs, AutoModuleKwargs):
    pass


class ICLClassifier(ClassifierMixin, ICL):
    def __init__(
        self,
        label_embedder: Optional[nn.Module] = None,
        **kwargs: Unpack[ICLClassifierKwargs],
    ):
        super().__init__(**kwargs)

        self.label_embedder = label_embedder

    def forward(self, X_train, y_train, X_test):
        y_train = self.label_embedder(y_train) if self.label_embedder is not None else y_train

        return super().forward(X_train, y_train, X_test)
