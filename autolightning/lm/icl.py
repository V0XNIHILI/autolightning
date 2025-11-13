from typing import Optional, TypedDict

import torch
import torch.nn as nn

from autolightning import AutoModule
from autolightning.lm.classifier import ClassifierMixin
from autolightning.types import NetType, Phase, AutoModuleKwargs, Unpack


def icl_forward(
    head_or_net: nn.Module,
    X_train,
    y_train,
    X_test,
    sample_embedder: Optional[nn.Module] = None,
    query_sample_embedder: Optional[nn.Module] = None,
    merge_embedding_strategy: str = "flatten",
    combine_batch_and_samples: bool = False,
):
    batch = X_train.size(0)

    if sample_embedder is not None or query_sample_embedder is not None:
        def embed(X, embedder):
            if embedder is None:
                return X

            if combine_batch_and_samples:
                # Combine first and second dimension (X: (batch, ways*shots, ...))
                X = X.view(-1, *X.size()[2:]) # (batch * n_train/test_samples, ...)

            X = embedder(X) # (batch * n_train/test_samples, ...)

            if combine_batch_and_samples:
                X = X.view(batch, -1, *X.size()[1:]) # (batch, n_train/test_samples, ...)

            return X

        X_train = embed(X_train, sample_embedder)
        X_test = embed(X_test, query_sample_embedder)

    X = None

    if merge_embedding_strategy.startswith("flatten"):
        X_train_test = torch.cat([X_train, X_test], dim=1).view(
            batch, -1
        )  # (batch, (n_train_samples + n_test_samples) * total_n_features)

        if merge_embedding_strategy == "flatten_without_labels":
            X = X_train_test
        else:
            # Flatten y_train separately to make sure that different embedding
            # size between y and X are handled correctly
            y_train = y_train.view(batch, -1)  # (batch, n_train_samples * n_label_features)

            X = torch.cat(
                [X_train_test, y_train], dim=1
            )  # (batch, (n_train_samples + n_test_samples) * total_n_features + n_train_samples * n_label_features)
    elif merge_embedding_strategy == "transpose":
        assert X_train.shape[2:] == y_train.shape[2:], (
            "X and y should have the same number of features for the transpose strategy to work"
        )

        X = torch.cat(
            [X_train, y_train, X_test], dim=1
        )  # (batch, 2 * n_train_samples + n_test_samples, total_n_features)
        X = X.transpose(1, 2)
    else:
        raise ValueError(f"Unknown strategy: {merge_embedding_strategy}")

    return head_or_net(X)


def icl_shared_step(module: nn.Module, batch):
    (X_train, y_train), (X_test, y_test) = batch

    output = module(X_train, y_train, X_test)

    # y_test is of shape (batch, 1) here, but we need it to be of shape (batch,) for CE loss
    y_test = y_test.flatten()

    return (output, y_test)


class ICLKwargs(TypedDict, total=False):
    sample_embedder: Optional[nn.Module]
    merge_embedding_strategy: str
    combine_batch_and_samples: bool


class ICLMixin:
    def __init__(
        self,
        sample_embedder: Optional[NetType] = None,
        query_sample_embedder: Optional[NetType] = None, # if None, use sample_embedder for query as well
        label_embedder: Optional[NetType] = None,
        merge_embedding_strategy: str = "flatten",
        combine_batch_and_samples: bool = False,
        **kwargs: Unpack[AutoModuleKwargs],
    ):
        super().__init__(**kwargs)

        self.sample_embedder = sample_embedder
        self.query_sample_embedder = query_sample_embedder
        self.label_embedder = label_embedder
        self.merge_embedding_strategy = merge_embedding_strategy
        self.combine_batch_and_samples = combine_batch_and_samples


class ICL(ICLMixin, AutoModule):
    def forward(self, X_train, y_train, X_test):
        y_train = self.label_embedder(y_train) if self.label_embedder is not None else y_train

        return icl_forward(
            self.net,
            X_train,
            y_train,
            X_test,
            self.sample_embedder,
            self.query_sample_embedder if self.query_sample_embedder is not None else self.sample_embedder,
            self.merge_embedding_strategy,
            self.combine_batch_and_samples,
        )

    def shared_step(self, phase: Phase, batch, batch_idx):
        return icl_shared_step(self, batch)


class ICLClassifier(ClassifierMixin, ICL):
    # Can also be seen as a RelationNet
    pass
