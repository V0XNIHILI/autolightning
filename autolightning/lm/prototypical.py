from typing import Literal

from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression

from torch_mate.typing import OptionalBatchTransform

from autolightning.lm import Classifier
from autolightning.types import AutoModuleKwargs, Unpack


MetaSample = Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
MetaBatch = List[MetaSample]
MetricType = Literal[
    "euclidean",
    "euclidean-squared",
    "logistic-regression",
    "dot",
    "dot-sqrt",
    "manhattan",
    "cosine",
]


UNKNOWN_METRIC_MESSAGE = "Must be one of [euclidean, euclidean-squared, manhattan, dot, cosine, logistic-regression]"


def prototypical_forward(
    embedder: nn.Module,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    train_labels: torch.Tensor,
    metric: str,
    average_support_embeddings: bool,
    batch_transform: OptionalBatchTransform = None,
    support_embedding_transform: Optional[nn.Module] = None,
):
    # It is assumed that the train labels are structured like [0] * k_shot + [1] * k_shot, ...
    # and the evaluation labels are structured like [0] * k_query_shot + [1] * k_query_shot, ...

    data = torch.cat([train_data, test_data], dim=0)

    if batch_transform:
        data = batch_transform(data)

    embeddings = embedder(data)

    support_embeddings = embeddings[: train_data.size(0)]
    query_embeddings = embeddings[train_data.size(0) :]

    k_shot = len(train_labels) // len(torch.unique(train_labels))

    if average_support_embeddings:
        # Average every k-shot embeddings to get a single embedding for each class
        grouped_embeddings = torch.reshape(support_embeddings, (-1, k_shot, support_embeddings.size(1)))
        support_embeddings = torch.mean(grouped_embeddings, dim=1)

        if metric == "logistic-regression":
            raise ValueError("Cannot use logistic regression with average support embeddings")
        
    if support_embedding_transform is not None:
        support_embeddings = support_embedding_transform(support_embeddings)

    if metric.startswith("euclidean"):
        similarities = -torch.cdist(query_embeddings, support_embeddings)

        if metric == "euclidean-squared":
            similarities = -(similarities**2)
        elif metric != "euclidean":
            raise ValueError(f"Unknown metric: {metric}. {UNKNOWN_METRIC_MESSAGE}")
    elif metric == "logistic-regression":
        # TODO: set random state
        clf = LogisticRegression(random_state=0).fit(support_embeddings.cpu().numpy(), train_labels.cpu().numpy())
        similarities = torch.tensor(
            clf.predict_proba(query_embeddings.cpu().numpy()),
            device=query_embeddings.device,
        )
    elif metric.startswith("dot"):
        similarities = torch.matmul(query_embeddings, support_embeddings.t())

        if metric == "dot-sqrt":
            similarities = torch.sqrt(similarities)
        elif metric != "dot":
            raise ValueError(f"Unknown metric: {metric}. {UNKNOWN_METRIC_MESSAGE}")
    elif metric == "manhattan":
        similarities = -torch.cdist(query_embeddings, support_embeddings, p=1)
    elif metric == "cosine":
        query_norms = F.normalize(query_embeddings, dim=1)
        support_norms = F.normalize(support_embeddings, dim=1)

        similarities = query_norms @ support_norms.T
    else:
        raise ValueError(f"Unknown metric: {metric}. {UNKNOWN_METRIC_MESSAGE}")

    if average_support_embeddings or metric == "logistic-regression":
        shots_per_class = 1
    else:
        shots_per_class = k_shot

    # Take the maximum similarity per query sample between all support samples of a class. This means that in case there
    # are multiple support samples (k_shot > 1) for a certain class, per support class we find the support sample that is most similar
    # to the query sample.
    # Evaluation data is shaped like [(class1, shot1, data), (class1, shot2, data), (class2, shot1, data), ...)]
    similarities = torch.reshape(
        torch.max(torch.reshape(similarities, (-1, shots_per_class)), dim=1).values,
        (-1, similarities.shape[1] // shots_per_class),
    )

    return similarities


def prototypical_shared_step(module, batch):
    ((X_train, y_train), (X_test, y_test)) = batch

    all_similarities = []
    all_evaluation_labels = []

    meta_batch_size = len(y_test)

    for task_idx in range(meta_batch_size):
        similarities = module(X_train[task_idx], X_test[task_idx], y_train[task_idx])

        all_similarities.append(similarities)
        all_evaluation_labels.append(y_test[task_idx])

    return torch.cat(all_similarities), torch.cat(all_evaluation_labels)


class PrototypicalMixin:
    def __init__(
        self,
        metric: MetricType = "euclidean",
        average_support_embeddings: bool = True,
        support_embedding_transform: Optional[nn.Module] = None,
        **kwargs: Unpack[AutoModuleKwargs],
    ):
        super().__init__(**kwargs)

        self.metric = metric
        self.average_support_embeddings = average_support_embeddings
        self.support_embedding_transform = support_embedding_transform


class Prototypical(PrototypicalMixin, Classifier):
    def forward(self, train_data, test_data, train_labels):
        return prototypical_forward(
            self.net,
            train_data,
            test_data,
            train_labels,
            self.metric,
            self.average_support_embeddings,
            support_embedding_transform=self.support_embedding_transform,
        )

    def shared_step(self, phase: str, batch, batch_idx):
        return prototypical_shared_step(self, batch)
