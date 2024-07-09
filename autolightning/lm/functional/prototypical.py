from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from sklearn.linear_model import LogisticRegression
from torch_mate.utils import calc_accuracy

from torch_mate.typing import OptionalBatchTransform

from autolightning import AutoModule

MetaSample = Tuple[Tuple[torch.Tensor, torch.Tensor],
                                            Tuple[torch.Tensor, torch.Tensor]]
MetaBatch = List[MetaSample]


UNKNOWN_METRIC_MESSAGE = 'Must be one of [euclidean, euclidean-squared, manhattan, dot, cosine, logistic-regression]'


def few_shot_nearest_neighbor(embedder: nn.Module,
                              loss: nn.Module,
                              metric: str,
                              average_support_embeddings: bool,
                              sample: MetaSample,
                              batch_transform: OptionalBatchTransform = None):
    # It is assumed that the train labels are structured like [0] * k_shot + [1] * k_shot, ...
    # and the evaluation labels are structured like [0] * k_query_shot + [1] * k_query_shot, ...
    ((train_data, test_data), (train_labels, evaluation_labels)) = sample

    data = torch.cat([train_data, test_data], dim=0)

    if batch_transform:
        data = batch_transform(data)

    embeddings = embedder(data)
    support_embeddings = embeddings[:train_data.size(0)]
    query_embeddings = embeddings[train_data.size(0):]

    k_shot = len(train_labels) // len(torch.unique(train_labels))

    if average_support_embeddings:
        # Average every k-shot embeddings to get a single embedding for each class
        grouped_embeddings = torch.reshape(support_embeddings, (-1, k_shot, support_embeddings.size(1)))
        support_embeddings = torch.mean(grouped_embeddings, dim=1)

        if metric == 'logistic-regression':
            raise ValueError('Cannot use logistic regression with average support embeddings')

    if metric.startswith('euclidean'):
        similarities = -torch.cdist(query_embeddings, support_embeddings)

        if metric == 'euclidean-squared':
            similarities = -similarities ** 2
        elif not metric.endswith('euclidean'):
            raise ValueError(f'Unknown metric: {metric}. {UNKNOWN_METRIC_MESSAGE}')
    elif metric == 'logistic-regression':
        # TODO: set random state
        clf = LogisticRegression(random_state=0).fit(support_embeddings.cpu().numpy(), train_labels.cpu().numpy())
        similarities = torch.tensor(clf.predict_proba(query_embeddings.cpu().numpy()), device=evaluation_labels.device)
    elif metric == 'dot':
        similarities = torch.matmul(query_embeddings, support_embeddings.t())
    elif metric == 'manhattan':
        similarities = -torch.cdist(query_embeddings, support_embeddings, p=1)
    elif metric == 'cosine':
        similarities = F.cosine_similarity(
                        query_embeddings.unsqueeze(1),  # Shape: (total_queries, 1, embedding_dim)
                        support_embeddings.unsqueeze(0),  # Shape: (1, support_size, embedding_dim)
                        dim=2  # Compute cosine similarity along the embedding dimension
                    )
    else:
        raise ValueError(f'Unknown metric: {metric}. {UNKNOWN_METRIC_MESSAGE}')

    if average_support_embeddings or metric == 'logistic-regression':
        shots_per_class = 1
    else:
        shots_per_class = k_shot

    # Take the maximum similarity per query sample between all support samples of a class. This means that in case there
    # are multiple support samples (k_shot > 1) for a certain class, per support class we find the support sample that is most similar
    # to the query sample.
    # Evaluation data is shaped like [(class1, shot1, data), (class1, shot2, data), (class2, shot1, data), ...)]
    similarities = torch.reshape(
        torch.max(torch.reshape(similarities, (-1, shots_per_class)),
                  dim=1).values, (-1, similarities.shape[1] // shots_per_class))

    evaluation_error = loss(similarities, evaluation_labels)

    evaluation_accuracy = calc_accuracy(similarities, evaluation_labels)

    return evaluation_error, evaluation_accuracy


def process_metric_learning_batch(embedder: nn.Module, batch: MetaBatch,
                                metric: str, average_support_embeddings: bool,
                                batch_transform: OptionalBatchTransform,
                                loss: nn.Module):
    ((X_train, X_test),
        (y_train, y_test)) = batch

    meta_error = 0.0
    meta_accuracies = []

    meta_batch_size = len(y_test)

    for task_idx in range(meta_batch_size):
        error, accuracy = few_shot_nearest_neighbor(
            embedder, loss, metric, average_support_embeddings,
            ((X_train[task_idx], X_test[task_idx]),
                (y_train[task_idx], y_test[task_idx])), batch_transform)

        meta_error += error
        meta_accuracies.append(accuracy.item())

    meta_accuracies = np.array(meta_accuracies)

    # Return float values instead of np.float64 types as this causes issues with logging
    # these values with PyTorch Lightning on macOS with MPS support.
    accuracy = float(np.mean(meta_accuracies))
    confidence_interval = 1.96 * float(np.std(meta_accuracies) / np.sqrt(meta_batch_size))

    return meta_error / meta_batch_size, (accuracy, confidence_interval)


def shared_step(module: AutoModule, batch, batch_idx, phase: str):
    loss, (accuracy, confidence_interval) = process_metric_learning_batch(
        module, batch, module.hparams.learner["cfg"]["metric"],
        module.hparams.learner["cfg"]["average_support_embeddings"],
        None, module.criteria)
    
    log_dict = {
        module.log_key(f"meta_{phase}", "loss"): loss,
        module.log_key(f"meta_{phase}", "accuracy"): accuracy
    }

    if module.hparams.learner["cfg"]["log_confidence_interval"]:
        log_dict[module.log_key(f"meta_{phase}", "accuracy@95")] = confidence_interval
    
    module.log_dict(log_dict, prog_bar=phase == 'val')
    
    return loss
