from typing import Dict, Optional, Union

import torch

from torch_mate.data.utils import FewShot as FS

from .. import AutoDataModule
from ..types import Unpack, AutoDataModuleKwargs


def before_after_batch_transfer_flatten(on_before_after_method, batch, dataloader_idx: int, flatten_for_batch_transform: bool):
    if flatten_for_batch_transform:
        ((X_support, y_support), (X_query, y_query)) = batch

        # Concat X_support and X_query along the batch dimension
        X_all = torch.cat([X_support, X_query], dim=0)
        # Combine first and second dimension
        X_all = X_all.view(-1, *X_all.shape[2:])

        y_all = torch.cat([y_support, y_query], dim=0)
        y_all = y_all.view(-1)

        batch = (X_all, y_all)

    batch = on_before_after_method(batch, dataloader_idx)

    if flatten_for_batch_transform:
        # Split X_all into X_support and X_query
        X_all, y_all = batch
        num_x_support = y_support.shape[0] * y_support.shape[1]
        X_support = X_all[:num_x_support].view(y_support.shape[0], y_support.shape[1], *X_all.shape[1:])
        X_query = X_all[num_x_support:].view(y_query.shape[0], y_query.shape[1], *X_all.shape[1:])
        y_support = y_all[:num_x_support].view(y_support.shape)
        y_query = y_all[num_x_support:].view(y_query.shape)

        batch = ((X_support, y_support), (X_query, y_query))

    return batch


class FewShotMixin:
    def __init__(self,
                 ways: int = 5,
                 shots: int = 1,
                 train_ways: int = -1,
                 query_shots: int = -1,
                 query_ways: int = -1,
                 train_query_shots: int = -1,
                 keep_original_labels: bool = False,
                 shuffle_labels: bool = False,
                 samples_per_class: Optional[Dict[str, int]] = None,
                 flatten_for_batch_transform: Union[bool, Dict[str, bool]] = False,
                 **kwargs: Unpack[AutoDataModuleKwargs]):
        super().__init__(**kwargs)
        
        self.ways = ways
        self.shots = shots
        self.train_ways = train_ways
        self.query_shots = query_shots
        self.query_ways = query_ways
        self.train_query_shots = train_query_shots
        self.keep_original_labels = keep_original_labels
        self.shuffle_labels = shuffle_labels
        self.samples_per_class = samples_per_class
        self.flatten_for_batch_transform = flatten_for_batch_transform

    def get_transformed_dataset(self, phase: str):
        dataset = super().get_transformed_dataset(phase)

        ways = self.ways
        query_shots = self.query_shots

        if phase == 'train':
            if self.train_ways != -1:
                ways = self.train_ways

            if self.train_query_shots != -1:
                query_shots = self.train_query_shots

        return FS(
            dataset,
            n_way=ways,
            k_shot=self.shots,
            query_shots=query_shots,
            query_ways=self.query_ways,
            keep_original_labels=self.keep_original_labels,
            shuffle_labels=self.shuffle_labels,
            samples_per_class=self.samples_per_class[phase] if self.samples_per_class else None
        )
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        flatten = self.flatten_for_batch_transform if isinstance(self.flatten_for_batch_transform, bool) else self.flatten_for_batch_transform.get('before', False)
        return before_after_batch_transfer_flatten(super().on_before_batch_transfer, batch, dataloader_idx, flatten)
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        flatten = self.flatten_for_batch_transform if isinstance(self.flatten_for_batch_transform, bool) else self.flatten_for_batch_transform.get('after', False)
        return before_after_batch_transfer_flatten(super().on_after_batch_transfer, batch, dataloader_idx, flatten)


class FewShot(FewShotMixin, AutoDataModule):
    pass
