import pytest
import torch.nn as nn
import torch.optim as optim
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import StepLR

from autolightning import AutoDataModule
from autolightning.auto_data_module import STAGES


TRAIN_KWARGS = {"batch_size": 32, "num_workers": 4}
VAL_KWARGS = {"batch_size": 64, "num_workers": 8}


def test_dataloader_kwargs_basic():
    kwargs = {"batch_size": 32, "num_workers": 4}

    data =  AutoDataModule(dataloaders=kwargs)

    for stage in STAGES:
        assert data.get_dataloader_kwargs(stage) == kwargs


def test_per_phase_dataloader_kwargs():
    kwargs = {"train": TRAIN_KWARGS, "val": VAL_KWARGS}

    data = AutoDataModule(dataloaders=kwargs)

    assert data.get_dataloader_kwargs("train") == TRAIN_KWARGS
    assert data.get_dataloader_kwargs("val") == VAL_KWARGS
    assert data.get_dataloader_kwargs("test") == data.get_dataloader_kwargs("pred") == {}


def test_per_phase_dataloader_kwargs():
    default_kwargs = {"batch_size": 16, "num_workers": 2}

    kwargs = {"train": TRAIN_KWARGS, "val": VAL_KWARGS, "defaults": default_kwargs}

    data = AutoDataModule(dataloaders=kwargs)

    assert data.get_dataloader_kwargs("train") == TRAIN_KWARGS
    assert data.get_dataloader_kwargs("val") == VAL_KWARGS
    assert data.get_dataloader_kwargs("test") == data.get_dataloader_kwargs("pred") == default_kwargs


def test_mixed_dataloader_kwargs():
    kwargs = {**TRAIN_KWARGS, "pred": {}}

    data = AutoDataModule(dataloaders=kwargs)

    for stage in STAGES:
        with pytest.raises(Exception):
            data.get_dataloader_kwargs(stage)
 