from typing import (
    Dict,
    Optional,
    Union,
    Callable,
    Any,
    Literal,
    Iterable,
    TypedDict,
    List,
)

try:
    from typing import Unpack  # Python 3.11+
except ImportError:
    from typing_extensions import Unpack  # For older Python versions

import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, IterableDataset

from torchmetrics.metric import Metric
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable, ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau as PTReduceLROnPlateau

LRSchedulerCallableWithPTPlateu = Callable[[Optimizer], Union[LRScheduler, PTReduceLROnPlateau]]

LRSchedulerCallableAll = Callable[[Optimizer], Union[LRScheduler, ReduceLROnPlateau, PTReduceLROnPlateau]]

DatasetType = Union[Dataset, IterableDataset]

class LrSchedulerConfigType(TypedDict, total=False):
    scheduler: Union[LRSchedulerCallableWithPTPlateu, LRScheduler, PTReduceLROnPlateau]
    interval: str
    frequency: int
    monitor: str
    strict: bool
    name: Optional[str]


MetricType = Dict[str, Union[Metric, Callable[..., Any]]]
OptimizerType = Union[
    Optimizer,
    OptimizerCallable
]
LrSchedulerType = Union[LRSchedulerCallable, LRScheduler, ReduceLROnPlateau, LrSchedulerConfigType]
IterableOfModules = Iterable[nn.Module]

try:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass as BAMC
except ImportError:
    BAMC = nn.Module  # fallback type
NetType = Union[nn.Module, BAMC]

CallableOrModule = Union[Callable, nn.Module]
TransformValue = Union[List[CallableOrModule], CallableOrModule]
# A transform is either a single value applied to every phase, or a per-phase mapping
# (e.g. {"train": ..., "post": ...}). Defined here rather than in auto_data_module so
# that the **kwargs TypedDicts below can mirror AutoDataModule.__init__ exactly.
TransformType = Union[Dict[str, TransformValue], TransformValue]
AllDatasetsType = Union[DatasetType, Dict]

PHASES = ["train", "val", "test", "pred"]

Phase = Literal["train", "val", "test", "pred"]


# The TypedDicts below are used as `**kwargs: Unpack[...]` annotations. Since jsonargparse
# 4.34 those annotations are what the CLI parser reads: it builds its arguments from these
# keys rather than following the `super().__init__(**kwargs)` chain. So a key missing here
# is rejected on the CLI, and a type narrower than the real parameter silently rejects
# valid configs. Keep them mirroring the __init__ they document; tests/test_cli_parser.py
# fails if they drift.


class AutoModuleKwargs(TypedDict, total=False):
    net: Optional[NetType]
    criterion: Optional[nn.Module]
    optimizer: Optional[OptimizerType]
    lr_scheduler: Optional[LrSchedulerType]
    metrics: Optional[MetricType]
    loss_log_key: Optional[str]
    log_metrics: bool
    exclude_no_grad: bool
    disable_prog_bar: bool


class AutoModuleKwargsNoCriterion(TypedDict, total=False):
    net: Optional[NetType]
    optimizer: Optional[OptimizerType]
    lr_scheduler: Optional[LrSchedulerType]
    metrics: Optional[MetricType]
    loss_log_key: Optional[str]
    log_metrics: bool
    exclude_no_grad: bool
    disable_prog_bar: bool


class AutoModuleKwargsNoNet(TypedDict, total=False):
    criterion: Optional[nn.Module]
    optimizer: Optional[OptimizerType]
    lr_scheduler: Optional[LrSchedulerType]
    metrics: Optional[MetricType]
    loss_log_key: Optional[str]
    log_metrics: bool
    exclude_no_grad: bool
    disable_prog_bar: bool


class AutoModuleKwargsNoNetCriterion(TypedDict, total=False):
    optimizer: Optional[OptimizerType]
    lr_scheduler: Optional[LrSchedulerType]
    metrics: Optional[MetricType]
    loss_log_key: Optional[str]
    log_metrics: bool
    exclude_no_grad: bool
    disable_prog_bar: bool


class AutoDataModuleKwargs(TypedDict, total=False):
    dataset: Optional[Union[Dict[str, AllDatasetsType], AllDatasetsType]]
    dataloaders: Optional[Dict]
    transforms: Optional[TransformType]
    target_transforms: Optional[TransformType]
    batch_transforms: Optional[TransformType]
    target_batch_transforms: Optional[Union[TransformType, Literal["combine"]]]
    requires_prepare: bool
    pre_load: Union[Dict[str, bool], bool]
    random_split: Optional[Dict[str, Union[int, float]]]
    cross_val: Optional[Dict[str, int]]
    seed: Optional[int]
    build_plan: bool


class AutoDataModuleKwargsNoDatasetPrepareSplit(TypedDict, total=False):
    dataloaders: Optional[Dict]
    transforms: Optional[TransformType]
    target_transforms: Optional[TransformType]
    batch_transforms: Optional[TransformType]
    target_batch_transforms: Optional[Union[TransformType, Literal["combine"]]]
    pre_load: Union[Dict[str, bool], bool]
    cross_val: Optional[Dict[str, int]]
    seed: Optional[int]
    build_plan: bool


class ClassifierKwargs(TypedDict, total=False):
    top_k: int
    net: Optional[NetType]
    criterion: Optional[nn.Module]
    optimizer: Optional[OptimizerType]
    lr_scheduler: Optional[LrSchedulerType]
    metrics: Optional[MetricType]
    loss_log_key: Optional[str]
    log_metrics: bool
    exclude_no_grad: bool
    disable_prog_bar: bool


__all__ = [
    "Unpack",
    "MetricType",
    "OptimizerType",
    "LrSchedulerType",
    "IterableOfModules",
    "NetType",
    "CallableOrModule",
    "TransformValue",
    "TransformType",
    "AllDatasetsType",
    "Phase",
    "AutoModuleKwargs",
    "AutoModuleKwargsNoCriterion",
    "AutoModuleKwargsNoNet",
    "AutoModuleKwargsNoNetCriterion",
    "AutoDataModuleKwargs",
    "AutoDataModuleKwargsNoDatasetPrepareSplit",
    "ClassifierKwargs",
]
