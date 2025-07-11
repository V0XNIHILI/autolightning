from typing import Dict, Optional, Union, Callable, Any, Literal, Iterable, TypedDict, List

try:
    from typing import Unpack  # Python 3.11+
except ImportError:
    from typing_extensions import Unpack  # For older Python versions

import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torchmetrics.metric import Metric
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

MetricType = Dict[str, Union[Metric, Callable[..., Any]]]
OptimizerType = Union[Optimizer, OptimizerCallable, Iterable[Union[Optimizer, OptimizerCallable]], Dict[str, OptimizerCallable]]
LrSchedulerType = Union[LRSchedulerCallable, Dict]
IterableOfModules = Iterable[nn.Module]

try:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass as BAMC
except ImportError:
    BAMC = nn.Module  # fallback type
NetType = Union[nn.Module, BAMC]

CallableOrModule = Union[Callable, nn.Module]
TransformValue = Union[List[CallableOrModule], CallableOrModule]

Phase = Literal["train", "val", "test", "pred"]


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
    dataset: Optional[Union[Dict[str, Any], Dict, Any]]
    dataloaders: Optional[Dict]
    transforms: Optional[Callable]
    target_transforms: Optional[Callable]
    batch_transforms: Optional[Callable]
    requires_prepare: bool
    pre_load: Union[Dict[str, bool], bool]
    random_split: Optional[Dict[str, Union[Union[int, float], Union[str, Dict[str, Union[int, float]]]]]]
    cross_val: Optional[Dict[str, int]]
    seed: Optional[int]


class AutoDataModuleKwargsNoDatasetPrepareSplit(TypedDict, total=False):
    dataloaders: Optional[Dict]
    transforms: Optional[Callable]
    target_transforms: Optional[Callable]
    batch_transforms: Optional[Callable]
    pre_load: Union[Dict[str, bool], bool]
    cross_val: Optional[Dict[str, int]]
    seed: Optional[int]


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
    "Phase",
    "AutoModuleKwargs",
    "AutoModuleKwargsNoCriterion",
    "AutoModuleKwargsNoNet",
    "AutoModuleKwargsNoNetCriterion",
    "AutoDataModuleKwargs",
    "AutoDataModuleKwargsNoDatasetPrepareSplit",
    "ClassifierKwargs"
]
