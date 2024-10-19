from typing import Dict, Optional, Union, Callable, Any, Literal, Iterable, TypedDict, Unpack

import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torchmetrics.metric import Metric
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable


MetricType = Dict[str, Union[Metric, Callable[..., Any]]]
OptimizerType = Union[Optimizer, OptimizerCallable, Iterable[Union[Optimizer, OptimizerCallable]], Dict[str, OptimizerCallable]]
LrSchedulerType = Union[LRSchedulerCallable, ]
IterableOfModules = Iterable[nn.Module]

Phase = Literal["train", "val", "test"]


class AutoModuleKwargs(TypedDict, total=False):
    net: Optional[nn.Module]
    criterion: Optional[nn.Module]
    optimizer: Optional[OptimizerType]
    lr_scheduler: Optional[LrSchedulerType]
    compiler: Optional[Callable[..., Callable]]
    metrics: Optional[MetricType]
    loss_log_key: Optional[str]
    log_metrics: bool
    exclude_no_grad: bool
    disable_prog_bar: bool


class AutoModuleKwargsNoCriterion(TypedDict, total=False):
    net: Optional[nn.Module]
    optimizer: Optional[OptimizerType]
    lr_scheduler: Optional[LrSchedulerType]
    compiler: Optional[Callable[..., Callable]]
    metrics: Optional[MetricType]
    loss_log_key: Optional[str]
    log_metrics: bool
    exclude_no_grad: bool
    disable_prog_bar: bool


class AutoModuleKwargsNoNet(TypedDict, total=False):
    criterion: Optional[nn.Module]
    optimizer: Optional[OptimizerType]
    lr_scheduler: Optional[LrSchedulerType]
    compiler: Optional[Callable[..., Callable]]
    metrics: Optional[MetricType]
    loss_log_key: Optional[str]
    log_metrics: bool
    exclude_no_grad: bool
    disable_prog_bar: bool
