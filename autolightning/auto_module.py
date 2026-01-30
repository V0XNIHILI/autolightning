from typing import Dict, Optional, Any, Iterator, Union, Callable, Tuple
import warnings

import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics.metric import Metric

from .types import MetricType, OptimizerType, LrSchedulerType, NetType, Phase


LOG_PHASE_KEYS = {"train", "val", "test", "predict"}
LOG_ORDER_OPTIONS = {"phase_first", "metric_first"}
KEYS_TO_IGNORE = [
    "net",
    "criterion",
    "metrics",
    "optimizer",
    "compiler",
    "loss_log_key",
    "log_metrics",
]


def _call_with_flexible_args(func: Callable, args: Any) -> Any:
    if isinstance(args, (tuple, list)):
        return func(*args)
    if isinstance(args, dict):
        return func(**args)
    raise TypeError(f"Invalid argument type: {type(args)}")


def _resolve_metric(metric, default_log_kwargs: Dict[str, Any]) -> Tuple[Union[Callable, Any], Dict[str, Any]]:
    metric_func_or_value = metric
    metric_specific_log_kwargs = default_log_kwargs

    if isinstance(metric, dict):
        metric_func_or_value = metric["func"]
        metric_specific_log_kwargs = default_log_kwargs | metric.get("log_kwargs", {})

    return metric_func_or_value, metric_specific_log_kwargs


def _get_scheduler(scheduler: LrSchedulerType, optimizer: optim.Optimizer, should_be_callable: bool = False):
    if isinstance(scheduler, optim.lr_scheduler.LRScheduler):
        if should_be_callable:
            raise TypeError("Expected scheduler to be a callable or a scheduler dict, but got a scheduler instance")

        return scheduler

    if callable(scheduler):
        return scheduler(optimizer)
    
    if isinstance(scheduler, dict):
        sched = scheduler["scheduler"]

        if callable(sched):
            sched_inst = sched(optimizer)
        elif should_be_callable:
            raise TypeError("Expected scheduler to be a callable, but got a scheduler instance")
        else:
            sched_inst = sched

        init_sched = {key: value for key, value in scheduler.items() if key != "scheduler"}
        init_sched["scheduler"] = sched_inst

        return init_sched

    raise TypeError(
        f"Invalid scheduler type: {type(scheduler)}; expected either a scheduler, scheduler dict or a callable"
    )


class AutoModule(L.LightningModule):
    def __init__(
        self,
        net: Optional[NetType] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[OptimizerType] = None,
        lr_scheduler: Optional[LrSchedulerType] = None,
        metrics: Optional[MetricType] = None,
        loss_log_key: Optional[str] = "loss",
        log_metrics: bool = True,
        exclude_no_grad: bool = True,
        disable_prog_bar: bool = False,
    ):
        """
        A lightweight wrapper around `LightningModule` that automates model, criterion, optimizer, scheduler,
        and metric creation using a simple interface.

        Key Features:
        - `net`: main model or container module (e.g. `ModuleList`, `ModuleDict`)
        - `criterion`: loss function
        - `metrics`: optional dict of metric functions or metric config dicts
        - `shared_step(batch, batch_idx, phase)`: user-implemented logic for a single step
        - `shared_logged_step(phase, ...)`: wraps `shared_step`, computes loss, logs loss and metrics
        - `configure_optimizers()`: supports single/multiple/dict/list optimizer and scheduler configurations

        Automatically implements:
        - `training_step`, `validation_step`, `test_step`, `predict_step` → delegate to `shared_logged_step`
        - Optimizer and LR scheduler setup via `register_optimizer` and `configure_optimizers`

        Args:
            net: Model or module container
            criterion: Loss function
            optimizer: Optimizer instance or callable or list/dict of such
            lr_scheduler: Scheduler instance, callable, or scheduler config dict
            metrics: Dict of metric functions or config dicts
            loss_log_key: Log key for loss (e.g. "loss", "nll", etc.)
            log_metrics: Whether to log metrics
            exclude_no_grad: Whether to exclude non-trainable parameters from optimizer
            disable_prog_bar: If True, disables progress bar updates during validation
        """

        super().__init__()

        self.net = net
        self.criterion = criterion
        self.optimizers_schedulers = {}
        self.metrics = {} if metrics is None else metrics

        self.register_optimizer(self, optimizer, lr_scheduler)

        self.metrics = self.configure_metrics() | self.metrics
        self.register_torchmetrics()

        self.loss_log_key = loss_log_key
        self.log_metrics = log_metrics

        self.exclude_no_grad = exclude_no_grad

        self.disable_prog_bar = disable_prog_bar

        self.save_hyperparameters(ignore=KEYS_TO_IGNORE)

    def parameters_for_optimizer(self, recurse: bool = True) -> Iterator[Parameter]:
        params = self.parameters(recurse)

        if self.exclude_no_grad:
            for param in params:
                if param.requires_grad:
                    yield param
        else:
            yield from params

    def register_optimizer(
        self,
        module: nn.Module,
        optimizer: Optional[OptimizerType] = None,
        lr_scheduler: Optional[LrSchedulerType] = None,
    ):
        if optimizer is not None:
            if module in self.optimizers_schedulers:
                warnings.warn(
                    f"Optimizer for module '{module}' already exists in optimizers_schedulers. Overwriting it."
                )

            self.optimizers_schedulers[module] = (optimizer, lr_scheduler)
        elif lr_scheduler is not None:
            raise ValueError("Cannot register a scheduler when the optimizer is None")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizers = []
        schedulers = []

        for module, (optimizer, scheduler) in self.optimizers_schedulers.items():
            # Single initialized optimizer, with optional scheduler
            if isinstance(optimizer, optim.Optimizer):
                optimizers.append(optimizer)

                if scheduler is not None:
                    schedulers.append(_get_scheduler(scheduler, optimizer))
            # Callable that returns an optimizer instance, with optional scheduler
            elif callable(optimizer):
                params = self.parameters_for_optimizer() if module == self else module.parameters()
                optimizers.append(optimizer(params))

                if scheduler is not None:
                    schedulers.append(_get_scheduler(scheduler, optimizers[-1], should_be_callable=True))
            else:
                raise TypeError(f"Invalid optimizer type: {type(optimizer)}")

        # Format return value according to Lightning's expectations.
        # See [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers)
        # for return values allowed by Lightning

        if schedulers == []:
            if optimizers == []:
                return None

            if len(optimizers) == 1:
                return optimizers[0]

            return optimizers

        if optimizers == []:
            raise ValueError("Schedulers were specified but no optimizers were provided")

        if len(optimizers) == 1 and len(schedulers) == 1:
            return {"optimizer": optimizers[0], "lr_scheduler": schedulers[0]}

        return optimizers, schedulers

    def configure_metrics(self) -> MetricType:
        return {}
    
    def register_torchmetrics(self):
        # It is necessary to have all torch metrics instances registered as sub NN modules
        # in order for the .update calls on the metrics to work without errors

        torch_metrics = {}

        for name, metric in self.metrics.items():
            if isinstance(metric, Metric):
                torch_metrics[name] = metric

        if torch_metrics != {}:
            self.torchmetrics = nn.ModuleDict(torch_metrics)

    def should_enable_prog_bar(self, phase: Phase):
        if self.disable_prog_bar:
            return False

        return phase == "val"

    def shared_step(self, phase: Phase, *args, **kwargs):
        """A call to shared_step should result in either:

        - a single loss value
        - a tuple/list of inputs for the loss function
        - a dict containing "loss" and "metric" keys
            where "loss" is the loss value or a tuple/list of inputs for the loss function
            and "metric" is a dict containing the metric values, inputs to the metric function
            or a single value that is passed to all metrics
        """

        raise NotImplementedError

    def shared_logged_step(self, phase: Phase, *args: Any, **kwargs: Any):
        # step_out can be:
        # - a tuple/iterable, all values of which will be fed into the loss function
        #   and that can be used for all metric computation
        # - a dictionary with two keys: "loss" and (optionally) "metric_args" that is a Dict of the metric name with the args for the metric function
        # - a dictionary with two keys: "criterion_args" and (optionally) "metric_args" that is a Dict of the metric name with the args for the metric function
        # - a torch tensor (the loss was already computed)
        # - None

        step_out = self.shared_step(phase, *args, **kwargs)

        default_log_kwargs: Dict[str, Any] = dict(prog_bar=self.should_enable_prog_bar(phase))
        loss = None

        if isinstance(step_out, (tuple, list)):
            if isinstance(step_out, tuple):
                loss = self.criterion(*step_out)
            else:
                loss = self.criterion(step_out)

            # TODO: maybe this functionality should be removed???
            for name, metric in self.metrics.items():
                metric_func, metric_specific_log_kwargs = _resolve_metric(metric, default_log_kwargs)

                if isinstance(metric_func, Metric):
                    metric_func.to(device=step_out[0].device)

                    if phase != "train":
                        metric_func.update(*step_out)
                        metric_val = metric_func
                    else:
                        metric_val = metric_func(*step_out)
                else:
                    metric_val = metric_func(*step_out)

                self.log(f"{phase}/{name}", metric_val, **metric_specific_log_kwargs)
        elif isinstance(step_out, dict):
            loss_computed = "loss" in step_out
            criterion_args_provided = "criterion_args" in step_out

            if loss_computed and criterion_args_provided:
                raise ValueError("Cannot have both 'loss' and 'criterion_args' in step_out")
            elif not loss_computed and not criterion_args_provided:
                raise ValueError("Either 'loss' or 'criterion_args' must be provided in step_out")
            elif loss_computed:
                loss = step_out["loss"]
            else:
                loss = _call_with_flexible_args(self.criterion, step_out["criterion_args"])

            curr_step_log_kwargs = default_log_kwargs | step_out.get("log_kwargs", {})
            metrics_to_log = []  # Store in list to avoid duplicate keys in the log by checking list before logging

            if "metric_args" in step_out:
                for name, args in step_out["metric_args"].items():
                    metric_func, metric_specific_log_kwargs = _resolve_metric(self.metrics[name], curr_step_log_kwargs)

                    if isinstance(metric_func, Metric):
                        metric_func.to(device=step_out[0].device)

                    metric_val = _call_with_flexible_args(metric_func, args)

                    metrics_to_log.append((f"{phase}/{name}", (metric_val, metric_specific_log_kwargs)))

            if "computed_metrics" in step_out:
                for name, val in step_out["computed_metrics"].items():
                    metric_val, metric_specific_log_kwargs = _resolve_metric(val, curr_step_log_kwargs)
                    metrics_to_log.append((f"{phase}/{name}", (metric_val, metric_specific_log_kwargs)))

            # TODO simplify this logic
            # ========================================
            # Prioritize computed_metrics over derived metrics
            from collections import Counter

            dup_keys = [k for k, c in Counter(k for k, _ in metrics_to_log).items() if c > 1]

            for dup in dup_keys:
                warnings.warn(f"Duplicate metric key '{dup}' found. Only pre-computed value will be logged.")
            # ========================================

            for key, (val, metric_kwargs) in dict(metrics_to_log).items():
                self.log(key, val, **metric_kwargs)
        else:
            loss = step_out

        # TODO add support for custom loss logging kwargs

        if self.loss_log_key and loss is not None:
            self.log(f"{phase}/{self.loss_log_key}", loss, **default_log_kwargs)

        return loss

    def training_step(self, *args: Any, **kwargs: Any):
        return self.shared_logged_step("train", *args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any):
        return self.shared_logged_step("val", *args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any):
        return self.shared_logged_step("test", *args, **kwargs)
