from typing import Dict, List, Optional, Any, Iterator, Union, Callable, Tuple
import warnings

import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics.metric import Metric

from .types import MetricType, OptimizerType, LrSchedulerType, NetType, Phase, PHASES


LOG_PHASE_KEYS = {"train", "val", "test", "predict"}
LOG_ORDER_OPTIONS = {"phase_first", "metric_first"}

# `nn.ModuleDict` keys go through `add_module`, which rejects names that shadow existing
# attributes -- "train" would collide with `nn.Module.train`. Prefix the phase keys.
PHASE_MODULE_KEY = "phase_{}".format

KEYS_TO_IGNORE = [
    "net",
    "criterion",
    "metrics",
    "optimizer",
    "loss_log_key",
    "log_metrics",
]


def _call_with_flexible_args(func: Callable, args: Any) -> Any:
    """Call `func` with `args`, using the calling convention implied by the container type.

    NOTE: tuples and lists are NOT interchangeable here:

    - `tuple` -> `func(*args)`     : each element becomes a separate positional argument
    - `dict`  -> `func(**args)`    : each item becomes a keyword argument
    - `list`  -> `func(args)`      : the list is passed as a SINGLE positional argument

    Use a tuple unless the callee genuinely expects one sequence argument
    (e.g. a loss over a variable-length list of tensors).
    """
    if isinstance(args, tuple):
        return func(*args)
    if isinstance(args, list):
        return func(args)
    if isinstance(args, dict):
        return func(**args)
    raise TypeError(f"Invalid argument type: {type(args)}")


def _resolve_metric(metric, default_log_kwargs: Dict[str, Any]) -> Tuple[Union[Callable, Any], Dict[str, Any]]:
    metric_func_or_value = metric
    metric_specific_log_kwargs = default_log_kwargs

    if isinstance(metric, dict):
        metric_func_or_value = metric["metric"]
        metric_specific_log_kwargs = default_log_kwargs | metric.get("log_kwargs", {})

    return metric_func_or_value, metric_specific_log_kwargs


def _unpack_metric_entry(entry: Any):
    """Split a metric registry entry into (metric, log_kwargs, phases).

    An entry is either the bare metric (a `Metric` instance or any callable), or a dict
    of the form `{"metric": ..., "log_kwargs": {...}, "phases": ("val", "test")}`.
    `phases` restricts the metric to a subset of phases and defaults to all of them.
    """
    if isinstance(entry, dict):
        phases = tuple(entry.get("phases", PHASES))

        for phase in phases:
            if phase not in PHASES:
                raise ValueError(f"Invalid phase '{phase}'; expected one of {PHASES}")

        return entry["metric"], entry.get("log_kwargs", {}), phases

    return entry, {}, PHASES


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


def _compute_metric(
    metric: Union[Metric, Callable[..., Any]],
    metric_input: Union[Tuple, List, Dict],
    log_kwargs: Optional[Dict[str, Any]] = None,
):
    """Update a stateful metric (returning the metric object itself, so that Lightning owns
    the compute/reset lifecycle) or call a stateless metric and return its value.

    `update()` vs `forward()` matters here. Lightning's `_ResultMetric` reads a logged `Metric`
    through `metric._forward_cache` when `on_step=True`, and only calls `compute()` when
    `on_epoch=True`. torchmetrics fills `_forward_cache` in `forward()` (i.e. `metric(...)`),
    NOT in `update()` -- so an `update()`-only metric logged with `on_step=True` silently
    produces nothing. Use `forward()` whenever step-level logging was actually requested.

    Stateful metrics are not moved to the input device here: they live in `self._metrics`,
    which is an `nn.ModuleDict` and therefore moved along with the LightningModule.
    """
    if isinstance(metric, Metric):
        if (log_kwargs or {}).get("on_step", False):
            # Only reachable when step-level logging was explicitly requested; see
            # `default_log_kwargs` in `shared_logged_step`, which makes on_step/on_epoch explicit
            # so this check can never disagree with what Lightning will actually do.
            # forward() updates the global state *and* returns/caches the batch value
            _call_with_flexible_args(metric, metric_input)
        else:
            _call_with_flexible_args(metric.update, metric_input)

        return metric

    return _call_with_flexible_args(metric, metric_input)


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

        Metrics may be given either as a bare metric (any callable, or a `torchmetrics.Metric`
        instance) or as a config dict::

            metrics = {
                "acc": Accuracy(task="multiclass", num_classes=10),
                "auroc": {
                    "metric": AUROC(task="multiclass", num_classes=10),
                    "log_kwargs": {"on_step": False, "on_epoch": True},
                    "phases": ("val", "test"),   # skip during training
                },
            }

        Stateful (`torchmetrics.Metric`) entries are cloned once per phase at construction time,
        so train/val/test each accumulate into their own buffers and cannot leak into one another.
        Stateless callables carry no state and are shared across phases.
        """

        super().__init__()

        self.net = net
        self.criterion = criterion
        self.optimizers_schedulers = {}

        self.exclude_no_grad = exclude_no_grad

        self.register_optimizer(self, optimizer, lr_scheduler)

        # `self.metrics` is kept as the raw, un-cloned user specification (for introspection).
        # The objects actually used during a step live in `self._metrics`.
        self.metrics = self.configure_metrics() | ({} if metrics is None else metrics)
        self._build_per_phase_metrics(self.metrics)

        self.loss_log_key = loss_log_key
        self.log_metrics = log_metrics

        self.disable_prog_bar = disable_prog_bar

        self.save_hyperparameters(ignore=KEYS_TO_IGNORE)

    def parameters_for_optimizer(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from self.module_parameters_for_optimizer(self, recurse)

    def module_parameters_for_optimizer(self, module: nn.Module, recurse: bool = True) -> Iterator[Parameter]:
        params = module.parameters(recurse)

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
        """Attach an optimizer (and optionally a scheduler) to `module`.

        NOTE: registering a module here does NOT add it to the module tree. `optimizers_schedulers`
        keys are only used to resolve parameters at `configure_optimizers` time. Any module passed
        here must also be reachable as an attribute (or inside a registered container such as
        `nn.ModuleList`/`nn.ModuleDict`) of this `LightningModule`, otherwise its parameters will
        not be moved to the accelerator, will not appear in `state_dict()`, and will not be synced
        under DDP.
        """
        if optimizer is not None:
            if module in self.optimizers_schedulers:
                warnings.warn(
                    f"Optimizer for module '{module}' already exists in optimizers_schedulers. Overwriting it."
                )

            self.optimizers_schedulers[module] = (optimizer, lr_scheduler)
        elif lr_scheduler is not None:
            raise ValueError("Cannot register a scheduler when the optimizer is None")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # One config dict per registered optimizer, so that each scheduler stays explicitly
        # paired with its own optimizer. Returning `(optimizers, schedulers)` instead would make
        # Lightning pair the two lists positionally, which silently mismatches as soon as one
        # optimizer has a scheduler and another does not.
        # See [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers)
        # for return values allowed by Lightning
        configs: List[Dict[str, Any]] = []

        for module, (optimizer, scheduler) in self.optimizers_schedulers.items():
            # Single initialized optimizer, with optional scheduler
            if isinstance(optimizer, optim.Optimizer):
                opt_inst = optimizer
                sched_inst = None if scheduler is None else _get_scheduler(scheduler, opt_inst)
            # Callable that returns an optimizer instance, with optional scheduler
            elif callable(optimizer):
                opt_inst = optimizer(self.module_parameters_for_optimizer(module))
                sched_inst = None if scheduler is None else _get_scheduler(scheduler, opt_inst, should_be_callable=True)
            else:
                raise TypeError(f"Invalid optimizer type: {type(optimizer)}")

            config: Dict[str, Any] = {"optimizer": opt_inst}

            if sched_inst is not None:
                config["lr_scheduler"] = sched_inst

            configs.append(config)

        if configs == []:
            return None

        if len(configs) == 1:
            if "lr_scheduler" not in configs[0]:
                return configs[0]["optimizer"]
            return configs[0]

        return configs

    def configure_metrics(self) -> MetricType:
        return {}

    def _build_per_phase_metrics(self, metrics: MetricType) -> None:
        """Materialise the metric registry once per phase.

        A `torchmetrics.Metric` is stateful: `update()` accumulates into buffers owned by that
        specific instance. Sharing one instance across phases means validation batches land in the
        same buffers the training epoch is accumulating into (validation runs *inside* the training
        epoch), and whichever phase resets first wipes the other's state. Giving each phase its own
        clone makes that structurally impossible. Stateless callables have nothing to leak, so they
        are shared rather than copied.
        """
        self._metric_log_kwargs: Dict[Tuple[Phase, str], Dict[str, Any]] = {}
        self._metric_names: Dict[Phase, List[str]] = {phase: [] for phase in PHASES}
        self._stateless_metrics: Dict[Phase, Dict[str, Callable]] = {phase: {} for phase in PHASES}
        stateful: Dict[Phase, Dict[str, Metric]] = {phase: {} for phase in PHASES}

        for name, entry in metrics.items():
            metric, log_kwargs, phases = _unpack_metric_entry(entry)

            for phase in phases:
                self._metric_names[phase].append(name)

                if isinstance(metric, Metric):
                    self._metric_log_kwargs[(phase, name)] = log_kwargs
                else:
                    self._metric_log_kwargs[(phase, name)] = log_kwargs

                if isinstance(metric, Metric):
                    stateful[phase][name] = metric.clone()
                else:
                    self._stateless_metrics[phase][name] = metric

        # Registering the clones as submodules is what gives them device placement and DDP sync.
        self._metrics = nn.ModuleDict(
            {PHASE_MODULE_KEY(phase): nn.ModuleDict(named) for phase, named in stateful.items() if named != {}}
        )

    def metrics_for(self, phase: Phase) -> Dict[str, Union[Metric, Callable]]:
        """The metric instances belonging to `phase`, keyed by name, in registration order.

        Stateful metrics are read live from `self._metrics` rather than from a cached dict,
        so that a deep-copied module (EMA, SWA, ...) resolves to its own clones.
        """
        key = PHASE_MODULE_KEY(phase)
        stateful = self._metrics[key] if key in self._metrics else {}
        stateless = self._stateless_metrics[phase]

        return {
            name: (stateful[name] if name in stateful else stateless[name])
            for name in self._metric_names[phase]
        }

    def log_kwargs_for(self, phase: Phase, name: str, base: Dict[str, Any]) -> Dict[str, Any]:
        return base | self._metric_log_kwargs.get((phase, name), {})

    def should_enable_prog_bar(self, phase: Phase):
        if self.disable_prog_bar:
            return False

        return phase == "val"

    def shared_step(self, phase: Phase, *args, **kwargs):
        """A call to shared_step should result in either:

        - a single loss value
        - a tuple/list of inputs for the loss function
        - a dict containing ("loss" OR "criterion_args"), and optionally "metric_args", "log_kwargs" and "metric_values" keys
            where "loss" is the loss value or a tuple/list of inputs for the loss function
            and "metric_values" is a dict containing the metric values, "metric_args" inputs to the metric function
            or a single value that is passed to all metrics

        NOTE on tuples vs lists: a tuple is splatted (`criterion(*step_out)`) while a list is passed
        as one argument (`criterion(step_out)`). Return a tuple for the usual `(y_hat, y)` case.
        See `_call_with_flexible_args`.
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

        # Spell out Lightning's per-hook defaults (training_step: on_step=True/on_epoch=False,
        # everything else: the reverse) instead of leaving them implicit. `_compute_metric` has to
        # know whether Lightning will read `metric._forward_cache` or call `metric.compute()`, and
        # an absent key would make it guess wrong. Can se seen in 
        # pytorch-lightning/src/lightning/pytorch/trainer/connectors/logger_connector/fx_validator.py,
        # search for "training_step".
        default_log_kwargs: Dict[str, Any] = dict(
            prog_bar=self.should_enable_prog_bar(phase),
        )

        if phase != 'predict':
            default_log_kwargs.update(
                on_step=phase == "train",
                on_epoch=phase != "train",
            )

        loss = None

        if isinstance(step_out, (tuple, list)):
            loss = _call_with_flexible_args(self.criterion, step_out)

            # Compute all the metrics registered for this phase using the same step_out as input,
            # and log them with their respective log kwargs (if provided)
            for name, metric in self.metrics_for(phase).items():
                metric_log_kwargs = self.log_kwargs_for(phase, name, default_log_kwargs)
                metric_val = _compute_metric(metric, step_out, metric_log_kwargs)
                self.log(f"{phase}/{name}", metric_val, **metric_log_kwargs)
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
            phase_metrics = self.metrics_for(phase)

            if "metric_args" in step_out:
                for name, args in step_out["metric_args"].items():
                    if name not in phase_metrics:
                        # Metric is registered but excluded from this phase via its "phases" key
                        continue

                    metric_specific_log_kwargs = self.log_kwargs_for(phase, name, curr_step_log_kwargs)
                    metric_val = _compute_metric(phase_metrics[name], args, metric_specific_log_kwargs)
                    metrics_to_log.append((f"{phase}/{name}", (metric_val, metric_specific_log_kwargs)))

            if "metric_values" in step_out:
                for name, val in step_out["metric_values"].items():
                    metric_val, metric_specific_log_kwargs = _resolve_metric(val, curr_step_log_kwargs)
                    metrics_to_log.append((f"{phase}/{name}", (metric_val, metric_specific_log_kwargs)))

            # Prioritize metric_values over derived metrics
            from collections import Counter

            dup_keys = [k for k, c in Counter(k for k, _ in metrics_to_log).items() if c > 1]

            for dup in dup_keys:
                warnings.warn(f"Duplicate metric key '{dup}' found. Only pre-computed value will be logged.")

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
