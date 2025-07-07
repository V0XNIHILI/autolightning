from typing import Dict, Optional, Any, Iterator
import warnings

import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from lightning.pytorch.cli import OptimizerCallable

from .types import MetricType, OptimizerType, LrSchedulerType, Phase


LOG_PHASE_KEYS = {"train", "val", "test", "predict"}
LOG_ORDER_OPTIONS = {"phase_first", "metric_first"}
KEYS_TO_IGNORE = ["net", "criterion", "metrics", "optimizer", "compiler", "metrics", "loss_log_key", "log_metrics"]


class AutoModule(L.LightningModule):
    def __init__(self,
                 net: Optional[nn.Module] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[OptimizerType] = None,
                 lr_scheduler: Optional[LrSchedulerType] = None,
                 metrics: Optional[MetricType] = None,
                 loss_log_key: Optional[str] = "loss",
                 log_metrics: bool = True,
                 exclude_no_grad: bool = True,
                 disable_prog_bar: bool = False):
        """Lightweight wrapper around PyTorch Lightning LightningModule that adds support for a configuration dictionary.
        Based on this configuration, it creates the model, criterion, optimizer, and scheduler. Overall, compared to the
        PyTorch Lightning LightningModule, the following three attributes are added:
        
        - `self.criterion`: the created criterion
        - `self.shared_step(self, batch, batch_idx, phase)`: a generic step function that is shared across all steps (train, val, test, predict)

        Based on these, the following methods are automatically implemented:

        - `self.training_step(self, batch, batch_idx)`: calls `self.shared_step(batch, batch_idx, "train")`
        - `self.validation_step(self, batch, batch_idx)`: calls `self.shared_step(batch, batch_idx, "val")`
        - `self.test_step(self, batch, batch_idx)`: calls `self.shared_step(batch, batch_idx, "test")`
        - `self.predict_step(self, batch, batch_idx)`: calls `self.shared_step(batch, batch_idx, "predict")`
        - `self.configure_optimizers(self)`: creates the optimizer and scheduler based on the configuration dictionary

        Args:
            net (Optional[CriterionNetType], optional): _description_. Defaults to None.
            criterion (Optional[CriterionNetType], optional): _description_. Defaults to None.
            optimizer (Optional[OptimizerType], optional): _description_. Defaults to None.
            compiler (Optional[Callable], optional): A dict describing the compiler config or callable that compiles a net. Defaults to None.
            metrics (Optional[MetricType], optional): _description_. Defaults to None
        """

        super().__init__()

        self.net = net
        self.criterion = criterion
        self.optimizers_schedulers = {}
        self.metrics = {} if metrics == None else metrics

        self.register_optimizer(self, optimizer, lr_scheduler)

        self.metrics = self.metrics | self.configure_metrics()

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

    def register_optimizer(self, module: nn.Module, optimizer: Optional[OptimizerCallable] = None, lr_scheduler: Optional[LrSchedulerType] = None):
        if optimizer != None:
            if module in self.optimizers_schedulers:
                warnings.warn(f"Optimizer for module '{module}' already exists in optimizers_schedulers. Overwriting it.")

            self.optimizers_schedulers[module] = (optimizer, lr_scheduler)
        elif lr_scheduler != None:
            raise ValueError("Cannot register a scheduler when the optimizer is None")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizers = []
        schedulers = []

        # Check the following loop for each attribute name that ends with "optimizer" or "opt"
        # Find the corresponding module by removing the "optimizer" or "opt" suffix
        # If the module name is "", then use the module name "net"
        # If instance of Optimizer, then return the optimizer
        # If instance of Callable, then call the callable with the parameters
        # If list or tuple:
        #     - If all elements are instances of Optimizer, then return the list
        #     - If all elements are instances of Callable, then call each callable with the parameters
        # If dict:
        #    - check if .net is a module dict, then assign each optimizer to the corresponding module

        for module, (optimizer, scheduler) in self.optimizers_schedulers.items():
            if isinstance(optimizer, optim.Optimizer):
                optimizers.append(optimizer)

                if scheduler != None:
                    if isinstance(scheduler, optim.lr_scheduler.LRScheduler):
                        schedulers.append(scheduler)
                    elif callable(scheduler):
                        schedulers.append(scheduler(optimizers[-1]))
                    elif isinstance(scheduler, dict):
                        sched = scheduler["scheduler"]

                        if callable(sched):
                            sched_inst = sched(optimizers[-1])
                        else:
                            sched_inst = sched

                        init_sched = {key: value for key, value in scheduler.items() if key != "scheduler"}
                        init_sched["scheduler"] = sched_inst

                        schedulers.append(init_sched)
                    else:
                        raise TypeError(f"Invalid scheduler type: {type(scheduler)}; expected either a scheduler or a callable")
            elif callable(optimizer):
                params = self.parameters_for_optimizer() if module == self else module.parameters()
                optimizers.append(optimizer(params))

                if scheduler != None:
                    if callable(scheduler):
                        schedulers.append(scheduler(optimizers[-1]))
                    elif isinstance(scheduler, dict):
                        sched = scheduler["scheduler"]

                        assert callable(sched), f"Scheduler for module '{module}' must be a callable"

                        sched_inst = sched(optimizers[-1])

                        init_sched = {key: value for key, value in scheduler.items() if key != "scheduler"}
                        init_sched["scheduler"] = sched_inst

                        schedulers.append(init_sched)
                    else:
                        raise TypeError(f"Invalid scheduler type: {type(scheduler)}; expected a callable")
            elif isinstance(optimizer, (list, tuple)):
                assert scheduler == None, "Cannot use a list of optimizers with a scheduler"

                if all(isinstance(opt, optim.Optimizer) for opt in optimizer):
                    optimizers.extend(optimizer)
                elif all(callable(opt) for opt in optimizer):
                    if isinstance(module, nn.ModuleList):
                        extra_optimizers = [opt(module[i].parameters()) for i, opt in enumerate(optimizer)]
                    else:
                        raise ValueError(f"Cannot use list of optimizers with non-ModuleList module: {module}")

                    optimizers.extend(extra_optimizers)
                else:
                    raise TypeError(f"Invalid optimizer type: {type(optimizer)}")
            elif isinstance(optimizer, dict):
                assert scheduler == None, "Cannot use a dict of optimizers with a scheduler"

                if isinstance(module, nn.ModuleDict):
                    for key in optimizer:
                        assert callable(optimizer[key]), f"Optimizer for key '{key}' must be a callable"

                        optimizers.append(optimizer[key](module[key].parameters()))
                else:
                    raise ValueError(f"Cannot use optimizer dict with non-ModuleDict module: {module}")
            else:
                raise TypeError(f"Invalid optimizer type: {type(optimizer)}")

        if schedulers == []:
            if optimizers == []:
                return None
            
            if len(optimizers) == 1:
                return optimizers[0]
            
            return optimizers
        
        if optimizers == []:
            raise ValueError("Schedulers were specified but no optimizers were provided")
      
        # See [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers)
        # for return values allowed by Lightning
        if len(optimizers) == 1 and len(schedulers) == 1:
            return {"optimizer": optimizers[0], "lr_scheduler": schedulers[0]}

        return optimizers, schedulers

    def configure_metrics(self) -> Dict[str, MetricType]:
        return {}

    def should_enable_prog_bar(self, phase: Phase):
        if self.disable_prog_bar:
            return False

        return phase == 'val'

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """The forward step in an `AutoModule` ideally implements ONLY the forward pass through the network,
        returning the output of the network that can directly be fed into the criterion.
        """

        raise NotImplementedError
    
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
        # - a dictionary with two keys: "loss" and (optionally) "metrics_args" that is a Dict of the metric name with the args for the metric function
        # - a dictionary with two keys: "criterion_args" and (optionally) "metrics_args" that is a Dict of the metric name with the args for the metric function
        # - a torch tensor (the loss was already computed)
        # - None
        step_out = self.shared_step(phase, *args, **kwargs)

        loss = None
        metrics_args = {}
        log_kwargs = {}
        log_dict = {}

        if isinstance(step_out, (tuple, list)):
            loss = self.criterion(*step_out)
            metrics_args = step_out
        elif isinstance(step_out, dict):
            loss_computed = "loss" in step_out
            criterion_args_provided = "criterion_args" in step_out

            if loss_computed and criterion_args_provided:
                raise ValueError("Cannot have both 'loss' and 'criterion_args' in step_out dictionary")

            if loss_computed:
                loss = step_out["loss"]
            elif criterion_args_provided:
                loss = self.criterion(*step_out["criterion_args"])
    
            log_kwargs = step_out.get("log_kwargs", {})
            log_dict = step_out.get("log_dict", {})
            metrics_args = step_out.get("metrics_args", step_out.get("metric_inputs", {}))
        else:
            loss = step_out

        prog_bar = self.should_enable_prog_bar(phase)
        main_log_kwargs = dict(prog_bar = prog_bar) | log_kwargs

        if self.loss_log_key != None and loss != None:
            self.log_dict({
                f"{phase}/{self.loss_log_key}": loss,
                **{f"{phase}/{key}": value for key, value in log_dict.items()}
            }, **main_log_kwargs)

        if isinstance(metrics_args, dict):
            for name, inputs in metrics_args.items():
                key = f"{phase}/{name}"

                metric = self.metrics.get(name, self.metrics[name])

                if isinstance(metric, dict):
                    self.log(key, metric["metric"](*inputs), **(dict(prog_bar = prog_bar) | metric["log_kwargs"]))
                else:
                    self.log(key, metric(*inputs), **main_log_kwargs)
        else: # if metric_inputs is a tuple/list, use it for all metrics
            for name, metric in self.metrics.items():
                key = f"{phase}/{name}"

                if isinstance(metric, dict):
                    self.log(key, metric["metric"](*metrics_args), **metric["log_kwargs"])
                else:
                    self.log(key, metric(*metrics_args), **main_log_kwargs)

        return loss
    
    def training_step(self, *args: Any, **kwargs: Any):
        return self.shared_logged_step("train", *args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any):
        return self.shared_logged_step("val", *args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any):
        return self.shared_logged_step("test", *args, **kwargs)
