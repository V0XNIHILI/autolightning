from typing import Dict, List, Optional

from itertools import repeat

import torch.nn as nn
import torch.optim as optim

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from torch_mate.utils import get_class_and_init, get_modules


class AutoModule(L.LightningModule):
    log_separator = "/"
    log_metrics = {}
    log_phases = {"train": "train", "val": "val", "test": "test", "predict": "predict"}
    log_order = "phase-metric" # or "metric-phase"

    def __init__(self, cfg: Dict):
        """Lightweight wrapper around PyTorch Lightning LightningModule that adds support for a configuration dictionary.
        Based on this configuration, it creates the model, criterion, optimizer, and scheduler. Overall, compared to the
        PyTorch Lightning LightningModule, the following three attributes are added:
        
        - `self.get_model()`: the created model
        - `self.criteria`: the created criterion
        - `self.shared_step(self, batch, batch_idx, phase)`: a generic step function that is shared across all steps (train, val, test, predict)

        Based on these, the following methods are automatically implemented:

        - `self.forward(self, x)`: calls `self.model(x)`
        - `self.training_step(self, batch, batch_idx)`: calls `self.shared_step(batch, batch_idx, "train")`
        - `self.validation_step(self, batch, batch_idx)`: calls `self.shared_step(batch, batch_idx, "val")`
        - `self.test_step(self, batch, batch_idx)`: calls `self.shared_step(batch, batch_idx, "test")`
        - `self.predict_step(self, batch, batch_idx)`: calls `self.shared_step(batch, batch_idx, "predict")`
        - `self.configure_optimizers(self)`: creates the optimizer and scheduler based on the configuration dictionary

        You can also override `self.config_model(self)`, `self.configure_criteria(self)` and `self.configure_configuration(self, cfg: Dict)` to customize the model and criterion creation and the hyperparameter setting.

        Args:
            cfg (Dict): configuration dictionary
        """

        super().__init__()

        self.save_hyperparameters(self.configure_configuration(cfg))

        self._model = self.compile_model(self.config_model())
        self.criteria = self.configure_criteria()

    def log_key(self, phase: str, metric: str, metric_postfix: str = None):
        metric_key = self.log_metrics[metric]
        phase_key = self.log_phases[phase]

        if metric_postfix:
            metric_key += metric_postfix

        if self.log_order == "phase-metric":
            return f"{phase_key}{self.log_separator}{metric_key}"
        elif self.log_order == "metric-phase":
            return f"{metric_key}{self.log_separator}{phase_key}"
        
        raise ValueError(f"Invalid log order: {self.log_order}; must be either 'phase-metric' or 'metric-phase'")

    def configure_configuration(self, cfg: Dict):
        return cfg

    def config_model(self):
        return get_modules(None, self.hparams.model)
    
    def compile_model(self, model: nn.Module) -> nn.Module:
        compile_cfg = self.hparams.model.get("extra", {}).get("compile", None)

        if compile_cfg:
            return get_class_and_init(None, compile_cfg, model)

        return model
    
    def get_model(self, *args):
        if len(args) == 0:
            return self._model
        
        if len(args) == 1 and type(args[0]) is list:
            return [getattr(self._model, name) if type(name) is str else self._model[name] for name in args[0]]
        
        models = [getattr(self._model, name) if type(name) is str else self._model[name] for name in args]
        return models[0] if len(models) == 1 else models

    def configure_criteria(self):
        return get_modules(nn, self.hparams.criterion)
    
    def configure_optimizers_only(self) -> List[optim.Optimizer]:
        opt_configs = self.hparams.optimizer if type(self.hparams.optimizer) is list else [self.hparams.optimizer]
        return [get_class_and_init(optim, opt_cfg, self.get_model().parameters()) for opt_cfg in opt_configs]
    
    def configure_schedulers(self, optimizers: List[optim.Optimizer]) -> Optional[List[optim.lr_scheduler._LRScheduler]]:
        schedulers = None

        if "lr_scheduler" in self.hparams:
            sched_configs = self.hparams.lr_scheduler if type(self.hparams.lr_scheduler) is list else [self.hparams.lr_scheduler]
            optimizers_for_schedulers = optimizers if len(sched_configs) > 1 else repeat(optimizers[0], len(sched_configs))
            schedulers = [get_class_and_init(optim.lr_scheduler, sched_cfg["scheduler"], optimizer) for sched_cfg, optimizer in zip(sched_configs, optimizers_for_schedulers)]
            schedulers = [{"scheduler": scheduler} for scheduler in schedulers]

            for i, sched_cfg in enumerate(sched_configs):
                for key in sched_cfg:
                    if key != "scheduler":
                        schedulers[i][key] = sched_cfg[key]

        return schedulers

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizers = self.configure_optimizers_only()
        schedulers = self.configure_schedulers(optimizers)
      
        # See [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers)
        # for return values allowed by Lightning
        if schedulers is None:
            if len(optimizers) == 1:
                return optimizers[0]
            
            return optimizers
        
        if len(optimizers) == 1 and len(schedulers) == 1:
            return {"optimizer": optimizers[0], "lr_scheduler": schedulers[0]}

        return optimizers, schedulers
    
    def forward(self, x):
        return self.get_model()(x)
    
    def shared_step(self, batch, batch_idx, phase: str):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        # TODO: make sure that predict step works even when there are no labels
        return self.shared_step(batch, batch_idx, "predict")
