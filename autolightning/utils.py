from typing import Any, Optional, Dict, List
from collections import OrderedDict
from functools import partial
import importlib

import torch
import torch.nn as nn

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from pytorch_lightning.cli import instantiate_class


LIGHTNING_STATE_DICT_KEYS = [
    "epoch",
    "global_step",
    "pytorch-lightning_version",
    "state_dict",
    "loops",
    "callbacks",
    "optimizer_states",
    "lr_schedulers",
    "hparams_name",
    "hyper_parameters"
]


def _import_module(module_path: str, default_module: Optional[str] = None) -> Any:
    split = module_path.rsplit(".")

    if len(split) == 1:
        module_name = default_module
        function_name = split[0]
    else:
        module_name, function_name = split

    function_module = importlib.import_module(module_name)
    function = getattr(function_module, function_name)

    return function


def load(module: nn.Module, file_path: str, submodule_path: Optional[str], strict: bool = True, assign: bool = False, **kwargs: Any) -> nn.Module:
    state_dict = torch.load(file_path, **kwargs)

    # Check if the state dict is a Lightning state dict
    if all(key in state_dict for key in LIGHTNING_STATE_DICT_KEYS):
        state_dict = state_dict["state_dict"]

    if submodule_path is not None:
        new_state_dict = {}
        
        # Make sure only the sub state dict at the submodule_path is loaded
        for key in state_dict:
            if key.startswith(submodule_path):
                new_state_dict[key[len(submodule_path) + 1:]] = state_dict[key]

        state_dict = OrderedDict(new_state_dict)
    
    module.load_state_dict(state_dict, strict=strict, assign=assign)
    
    return module


def compile(module: nn.Module, compiler_path: str, compiler_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
    function = _import_module(compiler_path)

    return function(module, **(compiler_kwargs if compiler_kwargs != None else {}))


def sequential(modules: List[nn.Module]) -> nn.Sequential:
    return nn.Sequential(*modules) if modules else nn.Sequential()


try:
    from transformers import PretrainedConfig
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    def from_transformers_config(
        auto_model: str,
        config: PretrainedConfig,
        auto_model_kwargs: Optional[Dict[str, Any]] = None
    ) -> _BaseAutoModelClass:
        model_class = _import_module(f"transformers.{auto_model}")
        return model_class.from_config(config, **(auto_model_kwargs or {}))

except ImportError:
    def from_transformers_config(auto_mode: str, config: Any, auto_model_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        raise ImportError(
            "The 'transformers' library is not installed. "
            "Please install it to use the 'from_transformers_config' function."
        )

def disable_grad(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False

    return module


def remove_n_layers(module: nn.Module, n: int = -1) -> nn.Module:
    selected_layers = list(module.children())

    if n < 0:
        selected_layers = selected_layers[:-n]
    else:
        selected_layers = selected_layers[n:]

    return nn.Sequential(*selected_layers)


def optim(optimizer_path: str, **kwargs: Any) -> OptimizerCallable:
    optimizer_class = _import_module(optimizer_path, default_module="torch.optim")

    return partial(optimizer_class, **kwargs)


def sched(lr_scheduler_path: str, **kwargs: Any) -> LRSchedulerCallable:
    scheduler_class = _import_module(lr_scheduler_path, default_module="torch.optim.lr_scheduler")

    return partial(scheduler_class, **kwargs)


def init_kwargs(config: Dict) -> Dict:
    """
    Recursively traverses a nested structure to replace dictionaries with a
    `class_path` key using the `instantiate_class` function.

    Args:
        config (Any): The input nested structure (dict, list, tuple, etc.).

    Returns:
        The modified structure with instantiated classes.
    """

    if isinstance(config, dict):
        if "class_path" in config:
            return instantiate_class(tuple(), config)

        return {key: init_kwargs(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [init_kwargs(item) for item in config]
    elif isinstance(config, tuple):
        return tuple(init_kwargs(item) for item in config)

    return config
