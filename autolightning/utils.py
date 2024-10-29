from typing import Any, Optional, Dict
from collections import OrderedDict
import importlib

import torch
import torch.nn as nn


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


def _import_module(module_path: str) -> Any:
    module_name, function_name = module_path.rsplit(".")

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


def disable_grad(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False

    return module
