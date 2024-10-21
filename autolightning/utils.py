from typing import Any, Optional
from collections import OrderedDict

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
