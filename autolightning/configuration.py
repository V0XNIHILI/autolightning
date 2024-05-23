from typing import Dict, Optional
from copy import deepcopy

from lightning import Trainer
import pytorch_lightning as pl

import torch_mate.lightning.datasets
import torch_mate.lightning.lm
from torch_mate.utils import get_class_and_init
from torch_mate.lightning.utils import build_trainer_kwargs


def set_seed(cfg: Dict, set_seeds: bool):
    if "seed" in cfg and set_seeds:
        pl.seed_everything(cfg["seed"])


def configure_trainer(cfg: Dict, **kwargs):
    trainer_cfg = build_trainer_kwargs(cfg)

    if kwargs != {}:
        trainer_cfg.update(kwargs)

    return Trainer(**trainer_cfg)


def configure_data(cfg: Dict, del_dataset_module_kwargs: bool = True, **kwargs):
    name_and_config = {"name": cfg["dataset"]["name"]}

    all_kwargs = {}

    if "kwargs" in cfg["dataset"]:
        all_kwargs.update(cfg["dataset"]["kwargs"])

        if del_dataset_module_kwargs:
            cfg["dataset"].pop("kwargs", None)

    all_kwargs.update(kwargs)

    if all_kwargs != {}:
        name_and_config["cfg"] = all_kwargs
    
    return get_class_and_init(torch_mate.lightning.datasets, name_and_config, cfg)


def configure_model(cfg: Dict, **kwargs):
    name_and_config = {"name": cfg["learner"]["name"]}

    all_kwargs = {}

    if "kwargs" in cfg["learner"]:
        all_kwargs.update(cfg["learner"]["kwargs"])

    all_kwargs.update(kwargs)

    if all_kwargs != {}:
        name_and_config["cfg"] = all_kwargs
    
    return get_class_and_init(torch_mate.lightning.lm, name_and_config, cfg)


def configure_model_data(cfg: Dict, model_kwargs: Optional[Dict] = None, data_kwargs: Optional[Dict] = None, del_dataset_module_kwargs: bool = True):
    cfg = deepcopy(cfg)

    set_seed(cfg, True)

    # Instantiate data module before model module as the the dataset
    # module kwargs might be deleted from the config dictionary
    data = configure_data(cfg, del_dataset_module_kwargs, **(data_kwargs if data_kwargs else {}))
    model = configure_model(cfg, **(model_kwargs if model_kwargs else {}))

    return model, data


def configure_all(cfg: Dict, trainer_kwargs: Optional[Dict] = None, model_kwargs: Optional[Dict] = None, data_kwargs: Optional[Dict] = None, del_dataset_module_kwargs: bool = True):
    model, data = configure_model_data(cfg, model_kwargs, data_kwargs, del_dataset_module_kwargs)
    trainer = configure_trainer(cfg, **(trainer_kwargs if trainer_kwargs else {}))

    return trainer, model, data
