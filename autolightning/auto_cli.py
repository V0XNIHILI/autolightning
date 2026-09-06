import sys
import os
import tempfile
import yaml
import importlib
from typing import Dict, Any, Union, List, Tuple, Optional, Literal, Callable

from jsonargparse import ActionConfigFile

import lightning as L
import pytorch_lightning as pl
from lightning.pytorch.cli import (
    LightningCLI,
    LightningArgumentParser,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import Trainer

from torch_mate.utils import disable_torch_debug_apis, configure_cuda


def cc(class_path: str, init_args: Optional[dict] = None, **kwargs: Any):
    out: Dict[str, Any] = {"class_path": class_path}

    # Use dict kwargs instead of init_args since both "autolightning.optim" and "autolightning.sched"
    # do not have named arguments but only **kwargs. The Lightning CLI does support that, but it requires
    # the use of the `dict_kwargs` key in the configuration file.
    use_dict_kwargs = class_path in ["autolightning.optim", "autolightning.sched"]

    if init_args is not None:
        if kwargs:
            raise ValueError("Cannot provide both init_args and kwargs")

        if use_dict_kwargs:
            out["dict_kwargs"] = init_args
        else:
            out["init_args"] = init_args
    elif kwargs != {}:
        if use_dict_kwargs:
            out["dict_kwargs"] = kwargs
        else:
            out["init_args"] = kwargs

    return out


def write_to_temp_file(content):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".yaml")

    try:
        # Write content to the temporary file
        temp_file.write(content)

        # Return the path of the temporary file
        return temp_file.name
    finally:
        # Close the file
        temp_file.close()


def replace_tuples_with_lists(obj: Union[Dict, List, Tuple, Any]):
    """Recursively replace all tuples in a nested dictionary or list with lists."""
    if isinstance(obj, dict):
        return {k: replace_tuples_with_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_tuples_with_lists(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return list(obj)

    return obj


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: L.LightningModule, stage: str) -> None:
        config = self.parser.dump(self.config, skip_none=True)  # Required for proper reproducibility
        config = yaml.safe_load(config)

        # Log the full config to the logger
        if isinstance(trainer.logger, Logger):
            trainer.logger.log_hyperparams(config)

        model_config = config.get("model", None)

        # Only save the model config to the pl_module as hyperparameters
        if model_config is not None:
            if "class_path" in model_config:
                model_config = model_config.get("init_args", None)

            if model_config is not None:
                pl_module.save_hyperparameters(model_config, logger=False)


class ActionConfigFilePython(ActionConfigFile):
    """ActionConfigFile with support for configurations stored in Python files with variable name `config`."""

    def __call__(self, parser, cfg, values, option_string=None):
        SEP = ".py"

        if SEP in values:
            path = values

            # Extract the directory and module name
            module_dir = os.path.dirname(path)
            module_name = os.path.splitext(os.path.basename(path))[0]

            # Add the directory to the system path
            sys.path.append(module_dir)

            # Import the module
            module = importlib.import_module(module_name)

            if hasattr(module, "CONFIG_NAME"):
                config_name = getattr(module, "CONFIG_NAME")
            else:
                config_name = "config"

            if not hasattr(module, config_name):
                raise AttributeError(f"The file '{module_name}.py' does not have a variable named '{config_name}'.")
            
            variable: Union[Callable[[], Dict], Dict] = getattr(module, config_name)

            if callable(variable):
                variable = variable()

            dict_variable = dict(variable)

            # Loop through all nested keys and replace all tuples with lists
            # as jsonargparse's implementation of YAML loading does not support tuples
            # (see line _loaders_dumpers.py#L19 in jsonargparse project where they
            # use a SafeLoader instead of a FullLoader which supports tuples)
            dict_variable = replace_tuples_with_lists(dict_variable)

            # Write the contents to a temporary file
            temp_file_path = write_to_temp_file(yaml.dump(dict_variable))

            values = temp_file_path
        elif values == "//wandb_sweep":
            import wandb

            # This run will be reused by Lightning during the run
            # when the WandbLogger is used
            wandb.init()

            wandb_config = dict(wandb.config)

            # Write the contents to a temporary file
            temp_file_path = write_to_temp_file(yaml.dump(wandb_config))

            values = temp_file_path

        super().__call__(parser, cfg, values, option_string)


class AutoCLI(LightningCLI):
    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""

        # This is a 1:1 copy of the LightningCLI.init_parser method, but now with the ActionConfigFilePython class
        # instead of the ActionConfigFile class. We add this argument here instead of in the add_arguments_to_parser
        # since otherwise we get "ValueError: A parser is only allowed to have a single ActionConfigFile argument."
        kwargs.setdefault("dump_header", [f"lightning.pytorch=={pl.__version__}"])

        parser = LightningArgumentParser(**kwargs)
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFilePython,
            help="Path to a configuration file in JSON, YAML or Python format.",
        )

        return parser

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Add arguments for disable_torch_debug_apis
        parser.add_argument(
            "--torch.autograd.set_detect_anomaly",
            type=bool,
            default=False,
            help="Whether to detect anomalies.",
        )
        parser.add_argument(
            "--torch.autograd.profiler.profile",
            type=bool,
            default=True,
            help="Whether to profile.",
        )
        parser.add_argument(
            "--torch.autograd.profiler.emit_nvtx",
            type=bool,
            default=True,
            help="Whether to emit nvtx.",
        )

        # Add arguments for configure_cuda
        parser.add_argument(
            "--torch.set_float32_matmul_precision",
            type=Literal["highest", "high", "medium"],
            default="highest",
            help="Precision of float32 matmul (highest, high, medium).",
        )
        parser.add_argument(
            "--torch.backends.cuda.matmul.allow_tf32",
            type=bool,
            default=False,
            help="Allow TF32 matmul.",
        )
        parser.add_argument(
            "--torch.backends.cudnn.allow_tf32",
            type=bool,
            default=True,
            help="Allow TF32 cuDNN operations.",
        )
        parser.add_argument(
            "--torch.backends.cudnn.benchmark",
            type=bool,
            default=False,
            help="Use cuDNN benchmark mode.",
        )

        # Add other arguments
        # torch.set_num_threads
        parser.add_argument(
            "--torch.set_num_threads",
            type=int,
            default=-1,
            help="Sets the number of threads used for intraop parallelism on CPU.",
        )

        # Add arguments for wandb.watch
        parser.add_argument(
            "--wandb.watch.enable",
            type=bool,
            default=False,
            help="Whether to enable wandb.watch.",
        )
        parser.add_argument(
            "--wandb.watch.models",
            type=Optional[Union[str, List[str]]],
            default="net",
            help="A single model or a sequence of models to be monitored.",
        )
        parser.add_argument(
            "--wandb.watch.criterion",
            type=Optional[str],
            default="criterion",
            help="The loss function being optimized (optional).",
        )
        parser.add_argument(
            "--wandb.watch.log",
            type=Optional[Literal["gradients", "parameters", "all"]],
            default="gradients",
            help="Specifies whether to log gradients, parameters, or all. Set to None to disable logging.",
        )
        parser.add_argument(
            "--wandb.watch.log_freq",
            type=int,
            default=1000,
            help="How frequently to log gradients and parameters, expressed in batches.",
        )
        parser.add_argument(
            "--wandb.watch.idx",
            type=int,
            default=None,
            help="Index used when tracking multiple models with wandb.watch.",
        )
        parser.add_argument(
            "--wandb.watch.log_graph",
            type=bool,
            default=False,
            help="Whether to log the model's computational graph.",
        )

    def _get_wandb_watch_models(self):
        wandb_watch_models = self.config["fit"]["wandb"]["watch"]["models"]

        if wandb_watch_models is not None:
            if isinstance(wandb_watch_models, str):
                models = getattr(self.model, wandb_watch_models)
            else:
                models = [getattr(self.model, model_name) for model_name in wandb_watch_models]
        else:
            models = self.model

        return models

    def before_fit(self):
        cfg = self.config["fit"]

        wandb_watch_cfg = cfg["wandb"]["watch"]

        if wandb_watch_cfg["enable"]:
            models = self._get_wandb_watch_models()

            if wandb_watch_cfg["criterion"] is None:
                criterion = None
            else:
                criterion = getattr(self.model, wandb_watch_cfg["criterion"])

            # Set up wandb logger watching
            self.trainer.logger.experiment.watch(
                models=models,
                criterion=criterion,
                log=wandb_watch_cfg["log"],
                log_freq=wandb_watch_cfg["log_freq"],
                idx=wandb_watch_cfg["idx"],
                log_graph=wandb_watch_cfg["log_graph"],
            )

    def after_fit(self):
        cfg = self.config["fit"]

        wandb_watch_cfg = cfg["wandb"]["watch"]

        if wandb_watch_cfg["enable"]:
            models = self._get_wandb_watch_models()
            self.trainer.logger.experiment.unwatch(models)

    def before_instantiate_classes(self):
        if self.subcommand is None:
            cfg = self.config
        else:
            cfg = self.config[self.subcommand]

        torch_cfg = cfg["torch"]
        torch_autograd_cfg = torch_cfg["autograd"]
        torch_backends_cfg = torch_cfg["backends"]
        torch_backends_cudnn_cfg = torch_backends_cfg["cudnn"]

        # Disable PyTorch debug APIs
        disable_torch_debug_apis(
            torch_autograd_cfg["set_detect_anomaly"],
            torch_autograd_cfg["profiler"]["profile"],
            torch_autograd_cfg["profiler"]["emit_nvtx"],
        )

        # Configure CUDA
        configure_cuda(
            torch_cfg["set_float32_matmul_precision"],
            torch_backends_cfg["cuda"]["matmul"]["allow_tf32"],
            torch_backends_cudnn_cfg["allow_tf32"],
            torch_backends_cudnn_cfg["benchmark"],
        )

        if torch_cfg["set_num_threads"] != -1:
            import torch

            torch.set_num_threads(torch_cfg["set_num_threads"])
