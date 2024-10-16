import sys
import os
import tempfile
import yaml
import importlib
from typing import Dict, Any, Union, List, Tuple

from jsonargparse import ActionConfigFile

import lightning as L
import pytorch_lightning as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import Trainer

from torch_mate.utils import disable_torch_debug_apis, configure_cuda


def write_to_temp_file(content):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.yaml')
    
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
        if model_config != None:
            if "class_path" in model_config:
                model_config = model_config.get("init_args", None)

            if model_config != None:
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

            variable = getattr(module, config_name)
            dict_variable = dict(variable)

            # Loop through all nested keys and replace all tuples with lists
            # as jsonargparse's implementation of YAML loading does not support tuples
            # (see line _loaders_dumpers.py#L19 in jsonargparse project where they
            # use a SafeLoader instead of a FullLoader which supports tuples)
            dict_variable = replace_tuples_with_lists(dict_variable)

            # Write the contents to a temporary file
            temp_file_path = write_to_temp_file(yaml.dump(dict_variable))

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
        parser.add_argument("-c", "--config", action=ActionConfigFilePython, help="Path to a configuration file in JSON, YAML or Python format.")

        return parser
    
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Add arguments for disable_torch_debug_apis
        parser.add_argument("--torch.autograd.set_detect_anomaly", type=bool, default=False, help="Whether to detect anomalies.")
        parser.add_argument("--torch.autograd.profiler.profile", type=bool, default=True, help="Whether to profile.")
        parser.add_argument("--torch.autograd.profiler.emit_nvtx", type=bool, default=True, help="Whether to emit nvtx.")

        # Add arguments for configure_cuda
        parser.add_argument("--torch.set_float32_matmul_precision", type=str, default='highest', help="Precision of float32 matmul (highest, high, medium).")
        parser.add_argument("--torch.backends.cuda.matmul.allow_tf32", type=bool, default=False, help="Allow TF32 matmul.")
        parser.add_argument("--torch.backends.cudnn.allow_tf32", type=bool, default=True, help="Allow TF32 cuDNN operations.")
        parser.add_argument("--torch.backends.cudnn.benchmark", type=bool, default=False, help="Use cuDNN benchmark mode.")
    
    def before_instantiate_classes(self):
        # Get the configuration
        cfg = self.config[self.config.subcommand]

        torch_cfg = cfg["torch"]
        torch_autograd_cfg = torch_cfg["autograd"]
        torch_backends_cfg = torch_cfg["backends"]
        torch_backends_cudnn_cfg = torch_backends_cfg["cudnn"]

        # Disable PyTorch debug APIs
        disable_torch_debug_apis(
            torch_autograd_cfg["set_detect_anomaly"],
            torch_autograd_cfg["profiler"]["profile"],
            torch_autograd_cfg["profiler"]["emit_nvtx"]
        )

        # Configure CUDA
        configure_cuda(
            torch_cfg["set_float32_matmul_precision"],
            torch_backends_cfg["cuda"]["matmul"]["allow_tf32"],
            torch_backends_cudnn_cfg["allow_tf32"],
            torch_backends_cudnn_cfg["benchmark"]
        )
