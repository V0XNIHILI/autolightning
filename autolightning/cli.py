import sys
import os
import tempfile
import yaml
from typing import Dict, Any, Union, List, Tuple

import inspect
from functools import wraps

from jsonargparse import ActionConfigFile

import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import lightning.pytorch as pl

from autolightning import AutoModule, AutoDataModule

from torch_mate.utils import disable_torch_debug_apis, configure_cuda


def create_factory(cls, pre_applied_first_arg, return_annotation):
    # Get the signature of the __init__ method, excluding 'self'
    init_signature = inspect.signature(cls.__init__)
    parameters = list(init_signature.parameters.values())[2:]  # Exclude 'self' and 'cfg'
    
    # Create a new signature for the factory function
    new_signature = inspect.Signature(parameters, return_annotation=return_annotation)
    
    # Create the factory function
    @wraps(cls.__init__)
    def factory(*args, **kwargs):
        # Create an instance of the class with the pre-applied first argument
        return cls(pre_applied_first_arg, *args, **kwargs)
    
    # Apply the new signature to the factory function
    factory.__signature__ = new_signature
    
    return factory


def pre_cli(module, cfg: Dict):
    # Get the method resolution order (MRO) of the class
    mro = inspect.getmro(module)

    if L.LightningDataModule in mro:
        return_annotation = L.LightningDataModule
    else:
        return_annotation = L.LightningModule

    return create_factory(module, cfg, return_annotation)


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


def replace_tuples_with_lists(obj: Union[Dict, List, Tuple]):
    """Recursively replace all tuples in a nested dictionary or list with lists."""
    if isinstance(obj, dict):
        return {k: replace_tuples_with_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_tuples_with_lists(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return obj


class ActionConfigFilePython(ActionConfigFile):
    """ActionConfigFile with support for configurations stored in Python files with variable name `self.config_var_name` (default: 'cfg')."""

    config_var_name = 'cfg'

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
            module = __import__(module_name)

            # Access the variable
            variable = getattr(module, self.config_var_name)
            dict_variable = dict(variable)

            yaml_contents = {}

            if "training" in dict_variable:
                yaml_contents["trainer"] = dict_variable["training"]

            if "learner" in dict_variable:
                yaml_contents["model"] = {
                    "class_path": dict_variable["learner"]["name"],
                    "init_args": {
                        "cfg": dict_variable
                    }
                }

            if "dataset" in dict_variable:
                yaml_contents["data"] = {
                    "class_path": dict_variable["dataset"]["name"],
                    "init_args": {
                        "cfg": dict_variable
                    }
                }

            if "seed" in dict_variable:
                yaml_contents["seed_everything"] = dict_variable["seed"]

            # Loop through all nested keys and replace all tuples with lists
            # as jsonargparse's implementation of YAML loading does not support tuples
            # (see line _loaders_dumpers.py#L19 in jsonargparse project where they
            # use a SafeLoader instead of a FullLoader which supports tuples)
            yaml_contents = replace_tuples_with_lists(yaml_contents)

            # Write the contents to a temporary file
            temp_file_path = write_to_temp_file(yaml.dump(yaml_contents))

            values = temp_file_path

        super().__call__(parser, cfg, values, option_string)


class AutoCLI(LightningCLI):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        kwargs.setdefault("dump_header", [f"lightning.pytorch=={pl.__version__}"])
        parser = LightningArgumentParser(**kwargs)
        parser.add_argument(
            "-c", "--config", action=ActionConfigFilePython, help="Path to a configuration file in json or yaml format."
        )

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


def main():
    AutoCLI(
        AutoModule,
        AutoDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"}
    )

if __name__ == "__main__":
    main()
