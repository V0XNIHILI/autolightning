import sys
import os
import tempfile
import yaml
from typing import Any

from jsonargparse import ActionConfigFile

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import lightning.pytorch as pl

from autolightning import AutoModule, AutoDataModule


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


def main():
    AutoCLI(
        AutoModule,
        AutoDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == "__main__":
    main()
