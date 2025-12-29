from __future__ import annotations

import contextlib
import io
import argparse
import copy
from pathlib import Path
from typing import Any, Dict
import yaml

import wandb

from autolightning.main import get_auto_cli


@contextlib.contextmanager
def suppress_stdout() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _add_parameters_key(dictionary):
    # Add a key parameters to every nested dictionary in a dictionary
    for key, value in dictionary.items():
        if isinstance(value, dict):
            if "values" not in value.keys() and "min" not in value.keys() and "max" not in value.keys():
                dictionary[key] = {"parameters": value}
                _add_parameters_key(value)


def _add_value_key(dictionary):
    # Add a key value to every nested value in a dictionary
    for key, value in dictionary.items():
        if isinstance(value, dict):
            _add_value_key(value)
        else:
            dictionary[key] = {"value": value}


def _remove_value_key_if_more_keys_exist(dictionary):
    # Remove a key value to every nested value in a dictionary if there are more keys
    for _, value in dictionary.items():
        if isinstance(value, dict):
            if "value" in value.keys() and len(value.keys()) > 1:
                del value["value"]
            _remove_value_key_if_more_keys_exist(value)


def _merge_dicts(dict1: dict, dict2: dict):
    """Code taken from: https://stackoverflow.com/a/58742155/11251769."""

    for key, val in dict1.items():
        if type(val) == dict:
            if key in dict2 and type(dict2[key] == dict):
                _merge_dicts(dict1[key], dict2[key])
        else:
            if key in dict2:
                dict1[key] = dict2[key]

    for key, val in dict2.items():
        if key not in dict1:
            dict1[key] = val

    return dict1


def transform_config(config: Dict[str, Any]) -> Dict[str, Any]:
    autolightning_args = " ".join(config["command"]).split("autolightning ")[1]
    args_without_subcommand = autolightning_args.split(" ")[1:]

    # Do not print (for example): 
    # GPU available: True (cuda), used: True
    # TPU available: False, using: 0 TPU cores
    # HPU available: False, using: 0 HPUs
    with suppress_stdout():
        cli = get_auto_cli(args_without_subcommand, run=False)

    sweep_param_config = cli.parser.dump(cli.config, skip_none=True)  # Required for proper reproducibility
    sweep_param_config = yaml.safe_load(sweep_param_config)

    # Update the sweep config with the variable parameters
    sweep_param_config = copy.deepcopy(sweep_param_config)

    _add_parameters_key(sweep_param_config)
    _add_value_key(sweep_param_config)

    sweep_variable_param_config = copy.deepcopy(config["parameters"])

    _add_parameters_key(sweep_variable_param_config)

    full_sweep_param_config = _merge_dicts(sweep_param_config, sweep_variable_param_config)
    _remove_value_key_if_more_keys_exist(full_sweep_param_config)

    config["parameters"] = full_sweep_param_config

    config["command"] = config["command"] + ["-c", "//wandb_sweep"]

    return config


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Sweep config must be a YAML mapping")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a W&B sweep from a YAML configuration"
    )

    parser.add_argument(
        "config_yaml",
        type=Path,
        help="YAML file defining the hyperparameter sweep",
    )

    parser.add_argument(
        "--project",
        "-p",
        type=str,
        default=None,
        help="W&B project name (default: Uncategorized)",
    )

    parser.add_argument(
        "--entity",
        "-e",
        type=str,
        default=None,
        help="W&B entity (user or team)",
    )

    parser.add_argument(
        "--prior_run",
        "-R",
        type=str,
        default=None,
        help="Existing run ID to add to this sweep",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config_yaml)
    config = transform_config(config)

    wandb.sweep(
        sweep=config,
        project=args.project,
        entity=args.entity,
        prior_runs=args.prior_run,
    )
