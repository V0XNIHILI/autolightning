from typing import Optional, Union, List
from pathlib import Path
import yaml

from autolightning import AutoModule, AutoCLI
from autolightning.auto_cli import LoggerSaveConfigCallback
from autolightning.utils import merge_dicts


def get_auto_cli(args, run: bool):
    return AutoCLI(
        AutoModule,
        args=args,
        run=run,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=LoggerSaveConfigCallback,
    )


def build_config(config: Union[List[dict], dict, List[str], List[Path], str, Path]):
    if isinstance(config, (str, Path)):
        config = [config]

    main_config = {}

    if isinstance(config, list):
        for subconfig in config:
            # TODOOO make function out of Python file loading as well
            if isinstance(subconfig, (str, Path)):
                with open(subconfig, "r") as f:
                    subconfig = yaml.safe_load(f)
            main_config = merge_dicts(main_config, subconfig)
    else:
        main_config = config

    return main_config


def auto_main(
    config: Optional[Union[List[dict], dict, List[str], List[Path], str, Path]] = None,
    subcommand: Optional[str] = None,
    run: bool = True,
):
    final_config = None

    if config is not None:
        if run:
            assert subcommand is not None, "subcommand must be provided if run is True and config is not None."
        else:
            assert subcommand is None, "subcommand must be None if run is False and config is not None."

        main_config = build_config(config)
        final_config = {}

        if subcommand is not None:
            final_config["subcommand"] = subcommand
            final_config[subcommand] = main_config
        else:
            final_config = main_config
    else:
        assert subcommand is None, "subcommand must be None if config is None."
        assert run, "run must be True if config is None."

    cli = get_auto_cli(final_config, run)

    return cli.trainer, cli.model, cli.datamodule


def auto_data(config: Union[List[dict], dict, List[str], List[Path], str, Path], config_includes_data_key: bool = True):
    final_config = build_config(config)

    if not config_includes_data_key:
        final_config = {"data": final_config}
    else:
        final_config["trainer"] = {}
        final_config["optimizer"] = None
        final_config["lr_scheduler"] = None

    final_config["model"] = {"class_path": "autolightning.AutoModule"}

    _, _, ds = auto_main(final_config, run=False)

    return ds


def cli_main():
    # Created a separate function for the CLI that does not
    # return anything to avoid a non-zero exit code.
    auto_main()
