from typing import Optional, Union, List

from autolightning import AutoModule, AutoCLI
from autolightning.auto_cli import LoggerSaveConfigCallback
from autolightning.utils import merge_dicts


def auto_main(
    config: Optional[Union[List[dict], dict]] = None,
    subcommand: Optional[str] = None,
    run: bool = True,
):
    final_config = None
    main_config = {}

    if config is not None:
        if run:
            assert subcommand is not None, "subcommand must be provided if run is True and config is not None."
        else:
            assert subcommand is None, "subcommand must be None if run is False and config is not None."

        if isinstance(config, list):
            for subconfig in config:
                main_config = merge_dicts(main_config, subconfig)
        else:
            main_config = config

        final_config = {}

        if subcommand is not None:
            final_config["subcommand"] = subcommand
            final_config[subcommand] = main_config
        else:
            final_config = main_config
    else:
        assert subcommand is None, "subcommand must be None if config is None."
        assert run, "run must be True if config is None."

    cli = AutoCLI(
        AutoModule,
        args=final_config,
        run=run,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=LoggerSaveConfigCallback,
    )

    return cli.trainer, cli.model, cli.datamodule


def auto_data(config: Union[List[dict], dict], config_includes_data_key: bool = True):
    final_config = {}

    if isinstance(config, list):
        for subconfig in config:
            final_config = merge_dicts(final_config, subconfig)
    else:
        final_config = config

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
