from autolightning import AutoModule, AutoCLI
from autolightning.auto_cli import LoggerSaveConfigCallback


def main():
    AutoCLI(
        AutoModule,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=LoggerSaveConfigCallback
    )

if __name__ == "__main__":
    main()
