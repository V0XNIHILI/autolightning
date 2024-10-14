from autolightning import AutoModule, AutoDataModule, AutoCLI
from autolightning.AutoCLI import LoggerSaveConfigCallback


def main():
    AutoCLI(
        AutoModule,
        AutoDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=LoggerSaveConfigCallback
    )

if __name__ == "__main__":
    main()
