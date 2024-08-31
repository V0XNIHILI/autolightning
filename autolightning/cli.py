from autolightning import AutoModule, AutoDataModule, AutoCLI


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
