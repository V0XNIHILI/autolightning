import inspect
from typing import Dict

import lightning as L

from autolightning import AutoModule, AutoDataModule, AutoCLI
from autolightning.AutoCLI import create_factory


def pre_cli(module, cfg: Dict):
    # Get the method resolution order (MRO) of the class
    mro = inspect.getmro(module)

    if L.LightningDataModule in mro:
        return_annotation = L.LightningDataModule
    else:
        return_annotation = L.LightningModule

    return create_factory(module, cfg, return_annotation)


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
