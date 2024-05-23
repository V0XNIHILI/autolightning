from typing import Type, Optional, Dict, Union, Callable, Any

from autolightning import AutoModule, AutoDataModule

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback, ArgsType
from pytorch_lightning import Trainer


class AutoCLI(LightningCLI):
    def __init__(
        self,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True
    ):
        """Small wrapper class around LightningCLI to make it easier to use with AutoModule and AutoDataModule.
        
        See the original LightningCLI [docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) for more information on the possible arguments.
        """

        full_parser_kwargs = {"parser_mode": "omegaconf"}

        if parser_kwargs is not None:
            full_parser_kwargs.update(parser_kwargs)

        super().__init__(
            AutoModule,
            AutoDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            parser_kwargs=full_parser_kwargs,
            save_config_callback=save_config_callback,
            save_config_kwargs=save_config_kwargs,
            trainer_class=trainer_class,
            trainer_defaults=trainer_defaults,
            seed_everything_default=seed_everything_default,
            args=args,
            run=run,
            auto_configure_optimizers=auto_configure_optimizers
        )
