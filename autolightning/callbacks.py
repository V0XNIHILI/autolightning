import os
from typing import Optional

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.callbacks import Callback


class SaveInitialCheckpoint(Callback):
    """Saves a checkpoint of the untrained model at the start of `fit`.

    By default, the checkpoint is written next to the checkpoints of the trainer's
    `ModelCheckpoint`, so it follows whatever directory the logger dictates.

    Args:
        dirpath (Optional[str]):
            Directory to save the checkpoint in. If not specified, the directory of the
            trainer's `ModelCheckpoint` is used, falling back to the same directory that
            `ModelCheckpoint` would have resolved to.
        filename (str):
            Name of the checkpoint file.
        overwrite (bool):
            Whether to overwrite an already existing checkpoint. Disabled by default,
            since on a resumed run the model is no longer at its initialization.
    """

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: str = "initial.ckpt",
        overwrite: bool = False,
    ):
        self.dirpath = dirpath
        self.filename = filename
        self.overwrite = overwrite

    def resolve_dirpath(self, trainer) -> str:
        if self.dirpath is not None:
            return self.dirpath

        # `ModelCheckpoint.setup` runs before `on_fit_start`, so its `dirpath` has
        # already been resolved and broadcast. Reusing it keeps the initial checkpoint
        # next to every other checkpoint, whatever the logger decided.
        for callback in trainer.checkpoint_callbacks:
            dirpath = getattr(callback, "dirpath", None)

            if dirpath:
                return dirpath

        # No `ModelCheckpoint`: mirror `ModelCheckpoint.__resolve_ckpt_dir`.
        if trainer.loggers:
            logger = trainer.loggers[0]
            save_dir = logger.save_dir if logger.save_dir is not None else trainer.default_root_dir
            version = logger.version
            version = version if isinstance(version, str) else f"version_{version}"

            return os.path.join(save_dir, str(logger.name), version, "checkpoints")

        return os.path.join(trainer.default_root_dir, "checkpoints")

    def on_fit_start(self, trainer, pl_module):
        dirpath = self.resolve_dirpath(trainer)
        path = os.path.join(dirpath, self.filename)

        fs = get_filesystem(dirpath)

        # Lightning restores weights before `on_fit_start`, so on a resumed run the model
        # is no longer at its initialization; do not clobber the real initial checkpoint.
        # Broadcast the decision so that every rank agrees before the collective save.
        should_save = self.overwrite or not fs.exists(path)
        should_save = trainer.strategy.broadcast(should_save)

        if not should_save:
            return

        fs.makedirs(dirpath, exist_ok=True)
        trainer.save_checkpoint(path)
