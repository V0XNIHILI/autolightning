import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from autolightning.callbacks import SaveInitialCheckpoint

warnings.filterwarnings("ignore")
torch.manual_seed(0)


def dl(n=8, batch=4):
    x = torch.randn(n, 4)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch)


class TinyModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 1)

    def _step(self, batch, key):
        x, y = batch
        loss = nn.functional.mse_loss(self.net(x), y)
        self.log(key, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train/loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val/loss")

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.1)


def trainer(**kw):
    kw.setdefault("logger", False)
    kw.setdefault("enable_checkpointing", False)

    return L.Trainer(
        max_epochs=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        **kw,
    )


def fit(t):
    t.fit(TinyModule(), dl(), dl())
    return t


def test_follows_model_checkpoint_dir(tmp_path):
    """The regression: with a logger that sets `save_dir`, the initial checkpoint must
    land next to the `ModelCheckpoint` files, not in the current working directory."""

    t = fit(
        trainer(
            logger=CSVLogger(save_dir=str(tmp_path)),
            enable_checkpointing=True,
            callbacks=[SaveInitialCheckpoint(), ModelCheckpoint(monitor="val/loss")],
        )
    )

    dirpath = Path(t.checkpoint_callback.dirpath)

    assert (dirpath / "initial.ckpt").is_file()
    # The old behaviour resolved to `<cwd>/checkpoints`, which `default_root_dir` still is.
    assert dirpath != Path(t.default_root_dir) / "checkpoints"
    assert not (Path(t.default_root_dir) / "checkpoints" / "initial.ckpt").exists()


def test_explicit_dirpath_wins(tmp_path):
    explicit = tmp_path / "explicit"

    t = fit(
        trainer(
            logger=CSVLogger(save_dir=str(tmp_path)),
            enable_checkpointing=True,
            callbacks=[SaveInitialCheckpoint(dirpath=str(explicit)), ModelCheckpoint(monitor="val/loss")],
        )
    )

    assert (explicit / "initial.ckpt").is_file()
    assert not (Path(t.checkpoint_callback.dirpath) / "initial.ckpt").exists()


def test_custom_filename(tmp_path):
    fit(
        trainer(
            default_root_dir=str(tmp_path),
            callbacks=[SaveInitialCheckpoint(filename="untrained.ckpt")],
        )
    )

    assert (tmp_path / "checkpoints" / "untrained.ckpt").is_file()


def test_falls_back_to_logger_dir_without_model_checkpoint(tmp_path):
    """Without a `ModelCheckpoint` the callback mirrors Lightning's own resolution."""

    logger = CSVLogger(save_dir=str(tmp_path))

    fit(trainer(logger=logger, callbacks=[SaveInitialCheckpoint()]))

    expected = Path(tmp_path) / str(logger.name) / f"version_{logger.version}" / "checkpoints"

    assert (expected / "initial.ckpt").is_file()


def test_falls_back_to_default_root_dir_without_logger(tmp_path):
    fit(trainer(default_root_dir=str(tmp_path), callbacks=[SaveInitialCheckpoint()]))

    assert (tmp_path / "checkpoints" / "initial.ckpt").is_file()


def test_does_not_overwrite_existing_checkpoint(tmp_path):
    """On a resumed run the model is no longer at its initialization, so an existing
    initial checkpoint must be left alone."""

    path = tmp_path / "checkpoints" / "initial.ckpt"
    os.makedirs(path.parent, exist_ok=True)
    path.write_bytes(b"sentinel")

    fit(trainer(default_root_dir=str(tmp_path), callbacks=[SaveInitialCheckpoint()]))

    assert path.read_bytes() == b"sentinel"


def test_overwrite_replaces_existing_checkpoint(tmp_path):
    path = tmp_path / "checkpoints" / "initial.ckpt"
    os.makedirs(path.parent, exist_ok=True)
    path.write_bytes(b"sentinel")

    fit(trainer(default_root_dir=str(tmp_path), callbacks=[SaveInitialCheckpoint(overwrite=True)]))

    assert path.read_bytes() != b"sentinel"
