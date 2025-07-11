import pytest

from torchvision.datasets import CIFAR10

from autolightning import AutoDataModule
from autolightning.auto_data_module import PHASES


TRAIN_KWARGS = {"batch_size": 32, "num_workers": 4}
VAL_KWARGS = {"batch_size": 64, "num_workers": 8}


def test_dataloader_kwargs_basic():
    kwargs = {"batch_size": 32, "num_workers": 4}

    data = AutoDataModule(dataloaders=kwargs)

    for phase in PHASES:
        assert data.get_dataloader_kwargs(phase) == kwargs


def test_per_phase_dataloader_kwargs():
    kwargs = {"train": TRAIN_KWARGS, "val": VAL_KWARGS}

    data = AutoDataModule(dataloaders=kwargs)

    assert data.get_dataloader_kwargs("train") == TRAIN_KWARGS
    assert data.get_dataloader_kwargs("val") == VAL_KWARGS
    assert data.get_dataloader_kwargs("test") == data.get_dataloader_kwargs("pred") == {}


def test_per_phase_dataloader_kwargs_including_defaults():
    default_kwargs = {"batch_size": 16, "num_workers": 2}

    kwargs = {"train": TRAIN_KWARGS, "val": VAL_KWARGS, "defaults": default_kwargs}

    data = AutoDataModule(dataloaders=kwargs)

    assert data.get_dataloader_kwargs("train") == TRAIN_KWARGS
    assert data.get_dataloader_kwargs("val") == VAL_KWARGS
    assert data.get_dataloader_kwargs("test") == data.get_dataloader_kwargs("pred") == default_kwargs


def test_mixed_dataloader_kwargs():
    kwargs = {**TRAIN_KWARGS, "pred": {}}

    data = AutoDataModule(dataloaders=kwargs)

    for stage in PHASES:
        with pytest.raises(Exception):
            data.get_dataloader_kwargs(stage)


def test_post_init_dataset():
    data = AutoDataModule(
        dataset=dict(
            class_name="torchvision.datasets.CIFAR10",
            args=dict(
                defaults=dict(root="data", download=True),
                train=dict(train=True),
                val=dict(train=False),
            ),
        )
    )

    data.prepare_data()

    data.setup("fit")

    train_ds = data.get_dataset("train")
    val_ds = data.get_dataset("val")

    assert len(train_ds) == 50000
    assert len(val_ds) == 10000

    assert train_ds.train
    assert not val_ds.train

    train_ds_manually = CIFAR10("data", train=True, download=True)
    val_ds_manually = CIFAR10("data", train=False, download=True)

    for i in range(len(train_ds)):
        assert train_ds[i] == train_ds_manually[i]

    for i in range(len(val_ds)):
        assert val_ds[i] == val_ds_manually[i]


def test_regular_dataset():
    ds = CIFAR10("data", train=True, download=True)

    data = AutoDataModule(dataset=ds)

    data.prepare_data()

    data.setup("fit")

    ds_from_data = data.get_dataset("train")

    for i in range(len(ds)):
        assert ds[i] == ds_from_data[i]

    ds_from_data = data.get_dataset("val")

    for i in range(len(ds)):
        assert ds[i] == ds_from_data[i]


def test_datasets_per_phase():
    ds_train = CIFAR10("data", train=True, download=True)
    ds_val = CIFAR10("data", train=False, download=True)

    data = AutoDataModule(dataset=dict(train=ds_train, val=ds_val))

    data.prepare_data()

    data.setup("fit")

    ds_from_data_train = data.get_dataset("train")
    ds_from_data_val = data.get_dataset("val")

    for i in range(len(ds_train)):
        assert ds_train[i] == ds_from_data_train[i]

    for i in range(len(ds_val)):
        assert ds_val[i] == ds_from_data_val[i]


def test_dataloaders_per_phase():
    ds_train = CIFAR10("data", train=True, download=True)
    ds_val = CIFAR10("data", train=False, download=True)

    data = AutoDataModule(
        dataset=dict(
            train=ds_train,
            val=ds_val,
            data_loader=dict(train=dict(batch_size=32), val=dict(batch_size=64)),
        )
    )

    data.prepare_data()

    data.setup("fit")

    train_dl = data.train_dataloader()
    val_dl = data.val_dataloader()

    for i in range(len(ds_train)):
        assert ds_train[i] == train_dl.dataset[i]

    for i in range(len(ds_val)):
        assert ds_val[i] == val_dl.dataset[i]

    assert train_dl.batch_size == 32
    assert val_dl.batch_size == 64


def test_post_init_dataset_per_phase():
    data = AutoDataModule(
        dataset=dict(
            train=dict(
                class_name="torchvision.datasets.CIFAR10",
                args=dict(root="data", download=True, train=True),
            ),
            val=dict(
                class_name="torchvision.datasets.CIFAR10",
                args=dict(root="data", download=True, train=False),
            ),
            test=CIFAR10("data", train=False, download=True),
        )
    )

    data.prepare_data()

    data.setup("fit")

    train_ds = data.get_dataset("train")
    val_ds = data.get_dataset("val")

    assert len(train_ds) == 50000
    assert len(val_ds) == 10000

    assert train_ds.train
    assert not val_ds.train

    train_ds_manually = CIFAR10("data", train=True, download=True)
    val_ds_manually = CIFAR10("data", train=False, download=True)

    for i in range(len(train_ds)):
        assert train_ds[i] == train_ds_manually[i]

    for i in range(len(val_ds)):
        assert val_ds[i] == val_ds_manually[i]

    data.setup("test")

    test_ds = data.get_dataset("test")

    assert len(test_ds) == 10000
    assert not test_ds.train

    for i in range(len(test_ds)):
        assert test_ds[i] == val_ds_manually[i]
