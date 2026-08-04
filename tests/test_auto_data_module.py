import pytest

from torch.utils.data import Dataset, IterableDataset
from torchvision.datasets import CIFAR10

from torch_mate.data.utils import Transformed, TransformedIterable, PreLoaded

from autolightning import AutoDataModule
from autolightning.auto_data_module import PHASES


TRAIN_KWARGS = {"batch_size": 32, "num_workers": 4}
VAL_KWARGS = {"batch_size": 64, "num_workers": 8}

CIFAR_TRAIN_LEN = 50000
CIFAR_VAL_LEN = 10000

# Comparing every CIFAR10 sample means 50k PIL image comparisons per assertion,
# which dominates the runtime of the whole suite. Sampling a spread of indices
# catches the same mistakes (wrong split, wrong ordering, wrong dataset object).
# Set to None to compare every sample instead.
SAMPLE_COUNT = 25


def sample_indices(length, count=SAMPLE_COUNT):
    if count is None or count >= length:
        return range(length)

    step = length // count

    return [i * step for i in range(count)]


def assert_same_samples(actual, expected):
    """Both datasets have the same length and agree on a spread of indices."""

    assert len(actual) == len(expected)

    for i in sample_indices(len(expected)):
        assert actual[i] == expected[i]


@pytest.fixture(scope="session")
def cifar_train():
    return CIFAR10("data", train=True, download=True)


@pytest.fixture(scope="session")
def cifar_val():
    return CIFAR10("data", train=False, download=True)


def cifar_spec(train: bool):
    return dict(class_name="torchvision.datasets.CIFAR10", args=dict(root="data", download=True, train=train))


class PairDataset(Dataset):
    """Yields `(x, y)` pairs so transform behaviour can be checked arithmetically."""

    def __init__(self, n=10, offset=0):
        self.n = n
        self.offset = offset

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.offset + i, (self.offset + i) * 10


class StreamDataset(IterableDataset):
    """An iterable dataset, i.e. one without `__len__`."""

    def __iter__(self):
        return iter([(i, i * 10) for i in range(10)])


class CountingDataset(Dataset):
    """Counts how many times it is constructed, to check nothing is built twice."""

    count = 0

    def __init__(self, n=10, tag="default"):
        CountingDataset.count += 1
        self.n = n
        self.tag = tag

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i, i * 10


@pytest.fixture(autouse=True)
def _reset_count():
    CountingDataset.count = 0
    yield
    CountingDataset.count = 0


def counting_spec(**args):
    return {"class_name": CountingDataset, "args": args}


# ---------------------------------------------------------------------------
# Dataloader keyword arguments
# ---------------------------------------------------------------------------


def test_dataloader_kwargs_basic():
    kwargs = {"batch_size": 32, "num_workers": 4}

    data = AutoDataModule(dataloaders=kwargs)

    for phase in PHASES:
        assert data.get_dataloader_kwargs(phase, ...) == kwargs


def test_dataloader_kwargs_default_to_empty():
    data = AutoDataModule()

    for phase in PHASES:
        assert data.get_dataloader_kwargs(phase, ...) == {}


def test_per_phase_dataloader_kwargs():
    kwargs = {"train": TRAIN_KWARGS, "val": VAL_KWARGS}

    data = AutoDataModule(dataloaders=kwargs)

    assert data.get_dataloader_kwargs("train", ...) == TRAIN_KWARGS
    assert data.get_dataloader_kwargs("val", ...) == VAL_KWARGS
    assert data.get_dataloader_kwargs("test", ...) == data.get_dataloader_kwargs("pred", ...) == {}


def test_per_phase_dataloader_kwargs_including_defaults():
    default_kwargs = {"batch_size": 16, "num_workers": 2}

    kwargs = {"train": TRAIN_KWARGS, "val": VAL_KWARGS, "defaults": default_kwargs}

    data = AutoDataModule(dataloaders=kwargs)

    assert data.get_dataloader_kwargs("train", ...) == TRAIN_KWARGS
    assert data.get_dataloader_kwargs("val", ...) == VAL_KWARGS
    assert data.get_dataloader_kwargs("test", ...) == data.get_dataloader_kwargs("pred", ...) == default_kwargs


def test_only_defaults_dataloader_kwargs():
    default_kwargs = {"batch_size": 16}

    data = AutoDataModule(dataloaders={"defaults": default_kwargs})

    for phase in PHASES:
        assert data.get_dataloader_kwargs(phase, ...) == default_kwargs


def test_per_phase_kwargs_override_individual_defaults():
    data = AutoDataModule(
        dataloaders={"defaults": {"batch_size": 16, "num_workers": 2}, "train": {"batch_size": 32}}
    )

    assert data.get_dataloader_kwargs("train", ...) == {"batch_size": 32, "num_workers": 2}


def test_mixed_dataloader_kwargs():
    # Mixing flat kwargs with per-phase kwargs is ambiguous and must be rejected.
    kwargs = {**TRAIN_KWARGS, "pred": {}}

    data = AutoDataModule(dataloaders=kwargs)

    for stage in PHASES:
        with pytest.raises(AssertionError, match="Unsupported keys in dataloader configuration"):
            data.get_dataloader_kwargs(stage, ...)


# ---------------------------------------------------------------------------
# Datasets: configuration shapes
# ---------------------------------------------------------------------------


def test_post_init_dataset(cifar_train, cifar_val):
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

    assert len(train_ds) == CIFAR_TRAIN_LEN
    assert len(val_ds) == CIFAR_VAL_LEN

    assert train_ds.train
    assert not val_ds.train

    assert_same_samples(train_ds, cifar_train)
    assert_same_samples(val_ds, cifar_val)


def test_regular_dataset(cifar_train):
    data = AutoDataModule(dataset=cifar_train)

    data.prepare_data()
    data.setup("fit")

    assert data.get_dataset("train") is cifar_train


def test_regular_dataset_has_no_validation_split(cifar_train):
    # A dataset for the train phase only is a valid configuration; it is asking
    # for the missing phase that fails, not setting up.
    data = AutoDataModule(dataset=cifar_train)
    data.setup("fit")

    assert not data.has_dataset_in_plan("val")

    with pytest.raises(KeyError, match="No dataset is configured for phase 'val'"):
        data.get_dataset("val")


def test_datasets_per_phase(cifar_train, cifar_val):
    data = AutoDataModule(dataset=dict(train=cifar_train, val=cifar_val))

    data.prepare_data()
    data.setup("fit")

    assert data.get_dataset("train") is cifar_train
    assert data.get_dataset("val") is cifar_val


def test_post_init_dataset_per_phase(cifar_train, cifar_val):
    data = AutoDataModule(
        dataset=dict(
            train=cifar_spec(train=True),
            val=cifar_spec(train=False),
            test=cifar_val,
        )
    )

    data.prepare_data()
    data.setup("fit")

    train_ds = data.get_dataset("train")
    val_ds = data.get_dataset("val")

    assert len(train_ds) == CIFAR_TRAIN_LEN
    assert len(val_ds) == CIFAR_VAL_LEN

    assert train_ds.train
    assert not val_ds.train

    assert_same_samples(train_ds, cifar_train)
    assert_same_samples(val_ds, cifar_val)

    data.setup("test")

    test_ds = data.get_dataset("test")

    assert len(test_ds) == CIFAR_VAL_LEN
    assert not test_ds.train
    assert_same_samples(test_ds, cifar_val)


def test_setup_only_builds_the_phases_of_the_stage():
    data = AutoDataModule(dataset={"train": counting_spec(), "test": counting_spec()})
    data.setup("test")

    assert set(data.instantiated_dataset) == {"test"}
    assert CountingDataset.count == 1


def test_get_dataset_before_setup():
    data = AutoDataModule(dataset={"train": counting_spec()})

    with pytest.raises(KeyError, match="has not been built"):
        data.get_dataset("train")


@pytest.mark.parametrize(
    "kwargs, phase, expected",
    [
        ({}, "train", True),
        ({}, "val", False),
        ({"random_split": {"train": 0.8, "val": 0.2}}, "val", True),
        ({"random_split": {"train": 0.8, "val": 0.2}}, "test", False),
        ({"cross_val": {"n_folds": 5, "fold_idx": 0}}, "val", True),
        ({"cross_val": {"n_folds": 5, "fold_idx": 0}}, "pred", False),
    ],
)
def test_has_dataset_in_plan(kwargs, phase, expected):
    assert AutoDataModule(dataset=counting_spec(), **kwargs).has_dataset_in_plan(phase) is expected


# ---------------------------------------------------------------------------
# Datasets: how often they get built
# ---------------------------------------------------------------------------


def test_prepare_data_builds_every_declared_phase_once():
    data = AutoDataModule(dataset={"train": counting_spec(), "val": counting_spec(), "test": counting_spec()})
    data.prepare_data()

    assert CountingDataset.count == 3


def test_prepare_data_can_be_disabled():
    data = AutoDataModule(dataset={"train": counting_spec()}, requires_prepare=False)
    data.prepare_data()

    assert CountingDataset.count == 0


def test_prepare_data_and_setup_build_separately():
    # `prepare_data` runs once per node (to download), `setup` runs in every
    # process, so the two deliberately do not share instances.
    data = AutoDataModule(dataset={"train": counting_spec(), "val": counting_spec()})
    data.prepare_data()

    assert CountingDataset.count == 2

    data.setup("fit")

    assert CountingDataset.count == 4


def test_random_split_builds_the_source_only_once():
    data = AutoDataModule(dataset=counting_spec(n=10), random_split={"train": 8, "val": 2})
    data.setup("fit")

    assert CountingDataset.count == 1


def test_cross_val_builds_the_source_only_once():
    data = AutoDataModule(dataset=counting_spec(n=10), cross_val={"n_folds": 5, "fold_idx": 0})
    data.setup("fit")

    assert CountingDataset.count == 1


def test_repeated_setup_rebuilds():
    data = AutoDataModule(dataset={"train": counting_spec()})
    data.setup("fit")
    data.setup("fit")

    assert CountingDataset.count == 2


# ---------------------------------------------------------------------------
# Splitting through the data module
# ---------------------------------------------------------------------------


def test_random_split_sizes(cifar_train):
    data = AutoDataModule(dataset=cifar_train, random_split={"train": 0.9, "val": 0.1})
    data.setup("fit")

    assert len(data.get_dataset("train")) == 45000
    assert len(data.get_dataset("val")) == 5000


def test_cross_val_sizes(cifar_train):
    data = AutoDataModule(dataset=cifar_train, cross_val={"n_folds": 5, "fold_idx": 0})
    data.setup("fit")

    assert len(data.get_dataset("train")) == 40000
    assert len(data.get_dataset("val")) == 10000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"random_split": {"train": 0.8, "val": 0.2}, "cross_val": {"n_folds": 5, "fold_idx": 0}},
        {"random_split": {"trian": 1.0}},
        {"cross_val": {"n_folds": 5}},
        {"cross_val": {"n_folds": 5, "fold_idx": 5}},
    ],
)
def test_invalid_split_configurations_fail_at_construction(kwargs):
    with pytest.raises((ValueError, TypeError)):
        AutoDataModule(dataset=counting_spec(), **kwargs)


def test_declaring_a_derived_phase_fails_at_construction():
    with pytest.raises(ValueError, match="only one of them"):
        AutoDataModule(
            dataset={"train": counting_spec(), "val": counting_spec()},
            cross_val={"n_folds": 5, "fold_idx": 0},
        )


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def add(n):
    def apply(x):
        return x + n

    return apply


def test_get_transform_without_configuration():
    data = AutoDataModule()

    assert data.get_transform("train") is None
    assert data.get_target_transform("train") is None


def test_get_transform_for_a_single_callable():
    tf = add(1)
    data = AutoDataModule(transforms=tf)

    assert data.get_transform("train") is tf


def test_get_transform_composes_a_list():
    data = AutoDataModule(transforms=[add(1), add(2)])

    assert data.get_transform("train")(0) == 3


def test_get_transform_applies_pre_phase_post_in_order():
    data = AutoDataModule(transforms={"pre": add(1), "train": add(10), "post": add(100), "val": add(1000)})

    assert data.get_transform("train")(0) == 111
    assert data.get_transform("val")(0) == 1101
    assert data.get_transform("test")(0) == 101


def test_get_transform_ignores_unrelated_phases():
    data = AutoDataModule(transforms={"train": add(1)})

    assert data.get_transform("val") is None


def test_target_transforms_are_built_separately():
    data = AutoDataModule(transforms={"train": add(1)}, target_transforms={"train": add(2)})

    assert data.get_transform("train")(0) == 1
    assert data.get_target_transform("train")(0) == 2


def test_transformed_dataset_is_the_raw_dataset_without_transforms():
    ds = PairDataset()
    data = AutoDataModule(dataset=ds)
    data.setup("fit")

    assert data.get_transformed_dataset("train") is ds


def test_transformed_dataset_applies_transforms():
    data = AutoDataModule(
        dataset=PairDataset(), transforms={"train": add(1)}, target_transforms={"train": add(2)}
    )
    data.setup("fit")

    transformed = data.get_transformed_dataset("train")

    assert isinstance(transformed, Transformed)
    assert transformed[0] == (1, 2)


def test_transformed_dataset_uses_the_iterable_wrapper_without_len():
    data = AutoDataModule(dataset=StreamDataset(), transforms={"train": add(1)})
    data.setup("fit")

    assert isinstance(data.get_transformed_dataset("train"), TransformedIterable)


# ---------------------------------------------------------------------------
# Pre-loading
# ---------------------------------------------------------------------------


def test_pre_load_wraps_the_dataset():
    data = AutoDataModule(dataset=PairDataset(), pre_load=True)
    data.setup("fit")

    assert isinstance(data.get_transformed_dataset("train"), PreLoaded)


def test_pre_load_per_phase():
    data = AutoDataModule(dataset={"train": PairDataset(), "val": PairDataset()}, pre_load={"train": True})
    data.setup("fit")

    assert isinstance(data.get_transformed_dataset("train"), PreLoaded)
    assert not isinstance(data.get_transformed_dataset("val"), PreLoaded)


def test_pre_load_transform_is_applied_before_loading():
    data = AutoDataModule(dataset=PairDataset(), pre_load=True, transforms={"pre_load": add(1)})
    data.setup("fit")

    assert data.get_transformed_dataset("train")[0] == (1, 0)


def test_pre_load_transform_without_pre_load_is_rejected():
    data = AutoDataModule(dataset=PairDataset(), pre_load={}, transforms={"pre_load": add(1)})
    data.setup("fit")

    with pytest.raises(ValueError, match="pre-load is not enabled"):
        data.get_transformed_dataset("train")


@pytest.mark.xfail(
    reason="`not any(self.pre_load)` tests the dict's keys, not its values, so a dict of "
    "all-False flags is treated as pre-loading being enabled somewhere",
    strict=True,
)
def test_pre_load_transform_with_all_phases_disabled_is_rejected():
    data = AutoDataModule(
        dataset=PairDataset(), pre_load={"train": False, "val": False}, transforms={"pre_load": add(1)}
    )
    data.setup("fit")

    with pytest.raises(ValueError, match="pre-load is not enabled"):
        data.get_transformed_dataset("train")


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def test_dataloaders_per_phase(cifar_train, cifar_val):
    data = AutoDataModule(
        dataset=dict(train=cifar_train, val=cifar_val),
        dataloaders=dict(train=dict(batch_size=32), val=dict(batch_size=64)),
    )

    data.prepare_data()
    data.setup("fit")

    train_dl = data.train_dataloader()
    val_dl = data.val_dataloader()

    assert train_dl.dataset is cifar_train
    assert val_dl.dataset is cifar_val

    assert train_dl.batch_size == 32
    assert val_dl.batch_size == 64


def test_test_and_predict_dataloaders():
    data = AutoDataModule(
        dataset={"test": PairDataset(n=4), "pred": PairDataset(n=6)},
        dataloaders={"defaults": {"batch_size": 2}, "pred": {"batch_size": 3}},
    )
    data.setup("test")
    data.setup("predict")

    assert data.test_dataloader().batch_size == 2
    assert data.predict_dataloader().batch_size == 3


def test_dataloader_for_an_unconfigured_phase():
    data = AutoDataModule(dataset={"train": PairDataset()})
    data.setup("fit")

    with pytest.raises(KeyError, match="No dataset is configured for phase 'val'"):
        data.val_dataloader()


def test_dataloader_applies_transforms():
    data = AutoDataModule(dataset=PairDataset(), transforms={"train": add(1)})
    data.setup("fit")

    assert data.train_dataloader().dataset[0] == (1, 0)


# ---------------------------------------------------------------------------
# Batch transfer hooks
# ---------------------------------------------------------------------------


def test_batch_transfer_without_transforms_passes_the_batch_through():
    data = AutoDataModule()
    batch = ([1, 2], [3, 4])

    assert data.on_before_batch_transfer(batch, 0) is batch
    assert data.on_after_batch_transfer(batch, 0) is batch


def test_batch_transforms_combined_receive_the_whole_batch():
    data = AutoDataModule(batch_transforms={"before": lambda batch: (batch[1], batch[0])})

    assert data.on_before_batch_transfer((1, 2), 0) == (2, 1)
    assert data.on_after_batch_transfer((1, 2), 0) == (1, 2)


def test_batch_transforms_before_and_after_are_independent():
    data = AutoDataModule(batch_transforms={"before": add(1), "after": add(10)})

    assert data.on_before_batch_transfer(0, 0) == 1
    assert data.on_after_batch_transfer(0, 0) == 10


def test_batch_transforms_are_applied_in_sequence():
    data = AutoDataModule(batch_transforms={"before": [add(1), add(2)]})

    assert data.on_before_batch_transfer(0, 0) == 3


def test_separate_input_and_target_batch_transforms():
    data = AutoDataModule(
        batch_transforms={"before": add(1)},
        target_batch_transforms={"before": add(10)},
    )

    assert data.on_before_batch_transfer((0, 0), 0) == (1, 10)


def test_target_batch_transform_alone_leaves_the_input_untouched():
    data = AutoDataModule(target_batch_transforms={"before": add(10)})

    assert data.on_before_batch_transfer((0, 0), 0) == (0, 10)
