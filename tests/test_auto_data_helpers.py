"""Tests for the dataset configuration, planning and splitting logic in auto_data_helpers."""

import pytest

from torch.utils.data import Subset
from jsonargparse import Namespace

from autolightning.datasets import DummyDataset
from autolightning.auto_data_module import AutoDataModule
from autolightning.auto_data_helpers import normalize_dataset_config


DUMMY_PATH = "autolightning.datasets.DummyDataset"

N = 10
N_TRAIN, N_VAL, N_TEST = 8, 1, 1


def dummy(**args):
    """A `{'class_name': ..., 'args': ...}` spec pointing at DummyDataset."""
    return {"class_name": DUMMY_PATH, "args": {"n": N, **args}}


class RecordingDataset(DummyDataset):
    """DummyDataset that remembers every instance, to check what gets built and when."""

    created = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        RecordingDataset.created.append(self)


@pytest.fixture(autouse=True)
def _reset_recording():
    RecordingDataset.created.clear()
    yield
    RecordingDataset.created.clear()


def indices_of(dataset):
    assert isinstance(dataset, Subset)
    return set(dataset.indices)


# ---------------------------------------------------------------------------
# Normalization: every config shape collapses to {phase: spec}
# ---------------------------------------------------------------------------


def test_none_declares_nothing():
    assert normalize_dataset_config(None) == {}


def test_dataset_instance_is_declared_as_train():
    ds = DummyDataset()
    assert normalize_dataset_config(ds) == {"train": ds}


def test_single_class_config_is_declared_as_train():
    assert set(normalize_dataset_config(dummy())) == {"train"}


def test_per_phase_args_expand_to_one_spec_per_phase():
    declared = normalize_dataset_config(
        {"class_name": DUMMY_PATH, "args": {"defaults": {"n": 4}, "train": {"tag": "tr"}, "test": {"tag": "te"}}}
    )

    assert set(declared) == {"train", "test"}
    assert declared["train"]["args"] == {"n": 4, "tag": "tr"}
    assert declared["test"]["args"] == {"n": 4, "tag": "te"}


def test_per_phase_declarations_are_passed_through():
    ds = DummyDataset()
    declared = normalize_dataset_config({"train": ds, "val": dummy()})

    assert declared["train"] is ds
    assert declared["val"]["class_name"] == DUMMY_PATH


@pytest.mark.parametrize(
    "config, message",
    [
        (42, "can either be None"),
        ({"foo": "bar"}, "Unsupported dataset configuration"),
        ({"train": DummyDataset(), "valid": DummyDataset()}, "Unsupported phase key"),
        ({"train": {"cls": DUMMY_PATH}}, "Unsupported dataset configuration for phase"),
        ({"class_name": DUMMY_PATH, "args": {"defaults": {"n": 4}}}, "only meaningful together with per-phase"),
        ({"class_name": DUMMY_PATH, "args": {"train": {"tag": "tr"}, "n": 4}}, "only other allowed key"),
        ({"class_name": DUMMY_PATH, "args": {"train": {"tag": "tr"}, "val": 4}}, "must all be dictionaries"),
    ],
)
def test_invalid_configurations_are_rejected(config, message):
    with pytest.raises((ValueError, TypeError), match=message):
        normalize_dataset_config(config)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_class_name_as_string_path():
    built = AutoDataModule(dataset=dummy(tag="tr")).plan.build(["train"], 42)["train"]

    assert isinstance(built, DummyDataset) and built.tag == "tr"


def test_class_name_as_class_object():
    built = AutoDataModule(dataset={"class_name": DummyDataset, "args": {"n": 7}}).plan.build(["train"], 42)["train"]

    assert len(built) == 7


def test_class_name_as_namespace():
    # jsonargparse can hand us a Namespace instead of a plain string when the
    # configuration comes from a YAML file.
    dm = AutoDataModule(dataset={"class_name": Namespace(class_path=DUMMY_PATH), "args": {"n": 5}})

    assert len(dm.plan.build(["train"], 42)["train"]) == 5


def test_class_name_of_the_wrong_type_is_rejected():
    dm = AutoDataModule(dataset={"class_name": 42})

    with pytest.raises(ValueError, match="must be a string or a Dataset subclass"):
        dm.setup("fit")


def test_defaults_are_merged_and_overridden_per_phase():
    dm = AutoDataModule(
        dataset={
            "class_name": DUMMY_PATH,
            "args": {"defaults": {"n": N, "tag": "base"}, "train": {"tag": "tr"}, "val": {"n": 4}},
        }
    )
    dm.setup("fit")

    train, val = dm.instantiated_dataset["train"], dm.instantiated_dataset["val"]

    assert (train.tag, len(train)) == ("tr", N)
    assert (val.tag, len(val)) == ("base", 4)


def test_extra_args_are_forwarded_to_the_constructor():
    dm = AutoDataModule(dataset=dummy(extra="value"))

    assert dm.plan.build(["train"], 42)["train"].kwargs == {"extra": "value"}


def test_prepare_data_instantiates_every_declared_phase():
    dm = AutoDataModule(
        dataset={"class_name": RecordingDataset, "args": {"train": {"tag": "tr"}, "test": {"tag": "te"}}}
    )
    dm.prepare_data()

    assert sorted(ds.tag for ds in RecordingDataset.created) == ["te", "tr"]


def test_prepare_data_is_skipped_when_not_required():
    dm = AutoDataModule(dataset={"class_name": RecordingDataset}, requires_prepare=False)
    dm.prepare_data()

    assert RecordingDataset.created == []


def test_setup_only_instantiates_the_phases_it_needs():
    dm = AutoDataModule(
        dataset={"class_name": RecordingDataset, "args": {"train": {"tag": "tr"}, "test": {"tag": "te"}}}
    )
    dm.setup("test")

    assert [ds.tag for ds in RecordingDataset.created] == ["te"]


# ---------------------------------------------------------------------------
# Per-phase datasets, no splitting
# ---------------------------------------------------------------------------


def test_fit_builds_train_and_val():
    dm = AutoDataModule(dataset={"train": dummy(tag="tr"), "val": dummy(tag="va")})
    dm.setup("fit")

    assert dm.instantiated_dataset["train"].tag == "tr"
    assert dm.instantiated_dataset["val"].tag == "va"


def test_setup_test_does_not_need_a_train_dataset():
    # Previously raised KeyError('train').
    dm = AutoDataModule(dataset={"train": dummy(), "test": dummy(tag="te")})
    dm.setup("test")

    assert set(dm.instantiated_dataset) == {"test"}
    assert dm.get_dataset("test").tag == "te"


def test_predict_stage_uses_the_pred_phase():
    dm = AutoDataModule(dataset={"train": dummy(), "pred": dummy(tag="pr")})
    dm.setup("predict")

    assert dm.instantiated_dataset["pred"].tag == "pr"


def test_setup_fit_works_without_a_validation_dataset():
    dm = AutoDataModule(dataset={"train": dummy()})
    dm.setup("fit")

    assert set(dm.instantiated_dataset) == {"train"}


def test_requesting_an_unconfigured_phase_reports_it():
    dm = AutoDataModule(dataset={"train": dummy()})
    dm.setup("fit")

    with pytest.raises(KeyError, match="No dataset is configured for phase 'val'"):
        dm.get_dataset("val")


def test_requesting_a_configured_phase_before_setup_reports_it():
    dm = AutoDataModule(dataset={"train": dummy(), "val": dummy()})

    with pytest.raises(KeyError, match="has not been built"):
        dm.get_dataset("val")


@pytest.mark.parametrize(
    "kwargs, phase, expected",
    [
        ({}, "train", True),
        ({}, "val", False),
        ({"random_split": {"train": 0.8, "val": 0.2}}, "val", True),
        ({"random_split": {"train": 0.8, "val": 0.2}}, "test", False),
        ({"cross_val": {"n_folds": 5, "fold_idx": 0}}, "val", True),
    ],
)
def test_has_dataset(kwargs, phase, expected):
    assert AutoDataModule(dataset=dummy(), **kwargs).has_dataset(phase) is expected


def test_unknown_stage_is_ignored():
    dm = AutoDataModule(dataset={"train": dummy()})
    dm.setup("something_else")

    assert dm.instantiated_dataset == {}


# ---------------------------------------------------------------------------
# Random splitting
# ---------------------------------------------------------------------------


def test_random_split_with_integers():
    dm = AutoDataModule(dataset=dummy(), random_split={"train": N_TRAIN, "val": N_VAL, "test": N_TEST})
    dm.setup("fit")

    train, val = dm.instantiated_dataset["train"], dm.instantiated_dataset["val"]

    assert (len(train), len(val)) == (N_TRAIN, N_VAL)
    assert not indices_of(train) & indices_of(val)


def test_random_split_with_fractions():
    dm = AutoDataModule(dataset=dummy(), random_split={"train": 0.8, "val": 0.2})
    dm.setup("fit")

    train, val = dm.instantiated_dataset["train"], dm.instantiated_dataset["val"]

    assert (len(train), len(val)) == (8, 2)
    assert not indices_of(train) & indices_of(val)


def test_random_split_only_returns_the_requested_phases():
    dm = AutoDataModule(dataset=dummy(), random_split={"train": N_TRAIN, "val": N_VAL, "test": N_TEST})
    dm.setup("test")

    assert set(dm.instantiated_dataset) == {"test"}
    assert len(dm.instantiated_dataset["test"]) == N_TEST


def test_random_split_replaces_train_rather_than_using_the_source():
    source = DummyDataset(n=N)
    dm = AutoDataModule(dataset=source, random_split={"train": N_TRAIN, "val": N - N_TRAIN})
    dm.setup("fit")

    assert dm.instantiated_dataset["train"] is not source
    assert len(dm.instantiated_dataset["train"]) == N_TRAIN


def test_random_split_without_the_requested_phase_reports_it_on_access():
    dm = AutoDataModule(dataset=dummy(), random_split={"train": 0.8, "val": 0.2})
    dm.setup("test")

    with pytest.raises(KeyError, match="No dataset is configured for phase 'test'"):
        dm.get_dataset("test")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def test_cross_val_produces_disjoint_folds():
    dm = AutoDataModule(dataset=dummy(), cross_val={"n_folds": 5, "fold_idx": 0})
    dm.setup("fit")

    train, val = dm.instantiated_dataset["train"], dm.instantiated_dataset["val"]

    assert (len(train), len(val)) == (8, 2)
    assert not indices_of(train) & indices_of(val)


def test_cross_val_folds_differ():
    def val_indices(fold_idx):
        dm = AutoDataModule(dataset=dummy(), cross_val={"n_folds": 5, "fold_idx": fold_idx})
        dm.setup("fit")
        return indices_of(dm.instantiated_dataset["val"])

    assert not val_indices(0) & val_indices(1)


def test_cross_val_folds_cover_the_dataset():
    seen = set()

    for fold_idx in range(5):
        dm = AutoDataModule(dataset=dummy(), cross_val={"n_folds": 5, "fold_idx": fold_idx})
        dm.setup("fit")
        seen |= indices_of(dm.instantiated_dataset["val"])

    assert seen == set(range(N))


def test_cross_val_with_per_phase_args():
    # Previously produced an empty configuration and fell through to a generic error.
    dm = AutoDataModule(
        dataset={"class_name": DUMMY_PATH, "args": {"defaults": {"n": N}, "train": {"tag": "tr"}}},
        cross_val={"n_folds": 5, "fold_idx": 0},
    )
    dm.setup("fit")

    assert len(dm.instantiated_dataset["train"]) + len(dm.instantiated_dataset["val"]) == N
    assert dm.instantiated_dataset["train"].dataset.tag == "tr"


def test_cross_val_with_a_single_class_config():
    # Previously dead: this shape never reached the cross-validation branch.
    dm = AutoDataModule(dataset=dummy(), cross_val={"n_folds": 5, "fold_idx": 0})
    dm.setup("fit")

    assert set(dm.instantiated_dataset) == {"train", "val"}


def test_cross_val_during_validate_stage():
    # Previously blocked by `assert stage == "fit"`.
    dm = AutoDataModule(dataset=dummy(), cross_val={"n_folds": 5, "fold_idx": 0})
    dm.setup("validate")

    assert set(dm.instantiated_dataset) == {"val"}


def test_cross_val_without_seed_does_not_shuffle():
    dm = AutoDataModule(dataset=dummy(), cross_val={"n_folds": 5, "fold_idx": 0}, seed=None)
    dm.setup("fit")

    assert indices_of(dm.instantiated_dataset["val"]) == {0, 1}


# ---------------------------------------------------------------------------
# Configuration-level validation (all of it at construction time)
# ---------------------------------------------------------------------------


def test_both_split_strategies_rejected():
    with pytest.raises(ValueError, match="only one of them"):
        AutoDataModule(
            dataset=dummy(), random_split={"train": 0.8, "val": 0.2}, cross_val={"n_folds": 5, "fold_idx": 0}
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cross_val": {"n_folds": 5, "fold_idx": 0}},
        {"random_split": {"train": 0.8, "val": 0.2}},
    ],
)
def test_declared_phase_clashing_with_a_derived_one_is_rejected(kwargs):
    with pytest.raises(ValueError, match="only one of them"):
        AutoDataModule(dataset={"train": dummy(), "val": dummy()}, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cross_val": {"n_folds": 5, "fold_idx": 0}},
        {"random_split": {"train": 0.8, "val": 0.2}},
    ],
)
def test_splitting_requires_a_train_dataset(kwargs):
    with pytest.raises(ValueError, match="no 'train' dataset is configured"):
        AutoDataModule(dataset={"test": dummy()}, **kwargs)


def test_a_bare_dataset_is_usable_as_the_train_split():
    ds = DummyDataset()
    dm = AutoDataModule(dataset=ds)
    dm.prepare_data()
    dm.setup("fit")

    assert dm.get_dataset("train") is ds


@pytest.mark.parametrize(
    "cross_val, message",
    [
        ({"n_folds": 3, "fold_idx": 3}, "Invalid fold index"),
        ({"n_folds": 3, "fold_idx": -1}, "Invalid fold index"),
        ({"n_folds": 3}, "missing the key"),
        ("five folds", "must be a dictionary"),
    ],
)
def test_invalid_cross_val_configuration(cross_val, message):
    with pytest.raises((ValueError, TypeError), match=message):
        AutoDataModule(dataset=dummy(), cross_val=cross_val)


@pytest.mark.parametrize(
    "random_split, message",
    [
        ({"trian": 1.0}, "Unsupported keys"),
        ("half", "must be a dictionary"),
    ],
)
def test_invalid_random_split_configuration(random_split, message):
    with pytest.raises((ValueError, TypeError), match=message):
        AutoDataModule(dataset=dummy(), random_split=random_split)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_setup_gives_the_same_split():
    dm = AutoDataModule(dataset=dummy(), random_split={"train": N_TRAIN, "val": N - N_TRAIN})
    dm.setup("fit")
    first = indices_of(dm.instantiated_dataset["val"])
    dm.setup("fit")

    assert indices_of(dm.instantiated_dataset["val"]) == first


def test_split_does_not_depend_on_which_stages_ran_before():
    def val_indices(stages):
        dm = AutoDataModule(dataset=dummy(), random_split={"train": N_TRAIN, "val": N_VAL, "test": N_TEST})
        for stage in stages:
            dm.setup(stage)
        return indices_of(dm.instantiated_dataset["val"])

    assert val_indices(["fit"]) == val_indices(["test", "fit"])


def test_different_seeds_give_different_splits():
    def val_indices(seed):
        dm = AutoDataModule(dataset=dummy(n=100), random_split={"train": 0.8, "val": 0.2}, seed=seed)
        dm.setup("fit")
        return indices_of(dm.instantiated_dataset["val"])

    assert val_indices(0) != val_indices(1234)
