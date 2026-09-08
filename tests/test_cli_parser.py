"""Guards for how jsonargparse resolves ``**kwargs: Unpack[TypedDict]`` signatures.

Since jsonargparse 4.34 (PEP 692 support), a ``**kwargs: Unpack[SomeTypedDict]``
annotation is authoritative: the parser builds its arguments from the TypedDict's
keys instead of following the ``super().__init__(**kwargs)`` chain. Any key missing
from the TypedDict is therefore rejected on the CLI, and any type that is narrower
than the real parameter silently rejects valid configs.

These tests fail loudly if the TypedDicts in ``autolightning.types`` drift away from
the signatures they mirror.
"""

import inspect
import typing

import pytest
from lightning.pytorch.cli import LightningArgumentParser

from autolightning import types as T
from autolightning.auto_module import AutoModule
from autolightning.auto_data_module import AutoDataModule
from autolightning.datasets import MNIST


# Keys a TypedDict deliberately drops because the subclass supplies them itself
# (e.g. RootDownloadTrain builds `dataset`/`random_split` from `root`/`val_percentage`).
CASES = [
    (T.AutoModuleKwargs, AutoModule, set()),
    (T.AutoModuleKwargsNoCriterion, AutoModule, {"criterion"}),
    (T.AutoModuleKwargsNoNet, AutoModule, {"net"}),
    (T.AutoModuleKwargsNoNetCriterion, AutoModule, {"net", "criterion"}),
    (T.AutoDataModuleKwargs, AutoDataModule, set()),
    (
        T.AutoDataModuleKwargsNoDatasetPrepareSplit,
        AutoDataModule,
        {"dataset", "random_split", "requires_prepare"},
    ),
]


@pytest.mark.parametrize("typed_dict,cls,omitted", CASES, ids=lambda v: getattr(v, "__name__", ""))
def test_typed_dict_covers_init_signature(typed_dict, cls, omitted):
    """Every real parameter must be reachable through the TypedDict."""
    params = {p for p in inspect.signature(cls.__init__).parameters if p != "self"}
    keys = set(typed_dict.__annotations__)

    assert keys - params == set(), f"{typed_dict.__name__} has keys {cls.__name__} does not accept"
    assert params - omitted - keys == set(), (
        f"{typed_dict.__name__} is missing parameters of {cls.__name__}; "
        "they would be rejected by the CLI on jsonargparse >= 4.34"
    )


@pytest.mark.parametrize("typed_dict,cls,omitted", CASES, ids=lambda v: getattr(v, "__name__", ""))
def test_typed_dict_types_match_init_signature(typed_dict, cls, omitted):
    """A narrower type in the TypedDict silently rejects otherwise valid configs."""
    hints = typing.get_type_hints(typed_dict)
    sig = inspect.signature(cls.__init__).parameters

    for key, annotation in hints.items():
        expected = sig[key].annotation
        assert annotation == expected, (
            f"{typed_dict.__name__}[{key!r}] is {annotation!r} but "
            f"{cls.__name__}.__init__ declares {expected!r}"
        )


def _parser():
    parser = LightningArgumentParser(exit_on_error=False)
    parser.add_class_arguments(MNIST, "data")
    return parser


def test_cli_accepts_per_phase_transforms():
    """The documented ``{phase: transform}`` mapping must survive Unpack resolution."""
    cfg = _parser().parse_args(
        [
            "--data.root=data",
            '--data.transforms={"post": [{"class_path": "torchvision.transforms.ToTensor"}]}',
        ]
    )
    assert "post" in cfg.data.transforms


@pytest.mark.parametrize(
    "arg,attr,value",
    [
        ("--data.build_plan=false", "build_plan", False),
        ("--data.target_batch_transforms=combine", "target_batch_transforms", "combine"),
        ("--data.seed=7", "seed", 7),
    ],
)
def test_cli_accepts_auto_data_module_kwargs(arg, attr, value):
    cfg = _parser().parse_args(["--data.root=data", arg])
    assert getattr(cfg.data, attr) == value
