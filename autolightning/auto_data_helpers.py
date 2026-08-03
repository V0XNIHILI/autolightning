from typing import Any, Dict, Optional, Union, List

import torch
from torch.utils.data import (
    Dataset,
    IterableDataset,
    Subset,
    random_split as torch_random_split,
)

from jsonargparse import Namespace

from pytorch_lightning.cli import instantiate_class

from autolightning.types import DatasetType, PHASES


ALLOWED_DATASET_KEYS = PHASES + ["defaults"]
PRE_LOAD_MOMENT = "pre_load"
ARGS_KEY = "args"
FOLD_IDX_KEY = "fold_idx"
N_FOLDS_KEY = "n_folds"
CLASS_KEY = "class_name"
DEFAULTS_KEY = "defaults"
TRAIN = "train"

STAGE_PHASES = {
    "fit": ["train", "val"],
    "validate": ["val"],
    "test": ["test"],
    "predict": ["pred"],
}

AllDatasetsType = Union[DatasetType, Dict]

# A spec is everything needed to produce one dataset: either an already built
# dataset, or a {"class_name": ..., "args": {...}} dictionary.
DatasetSpec = Union[Dataset, Dict[str, Any]]


def _is_dataset(obj: Any) -> bool:
    return isinstance(obj, (Dataset, IterableDataset))


# ---------------------------------------------------------------------------
# 1. Normalization: every supported configuration shape -> {phase: spec}
# ---------------------------------------------------------------------------


def normalize_dataset_config(
    dataset: Optional[Union[Dict[str, AllDatasetsType], AllDatasetsType]],
) -> Dict[str, DatasetSpec]:
    """Rewrite any supported dataset configuration into a `{phase: spec}` dictionary.

    Supported shapes:
      - `None`                                        -> `{}`
      - a `Dataset` instance                          -> `{"train": ds}`
      - `{"class_name": X, "args": {...}}`            -> `{"train": spec}`
      - `{"class_name": X, "args": {"defaults": {...}, "train": {...}, ...}}`
                                                      -> one spec per phase
      - `{"train": <Dataset|spec>, "val": ...}`       -> one spec per phase

    This function performs no instantiation and knows nothing about splitting.
    """

    if dataset is None:
        return {}

    if _is_dataset(dataset):
        return {TRAIN: dataset}

    if not isinstance(dataset, dict):
        raise ValueError(
            f"Unsupported dataset configuration: {dataset}; can either be None, a Dataset instance or a dictionary"
        )

    if CLASS_KEY in dataset:
        return _normalize_single_class(dataset)

    if any(key in dataset for key in PHASES):
        return _normalize_per_phase(dataset)

    raise ValueError(
        f"Unsupported dataset configuration: {dataset}; should either be a dataset instance, a dictionary "
        f"with a '{CLASS_KEY}' key or a dictionary with dataset instances for one or more phases"
    )


def _normalize_single_class(dataset: Dict[str, Any]) -> Dict[str, DatasetSpec]:
    """Handle `{"class_name": ..., "args": ...}`, expanding per-phase args if present."""

    args = dataset.get(ARGS_KEY) or {}

    if not isinstance(args, dict):
        raise ValueError(f"Unsupported dataset configuration; '{ARGS_KEY}' must be a dictionary, but got {type(args)}")

    phase_keys = [phase for phase in PHASES if phase in args]

    if not phase_keys:
        if DEFAULTS_KEY in args:
            raise ValueError(
                f"Unsupported dataset configuration; '{DEFAULTS_KEY}' in '{ARGS_KEY}' is only meaningful "
                f"together with per-phase argument dictionaries ({PHASES})"
            )

        return {TRAIN: dataset}

    unknown = set(args) - set(ALLOWED_DATASET_KEYS)

    if unknown:
        raise ValueError(
            f"Unsupported dataset configuration; if any of the keys {PHASES} are present in the '{ARGS_KEY}' "
            f"dictionary, the only other allowed key is '{DEFAULTS_KEY}', but also got {sorted(unknown)}"
        )

    non_dict = sorted(key for key, value in args.items() if not isinstance(value, dict))

    if non_dict:
        raise ValueError(
            f"Unsupported dataset configuration; if any of the keys {PHASES} are present in the '{ARGS_KEY}' "
            f"dictionary, they must all be dictionaries, but {non_dict} are not"
        )

    defaults = args.get(DEFAULTS_KEY, {})

    return {phase: {CLASS_KEY: dataset[CLASS_KEY], ARGS_KEY: dict(defaults) | args[phase]} for phase in phase_keys}


def _normalize_per_phase(dataset: Dict[str, Any]) -> Dict[str, DatasetSpec]:
    """Handle `{"train": <Dataset|spec>, "val": <Dataset|spec>, ...}`."""

    unknown = set(dataset) - set(PHASES)

    if unknown:
        raise ValueError(f"Unsupported phase key(s) in dataset configuration: {sorted(unknown)}")

    for phase, spec in dataset.items():
        if not _is_dataset(spec) and not (isinstance(spec, dict) and CLASS_KEY in spec):
            raise ValueError(
                f"Unsupported dataset configuration for phase '{phase}'; should be a Dataset instance or a "
                f"dictionary with a '{CLASS_KEY}' key: {spec}"
            )

    return dict(dataset)


# ---------------------------------------------------------------------------
# 2. Instantiation: one spec -> one dataset
# ---------------------------------------------------------------------------


def instantiate_dataset(spec: DatasetSpec) -> Dataset:
    if _is_dataset(spec):
        return spec

    class_path = spec[CLASS_KEY]

    # There is this weird bug, where the class name can be a Namespace object with empty args
    # if this dictionary is set via a YAML file
    if isinstance(class_path, Namespace):
        class_path = dict(class_path)["class_path"]

    init_args = spec.get(ARGS_KEY) or {}

    if isinstance(class_path, str):
        return instantiate_class(tuple(), {"class_path": class_path, "init_args": init_args})

    if isinstance(class_path, type) and issubclass(class_path, (Dataset, IterableDataset)):
        return class_path(**init_args)

    raise ValueError(
        f"Unsupported dataset configuration; '{CLASS_KEY}' must be a string or a Dataset subclass, "
        f"but got {type(class_path)}"
    )


# ---------------------------------------------------------------------------
# 3. Split strategies: derive phases from the 'train' dataset
# ---------------------------------------------------------------------------


class _SplitStrategy:
    """Produces one or more phases by splitting the configured 'train' dataset.

    `produces` are all phases the strategy returns; `derived` are the phases that
    therefore may not be declared explicitly. They differ for random splitting,
    where 'train' is both the source and one of the outputs.
    """

    name: str
    produces: set
    derived: set

    def split(self, source: Dataset, seed: Optional[int]) -> Dict[str, Dataset]:
        raise NotImplementedError


class _RandomSplit(_SplitStrategy):
    name = "random_split"

    def __init__(self, config: Any):
        if not isinstance(config, dict):
            raise TypeError(f"Unsupported random split configuration: {config}; must be a dictionary")

        unknown = set(config) - set(PHASES)

        if unknown:
            raise ValueError(f"Unsupported keys in random split configuration: {sorted(unknown)}")

        self.config = config
        self.produces = set(config)
        self.derived = self.produces - {TRAIN}

    def split(self, source: Dataset, seed: Optional[int]) -> Dict[str, Dataset]:
        generator = torch.Generator()

        if seed is not None:
            generator = generator.manual_seed(seed)

        splits = torch_random_split(source, list(self.config.values()), generator=generator)

        return dict(zip(self.config.keys(), splits))


class _CrossVal(_SplitStrategy):
    name = "cross_val"
    produces = {TRAIN, "val"}
    derived = {"val"}

    def __init__(self, config: Any):
        if not isinstance(config, dict):
            raise TypeError(f"Unsupported cross-validation configuration: {config}; must be a dictionary")

        missing = {N_FOLDS_KEY, FOLD_IDX_KEY} - set(config)

        if missing:
            raise ValueError(f"Cross-validation configuration is missing the key(s) {sorted(missing)}")

        if not config[N_FOLDS_KEY] > config[FOLD_IDX_KEY] >= 0:
            raise ValueError(
                f"Invalid fold index {config[FOLD_IDX_KEY]} for {config[N_FOLDS_KEY]} folds; "
                f"expected 0 <= {FOLD_IDX_KEY} < {N_FOLDS_KEY}"
            )

        self.config = config

    def split(self, source: Dataset, seed: Optional[int]) -> Dict[str, Dataset]:
        from sklearn.model_selection import KFold

        shuffle = seed is not None

        kf = KFold(n_splits=self.config[N_FOLDS_KEY], shuffle=shuffle, random_state=seed)

        train_indices, val_indices = list(kf.split(source))[self.config[FOLD_IDX_KEY]]

        return {TRAIN: Subset(source, train_indices), "val": Subset(source, val_indices)}


# ---------------------------------------------------------------------------
# 4. Plan: which phases are declared, which are derived
# ---------------------------------------------------------------------------


class DatasetPlan:
    """Resolved, dataset-free answer to 'where does each phase come from?'.

    All configuration-level validation happens here, once, at construction time.
    """

    def __init__(self, declared: Dict[str, DatasetSpec], strategy: Optional[_SplitStrategy]):
        self.declared = declared
        self.strategy = strategy

        if strategy is None:
            return

        if TRAIN not in declared:
            raise ValueError(f"'{strategy.name}' splits the '{TRAIN}' dataset, but no '{TRAIN}' dataset is configured")

        clash = sorted(strategy.derived & set(declared))

        if clash:
            raise ValueError(
                f"'{strategy.name}' is specified, but dataset(s) for phase(s) {clash} are also provided; "
                f"only one of them can be used at a time."
            )

    @property
    def available_phases(self) -> set:
        """Every phase this configuration can produce, whether declared or derived."""

        phases = set(self.declared)

        if self.strategy is not None:
            phases |= self.strategy.produces

        return phases

    def instantiate_all(self) -> None:
        """Build every declared dataset once, e.g. to trigger downloads in `prepare_data`."""

        for spec in self.declared.values():
            instantiate_dataset(spec)

    def build(self, phases: List[str], seed: Optional[int]) -> Dict[str, Dataset]:
        """Build whichever of `phases` this configuration can produce.

        Phases that are not configured are simply absent from the result; it is up
        to the caller to complain if and when such a dataset is actually needed.
        """

        built = {phase: instantiate_dataset(self.declared[phase]) for phase in phases if phase in self.declared}

        if self.strategy is not None and any(phase in phases for phase in self.strategy.produces):
            source = built[TRAIN] if TRAIN in built else instantiate_dataset(self.declared[TRAIN])

            for phase, split in self.strategy.split(source, seed).items():
                if phase in phases:
                    built[phase] = split

        return built


def build_dataset_plan(
    dataset: Optional[Union[Dict[str, AllDatasetsType], AllDatasetsType]],
    random_split: Optional[Dict[str, Union[int, float]]] = None,
    cross_val: Optional[Dict[str, int]] = None,
) -> DatasetPlan:
    if random_split and cross_val:
        raise ValueError("Both random_split and cross_val are specified; only one of them can be used at a time.")

    strategy = None

    if random_split:
        strategy = _RandomSplit(random_split)
    elif cross_val:
        strategy = _CrossVal(cross_val)

    return DatasetPlan(normalize_dataset_config(dataset), strategy)
