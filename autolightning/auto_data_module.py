from typing import Dict, Optional, Union, Callable, Literal

import lightning as L

from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset
)
from torchvision.transforms import Compose

from torch_mate.data.utils import Transformed, TransformedIterable, PreLoaded

from autolightning.types import (
    AllDatasetsType,
    DatasetType,
    Phase,
    TransformType,
    TransformValue,
    PHASES,
)
from autolightning.auto_data_helpers import STAGE_PHASES, build_dataset_plan


ALLOWED_DATASET_KEYS = PHASES + ["defaults"]
PRE_LOAD_MOMENT = "pre_load"
ARGS_KEY = "args"
FOLD_IDX_KEY = "fold_idx"
N_FOLDS_KEY = "n_folds"


def compose_if_list(tf: Optional[TransformValue]) -> Optional[Callable]:
    if type(tf) is list:
        if len(tf) == 0:
            return None

        if len(tf) == 1:
            return tf[0]

        return Compose(tf)

    return tf


def build_transform(phase: str, transforms: TransformType) -> Callable | None:
    if type(transforms) is not dict:
        return compose_if_list(transforms)

    tfs = []

    for key in ["pre", phase, "post"]:
        if key in transforms:
            tf = compose_if_list(transforms[key])
            tfs.append(tf)

    return compose_if_list(tfs)


def iteratively_apply_transform(batch, transforms):
    if type(transforms) is list:
        for tf in transforms:
            batch = tf(batch)
    else:
        batch = transforms(batch)

    return batch


def apply_batch_transforms(batch, key: str, transforms: dict, target_batch_transforms: Union[dict, str]):
    tf = transforms.get(key, None)

    def id(x):
        return x

    if tf is not None:
        if target_batch_transforms == "combine":
            return iteratively_apply_transform(batch, tf)

        tft = target_batch_transforms.get(key, None) or id

        x, y = batch

        return iteratively_apply_transform(x, tf), iteratively_apply_transform(y, tft)

    # If tf is already None and target_batch_transforms is also not specified, we return the batch as is
    if target_batch_transforms == "combine":
        return batch

    tft = target_batch_transforms.get(key, None) or id

    x, y = batch

    return x, iteratively_apply_transform(y, tft)


class AutoDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Optional[Union[Dict[str, AllDatasetsType], AllDatasetsType]] = None,
        dataloaders: Optional[Dict] = None,
        transforms: Optional[TransformType] = None,
        target_transforms: Optional[TransformType] = None,
        batch_transforms: Optional[TransformType] = None,
        target_batch_transforms: Optional[Union[TransformType, Literal["combine"]]] = "combine",
        requires_prepare: bool = True,
        pre_load: Union[Dict[str, bool], bool] = False,
        random_split: Optional[Dict[str, Union[int, float]]] = None,
        cross_val: Optional[Dict[str, int]] = None,
        seed: Optional[int] = 42,
        build_plan: bool = True
    ):
        """Lightweight wrapper around PyTorch Lightning LightningDataModule that adds support for configuration via a dictionary.

        Overall, compared to the PyTorch Lightning LightningModule, the following two attributes are added:
        - `self.get_dataset(self, phase)`: a function that returns the dataset for a given phase

        Based on these, the following methods are automatically implemented:
        - `self.train_dataloader(self)`: calls `DataLoader(self.get_dataset('train'), **self.train_dataloader_kwargs)`
        - `self.val_dataloader(self)`: calls `DataLoader(self.get_dataset('val'), **self.val_dataloader_kwargs)`
        - `self.test_dataloader(self)`: calls `DataLoader(self.get_dataset('test'), **self.test_dataloader_kwargs)`
        - `self.predict_dataloader(self)`: calls `DataLoader(self.get_dataset('pred'), **self.test_dataloader_kwargs)`
        - `self.on_before_batch_transfer(self, batch, dataloader_idx)`: calls `self.reshape_batch_during_transfer(batch, dataloader_idx, "before")` followed by `self.post_transfer_batch_transform(batch)`
        - `self.on_after_batch_transfer(self, batch, dataloader_idx)`: calls `self.reshape_batch_during_transfer(batch, dataloader_idx, "after")` followed by `self.pre_transfer_batch_transform(batch)`

        Args:
            requires_prepare (bool):
                A boolean that specifies whether the dataset needs to be prepared before it can be used.
            pre_load (Union[Dict[str, bool], bool]):
                A boolean or dictionary that specifies whether to pre-load the dataset into memory before training.
                If a dictionary is specified, it must contain one or more of the keys `train`, `val`, `test` and
                `pred` to specify whether to pre-load the respective dataset. If a boolean is specified, it will
                be used as the default value for all phases.
            random_split (Optional[Dict[str, Union[int, float]]]):
                A dictionary that specifies how to split the dataset into `train`, `val`, `test` and `pred` sets.
                For each of these keys, it is possible to specify either a float or an integer to indicate the
                percentage or the number of samples to be used for the respective set. It is also possible split
                the `default` dataset into multiple sets by specifying the desired set keys and the number of
                samples per set in this dictionary.
            cross_val (Optional[Dict[str, int]]):
                A dictionary that specifies how to perform cross-validation on the dataset. The dictionary must
                contain the keys `n_folds` and `fold_idx` to specify the number of splits and the fold index to be
                used.
            seed (Optional[int]):
                Seed to be used for random splitting and cross-validation. If not specified, the dataset will not
                be shuffled before cross-validation.
            plan (bool):
                Whether to build the dataset plan during initialization. If set to False, the dataset plan will
                not be built and the dataset will not be instantiated until `setup` is called. This
        """

        super().__init__()

        self.dataset = dataset
        self.dataloaders = {} if dataloaders is None else dataloaders
        self.transforms = {} if transforms is None else transforms
        self.target_transforms = {} if target_transforms is None else target_transforms
        self.batch_transforms = {} if batch_transforms is None else batch_transforms
        self.target_batch_transforms = {} if target_batch_transforms is None else target_batch_transforms

        self.requires_prepare = requires_prepare
        self.pre_load = pre_load

        self.random_split = random_split
        self.cross_val = cross_val

        self.seed = seed

        self.build_plan = build_plan

        # All configuration-level validation happens here, before any dataset is built.
        if self.build_plan:
            self.plan = build_dataset_plan(dataset, random_split, cross_val)
        else:
            self.plan = None

        self.instantiated_dataset: Dict[str, Dataset] = {}

    def prepare_data(self) -> None:
        if self.requires_prepare and self.plan is not None:
            self.plan.instantiate_all()

    def setup(self, stage: str):
        phases = STAGE_PHASES.get(stage)

        if phases is None:
            return

        assert self.plan is not None, "Dataset plan must be built before calling setup"

        self.instantiated_dataset.update(self.plan.build(phases, self.seed))

    def has_dataset_in_plan(self, phase: Phase) -> bool:
        """Whether this configuration can produce a dataset for `phase` at all."""

        assert self.plan is not None, "Dataset plan must be built before calling has_dataset_in_plan"

        return phase in self.plan.available_phases

    def get_dataset(self, phase: Phase) -> Union[Dataset, IterableDataset]:
        if phase in self.instantiated_dataset:
            return self.instantiated_dataset[phase]

        if not self.has_dataset_in_plan(phase):
            raise KeyError(
                f"No dataset is configured for phase '{phase}'; configure one explicitly, or use "
                f"'random_split' or 'cross_val' to derive it from the 'train' dataset"
            )

        raise KeyError(
            f"Dataset for phase '{phase}' has not been built; make sure `setup` has been called for a "
            f"stage that includes this phase"
        )

    def get_transform(self, stage: str):
        return build_transform(stage, self.transforms)

    def get_target_transform(self, stage: str):
        return build_transform(stage, self.target_transforms)

    def get_transformed_dataset(self, phase: Phase):
        dataset = self.get_dataset(phase)

        if (isinstance(self.pre_load, bool) and self.pre_load) or (
            isinstance(self.pre_load, dict) and self.pre_load.get(phase, False)
        ):
            pre_load_tf = None
            pre_load_target_tf = None

            if isinstance(self.transforms, dict):
                pre_load_tf = compose_if_list(self.transforms.get(PRE_LOAD_MOMENT, None))
                pre_load_target_tf = compose_if_list(self.target_transforms.get(PRE_LOAD_MOMENT, None))

            if pre_load_tf is not None or pre_load_target_tf is not None:
                dataset = Transformed(dataset, pre_load_tf, pre_load_target_tf)

            dataset = PreLoaded(dataset)
        elif isinstance(self.transforms, dict) and isinstance(self.pre_load, dict) and not any(self.pre_load.values()):
            # Check if a pre-load transform is specified but not used at all in any of the phases
            if self.transforms.get(PRE_LOAD_MOMENT, None) is not None:
                raise ValueError(f"Pre-load transform specified for phase {phase} but pre-load is not enabled")
            elif self.target_transforms.get(PRE_LOAD_MOMENT, None) is not None:
                raise ValueError(f"Pre-load target transform specified for phase {phase} but pre-load is not enabled")

        transform = self.get_transform(phase)
        target_transform = self.get_target_transform(phase)

        if transform is None and target_transform is None:
            return dataset

        if not hasattr(dataset, "__len__"):
            return TransformedIterable(dataset, transform, target_transform)
        else:
            return Transformed(dataset, transform, target_transform)

    def get_dataloader_kwargs(self, phase: Phase, dataset: DatasetType) -> dict:
        # If the dataloader configuration is specified per phase...
        if any(key in self.dataloaders for key in ALLOWED_DATASET_KEYS):
            unsupported_keys = set(self.dataloaders.keys()) - set(ALLOWED_DATASET_KEYS)

            assert unsupported_keys == set(), (
                f"Unsupported keys in dataloader configuration: {unsupported_keys}; only {ALLOWED_DATASET_KEYS} are allowed"
            )

            kwargs = self.dataloaders.get("defaults", {}) | self.dataloaders.get(phase, {})
        else:
            kwargs = self.dataloaders

        return kwargs

    def get_dataloader(self, phase: Phase):
        dataset = self.get_transformed_dataset(phase)
        kwargs = self.get_dataloader_kwargs(phase, dataset)
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def predict_dataloader(self):
        return self.get_dataloader("pred")

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        return apply_batch_transforms(batch, "before", self.batch_transforms, self.target_batch_transforms)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return apply_batch_transforms(batch, "after", self.batch_transforms, self.target_batch_transforms)
