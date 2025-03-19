from typing import Dict, Optional, Union, Callable, Literal

import warnings

import lightning as L

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split as torch_random_split
from torchvision.transforms import Compose

from jsonargparse import Namespace

from pytorch_lightning.cli import instantiate_class

from .types import Phase, TransformValue

from torch_mate.data.utils import Transformed, TransformedIterable, PreLoaded


PHASES = ["train", "val", "test", "pred"]
ALLOWED_DATASET_KEYS = PHASES + ["defaults"]
PRE_LOAD_MOMENT = "pre_load"
ARGS_KEY = "args"
FOLD_IDX_KEY = "fold_idx"
N_FOLDS_KEY = "n_folds"

AllDatasetsType = Union[Dataset, IterableDataset]
TransformType = Union[Dict[str, TransformValue], TransformValue]


def instantiate_datasets(dataset: Optional[Union[Dict[str, Dataset], Dict, Dataset]]) -> (Dataset | Dict[str, Dataset] | None):
    if dataset is None or isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
        return dataset

    if not isinstance(dataset, dict):
        raise ValueError(f"Unsupported dataset configuration: {dataset}; can either be None, a Dataset instance or a dictionary")
    
    # If the dictionary has any of the phases, then it is a dictionary of datasets per stage
    if any(key in dataset for key in PHASES):
        dataset_dict = {}

        for key, ds in dataset.items():
            if key not in PHASES:
                raise ValueError(f"Unsupported phase key in dataset configuration: {key}")
            
            if isinstance(ds, Dataset):
                dataset_dict[key] = ds
            elif "class_name" in ds:
                init = {"class_path": ds["class_name"], "init_args": ds.get(ARGS_KEY, {})}

                if type(init["class_path"]) is str:
                    dataset_dict[key] = instantiate_class(tuple(), init)
                else:
                    dataset_dict[key] = init["class_path"](**init["init_args"])

                dataset_dict[key] = instantiate_class(tuple(), init)
            else:
                raise ValueError(f"Unsupported dataset configuration; should be a Dataset instance or a dictionary with a 'class_name' key: {ds}")

        return dataset_dict
    
    if "class_name" in dataset:
        init = {"class_path": dataset["class_name"]}

        # There is this weird bug, where dataset["class_name"] can be a Namespace object with empty args
        # if this dictionary is set via a YAML file
        if isinstance(init["class_path"], Namespace):
            init["class_path"] = dict(init["class_path"])["class_path"]

        # If any of the keys "train", "val", "test" or "predict" are present and they are all dictionaries, we build the datasets separately
        if ARGS_KEY in dataset and any(key in ALLOWED_DATASET_KEYS for key in dataset[ARGS_KEY].keys()) and all(isinstance(phase_val, dict) for phase_val in dataset[ARGS_KEY].values()):
            # Make sure no other keys are present except for the stages
            assert set(dataset[ARGS_KEY].keys()) - set(ALLOWED_DATASET_KEYS) == set(), f"Unsupported keys in dataset configuration: {set(dataset['args'].keys()) - set(ALLOWED_DATASET_KEYS)}"

            defaults = dataset[ARGS_KEY].get("defaults", {})

            dataset_dict = {}

            contains_all_keys = all(key in dataset[ARGS_KEY] for key in ALLOWED_DATASET_KEYS)
            
            for key in dataset[ARGS_KEY].keys():
                # If all stage keys are present, then there is no need to instantiate the default dataset
                if contains_all_keys and key == "defaults":
                    continue

                init["init_args"] = dict(defaults) | (dataset[ARGS_KEY].get(key, {}))

                if type(init["class_path"]) is str:
                    dataset_dict[key] = instantiate_class(tuple(), init)
                else:
                    dataset_dict[key] = init["class_path"](**init["init_args"])

            return dataset_dict
        
        return instantiate_class(tuple(), init | {"init_args": dataset.get(ARGS_KEY, {})})
    
    raise ValueError(f"Unsupported dataset configuration: {dataset}")


def compose_if_list(tf: Optional[TransformValue]) -> Optional[Callable]:
    if type(tf) is list:
        if len(tf) == 0:
            return None
        
        if len(tf) == 1:
            return tf[0]
        
        return Compose(tf)
    
    return tf


def build_transform(phase: str, transforms: TransformType) -> (Callable | None):
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
    id = lambda x: x

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

    def __init__(self,
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
                 seed: Optional[int] = 42):
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

        # Perform XOR
        if self.cross_val and self.random_split:
            raise ValueError("Both random_split and cross_val are specified; only one of them can be used at a time.")

        self.seed = seed

        self.instantiated_dataset: Union[Dataset, Dict[str, Dataset]] = {}
        
    def prepare_data(self) -> None:
        if self.requires_prepare:
            instantiate_datasets(self.dataset)

    def get_transform(self, stage: str):
        return build_transform(stage, self.transforms)

    def get_target_transform(self, stage: str):
        return build_transform(stage, self.target_transforms)
    
    def setup(self, stage: str):
        datasets = instantiate_datasets(self.dataset)

        if datasets == None:
            return
        
        relevant_phases = []

        if stage == 'fit':
            relevant_phases = ['train', 'val']
        elif stage == 'test':
            relevant_phases = ['test']
        elif stage == 'predict':
            relevant_phases = ['pred']
        elif stage == 'validate':
            relevant_phases = ['val']

        generator = torch.Generator()

        if self.seed is not None:
            generator = generator.manual_seed(self.seed)

        if isinstance(datasets, Dataset):
            if self.cross_val:
                if not isinstance(self.cross_val, dict):
                    raise TypeError(f"Unsupported cross-validation configuration: {self.cross_val}; must be a dictionary")

                assert self.cross_val[N_FOLDS_KEY] > self.cross_val[FOLD_IDX_KEY] >= 0, f"Invalid fold index {self.cross_val['fold']} for {self.cross_val['n_folds']} splits"

                from sklearn.model_selection import KFold

                shuffle = self.seed is not None

                kf = KFold(n_folds=self.cross_val[N_FOLDS_KEY], shuffle=shuffle, random_state=self.seed)

                for i, (train_indices, val_indices) in enumerate(kf.split(datasets)):
                    if i == self.cross_val[FOLD_IDX_KEY]:
                        self.instantiated_dataset['train'] = torch.utils.data.Subset(datasets, train_indices)
                        self.instantiated_dataset['val'] = torch.utils.data.Subset(datasets, val_indices)
                        break
            elif self.random_split:
                if not isinstance(self.random_split, dict):
                    raise TypeError(f"Unsupported random split configuration: {self.random_split}; must be a dictionary")

                assert set(self.random_split.keys()) - set(PHASES) == set(), f"Unsupported keys in random split configuration: {set(self.random_split.keys()) - set(PHASES)}"

                dataset_splits = torch_random_split(datasets, self.random_split.values(), generator=generator)
                datasets = dict(zip(self.random_split.keys(), dataset_splits))

                for phase_key in relevant_phases:
                    self.instantiated_dataset[phase_key] = datasets[phase_key]
            else:
                for phase_key in relevant_phases:
                    if phase_key != 'train':
                        warnings.warn(f"Only one dataset was specified, but it will be used for multiple phases: {relevant_phases}")
            
                    self.instantiated_dataset[phase_key] = datasets
        else:
            instantiate_dataset_keys = []

            for phase_key in relevant_phases:
                if phase_key in instantiate_dataset_keys:
                    continue

                is_default = phase_key not in datasets
                new_phase_key = "defaults" if is_default else phase_key

                if is_default and "defaults" not in datasets:
                    raise ValueError(f"Phase key {phase_key} not found in dataset configuration; also no defaults found")

                dataset = datasets[new_phase_key]
                    
                if is_default and (self.random_split is not None or self.cross_val is not None):
                    if self.cross_val:
                        if not isinstance(self.cross_val, dict):
                            raise TypeError(f"Unsupported cross-validation configuration: {self.cross_val}; must be a dictionary")

                        assert self.cross_val[N_FOLDS_KEY] > self.cross_val[FOLD_IDX_KEY] >= 0, f"Invalid fold index {self.cross_val['fold']} for {self.cross_val['n_folds']} splits"

                        from sklearn.model_selection import KFold

                        shuffle = self.seed is not None

                        kf = KFold(n_folds=self.cross_val[N_FOLDS_KEY], shuffle=shuffle, random_state=self.seed)

                        for i, (train_indices, val_indices) in enumerate(kf.split(dataset)):
                            if i == self.cross_val[FOLD_IDX_KEY]:
                                self.instantiated_dataset['train'] = torch.utils.data.Subset(dataset, train_indices)
                                self.instantiated_dataset['val'] = torch.utils.data.Subset(dataset, val_indices)
                                instantiate_dataset_keys.extend(['train', 'val'])
                    else:
                        # Cannot create a random split for a dataset that is already specified
                        assert set(datasets.keys()).isdisjoint(set(self.random_split.keys())), f"Random split configuration contains keys that are already present in the dataset configuration: {set(datasets.keys()) & set(self.random_split.keys())}"

                        dataset_splits = torch_random_split(dataset, self.random_split["dest"].values(), generator=generator)
                        
                        new_datasets = dict(zip(self.random_split["dest"].keys(), dataset_splits))

                        for split_key, split_dataset in new_datasets.items():
                            self.instantiated_dataset[split_key] = split_dataset

                        instantiate_dataset_keys.extend(new_datasets.keys())
                else:
                    self.instantiated_dataset[phase_key] = dataset
                    instantiate_dataset_keys.append(phase_key)

    def get_dataset(self, phase: Phase) -> Union[Dataset, IterableDataset]:
        if phase not in self.instantiated_dataset:
            raise KeyError(f"Dataset for phase {phase} not found; make sure to call `setup` before accessing the dataset")
            
        return self.instantiated_dataset[phase]
    
    def get_transformed_dataset(self, phase: Phase):
        dataset = self.get_dataset(phase)

        if self.pre_load == True or (isinstance(self.pre_load, dict) and self.pre_load.get(phase, False)):
            pre_load_tf = None
            pre_load_target_tf = None
    
            if isinstance(self.transforms, dict):
                pre_load_tf = compose_if_list(self.transforms.get(PRE_LOAD_MOMENT, None))
                pre_load_target_tf = compose_if_list(self.target_transforms.get(PRE_LOAD_MOMENT, None))

            if pre_load_tf is not None or pre_load_target_tf is not None:
                dataset = Transformed(dataset, pre_load_tf, pre_load_target_tf)

            dataset = PreLoaded(dataset)
        elif isinstance(self.transforms, dict):
            if self.transforms.get(PRE_LOAD_MOMENT, None) != None:
                raise ValueError(f"Pre-load transform specified for phase {phase} but pre-load is not enabled")
            elif self.target_transforms.get(PRE_LOAD_MOMENT, None) != None:
                raise ValueError(f"Pre-load target transform specified for phase {phase} but pre-load is not enabled")

        transform = self.get_transform(phase)
        target_transform = self.get_target_transform(phase)

        if transform is None and target_transform is None:
            return dataset

        if not hasattr(dataset, '__len__'):
            return TransformedIterable(dataset, transform, target_transform)
        else:
            return Transformed(dataset, transform, target_transform)
        
    def get_dataloader_kwargs(self, phase: Phase):
        # If the dataloader configuration is specified per phase...
        if any(key in self.dataloaders for key in ALLOWED_DATASET_KEYS):
            unsupported_keys = set(self.dataloaders.keys()) - set(ALLOWED_DATASET_KEYS)

            assert unsupported_keys == set(), f"Unsupported keys in dataloader configuration: {unsupported_keys}; only {ALLOWED_DATASET_KEYS} are allowed"

            kwargs = self.dataloaders.get("defaults", {}) | self.dataloaders.get(phase, {})
        else:
            kwargs = self.dataloaders

        return kwargs
    
    def get_dataloader(self, phase: Phase):
        dataset = self.get_transformed_dataset(dataset)
        kwargs = self.get_dataloader_kwargs(phase)
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self.get_dataloader('train')
    
    def val_dataloader(self):
        return self.get_dataloader('val')
    
    def test_dataloader(self):
        return self.get_dataloader('test')
    
    def predict_dataloader(self):
        return self.get_dataloader('pred')
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        return apply_batch_transforms(
            batch, "before",
            self.batch_transforms,
            self.target_batch_transforms
        )
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return apply_batch_transforms(
            batch, "after",
            self.batch_transforms,
            self.target_batch_transforms
        )
