from typing import Dict, Optional, Union, List, Callable

import lightning as L

import torch
from torch.utils.data import DataLoader, Dataset, random_split as torch_random_split
from torchvision.transforms import Compose

from jsonargparse import Namespace

from pytorch_lightning.cli import instantiate_class

from .types import Phase

from torch_mate.data.utils import Transformed, PreLoaded

STAGES = ["train", "val", "test", "predict"]
ALLOWED_DATASET_KEYS = STAGES + ["defaults"]
PRE_LOAD_MOMENT = "pre_load"
ARGS_KEY = "args"

TransformValue = Union[List[Callable], Callable]
TransformType = Union[Dict[str, TransformValue], TransformValue]


def instantiate_datasets(dataset: Optional[Union[Dict[str, Dataset], Dict, Dataset]]) -> (Dataset | Dict[str, Dataset] | None):
    if dataset is None or isinstance(dataset, Dataset):
        return dataset

    if not isinstance(dataset, dict):
        raise ValueError(f"Unsupported dataset configuration: {dataset}; can either be None, a Dataset instance or a dictionary")
    
    # If the dictionary has any of the stages, then it is a dictionary of datasets per stage
    if any(key in dataset for key in STAGES):
        # Make sure no other keys are present except for the stages
        assert set(dataset.keys()) - set(STAGES) == set(), f"Unsupported keys in dataset configuration: {set(dataset.keys()) - set(STAGES)}"
        # Make sure all values are datasets
        assert all(isinstance(ds, Dataset) for ds in dataset.values()), f"Unsupported values in dataset configuration that are not an instance of a Dataset: {set(type(ds) for ds in dataset.values())}"

        return dataset
    
    if "class" in dataset:
        init = {"class_path": dataset["class"]}

        # There is this weird bug, where dataset["class"] can be a Namespace object with empty args
        # if this dictionary is set via a YAML file
        if isinstance(init["class_path"], Namespace):
            init["class_path"] = dict(init["class_path"])["class_path"]

        if ARGS_KEY in dataset and any(key in ALLOWED_DATASET_KEYS for key in dataset[ARGS_KEY].keys()) and all(isinstance(phase_val, dict) for phase_val in dataset[ARGS_KEY].values()):
            # If any of the keys "train", "val", "test" or "predict" are present and they are all dictionaries, we build the datasets separately

            # Make sure no other keys are present except for the stages
            assert set(dataset[ARGS_KEY].keys()) - set(ALLOWED_DATASET_KEYS) == set(), f"Unsupported keys in dataset configuration: {set(dataset['args'].keys()) - set(ALLOWED_DATASET_KEYS)}"

            defaults = dataset[ARGS_KEY].get("defaults", {})

            # only stage keys, then number of stages
            # stage keys and defaults, then number of stages + 1, except for when all stages are present
            keys_to_init = []

            if "defaults" in dataset[ARGS_KEY]:
                if len(dataset[ARGS_KEY]) == len(STAGES) + 1:
                    keys_to_init = STAGES
                else:
                    keys_to_init = dataset[ARGS_KEY]
            else:
                keys_to_init = dataset[ARGS_KEY]

            dataset_dict = {}
            
            for key in keys_to_init:
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


def build_transform(stage: str, transforms: TransformType) -> (Callable | None):
    if type(transforms) is not dict:
        return compose_if_list(transforms)

    tfs = []

    for key in ["pre", stage, "post"]:
        if key in transforms:
            tf = compose_if_list(transforms[key])
            tfs.append(tf)

    return compose_if_list(tfs)


class AutoDataModule(L.LightningDataModule):

    def __init__(self,
                 dataset: Optional[Union[Dict[str, Dataset], Dict, Dataset]] = None,
                 dataloaders: Optional[Dict] = None,
                 transforms: Optional[TransformType] = None,
                 target_transforms: Optional[TransformType] = None,
                 batch_transforms: Optional[TransformType] = None,
                 requires_prepare: bool = True,
                 pre_load: Union[Dict[str, bool], bool] = False,
                 random_split: Optional[Dict[str, Union[Union[int, float], Union[str, Dict[str, Union[int, float]]]]]] = None):
        """Lightweight wrapper around PyTorch Lightning LightningDataModule that adds support for configuration via a dictionary.

        Overall, compared to the PyTorch Lightning LightningModule, the following two attributes are added:
        - `self.get_dataset(self, phase)`: a function that returns the dataset for a given phase

        Based on these, the following methods are automatically implemented:
        - `self.train_dataloader(self)`: calls `DataLoader(self.get_dataset('train'), **self.train_dataloader_kwargs)`
        - `self.val_dataloader(self)`: calls `DataLoader(self.get_dataset('val'), **self.val_dataloader_kwargs)`
        - `self.test_dataloader(self)`: calls `DataLoader(self.get_dataset('test'), **self.test_dataloader_kwargs)`
        - `self.predict_dataloader(self)`: calls `DataLoader(self.get_dataset('predict'), **self.test_dataloader_kwargs)`
        - `self.on_before_batch_transfer(self, batch, dataloader_idx)`: calls `self.reshape_batch_during_transfer(batch, dataloader_idx, "before")` followed by `self.post_transfer_batch_transform(batch)`
        - `self.on_after_batch_transfer(self, batch, dataloader_idx)`: calls `self.reshape_batch_during_transfer(batch, dataloader_idx, "after")` followed by `self.pre_transfer_batch_transform(batch)`

        Args:
            cfg (Dict): configuration dictionary
        """

        super().__init__()

        self.dataset = dataset
        self.dataloaders = {} if dataloaders is None else dataloaders
        self.transforms = {} if transforms is None else transforms
        self.target_transforms = {} if target_transforms is None else target_transforms
        self.batch_transforms = {} if batch_transforms is None else batch_transforms

        self.requires_prepare = requires_prepare
        self.pre_load = pre_load

        self.random_split = random_split

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
        
        relevant_keys = []

        if stage == 'fit':
            relevant_keys = ['train', 'val']
        elif stage == 'test':
            relevant_keys = ['test']
        elif stage == 'predict':
            relevant_keys = ['predict']
        elif stage == 'validate':
            relevant_keys = ['val']

        generator = torch.Generator().manual_seed(42)

        if isinstance(datasets, Dataset):
            if isinstance(self.random_split, dict):
                assert set(self.random_split.keys()) - set(STAGES) == set(), f"Unsupported keys in random split configuration: {set(self.random_split.keys()) - set(STAGES)}"

                dataset_splits = torch_random_split(datasets, self.random_split.values(), generator=generator)
                datasets = dict(zip(self.random_split.keys(), dataset_splits))

                for phase_key in relevant_keys:
                    self.instantiated_dataset[phase_key] = datasets[phase_key]

                # TODO! ADD WARNING HERE!
            else:
                for phase_key in relevant_keys:
                    self.instantiated_dataset[phase_key] = datasets
        else:
            instantiate_dataset_keys = []

            for phase_key in relevant_keys:
                if phase_key in instantiate_dataset_keys:
                    continue

                new_phase_key = "defaults" if phase_key not in datasets else phase_key

                if new_phase_key == "defaults" and new_phase_key not in datasets:
                    raise ValueError(f"Phase key {phase_key} not found in dataset configuration; also no defaults found")

                dataset = datasets[new_phase_key]
                    
                if isinstance(self.random_split, dict) and self.random_split["source"] == new_phase_key:
                    dataset_splits = torch_random_split(dataset, self.random_split["dest"].values(), generator=generator)
                    
                    new_datasets = dict(zip(self.random_split["dest"].keys(), dataset_splits))

                    for split_key, split_dataset in new_datasets.items():
                        self.instantiated_dataset[split_key] = split_dataset

                    instantiate_dataset_keys.extend(new_datasets.keys())
                else:
                    self.instantiated_dataset[phase_key] = dataset
                    instantiate_dataset_keys.append(phase_key)

    def get_dataset(self, phase: Phase):
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
        
        return Transformed(dataset, transform, target_transform)
    
    def get_dataloader(self, phase: Phase):
        dataset = self.get_transformed_dataset(phase)

        # If the dataloader configuration is specified per phase...
        if any(key in self.dataloaders for key in ALLOWED_DATASET_KEYS):
            assert set(self.dataloaders.keys()) - set(ALLOWED_DATASET_KEYS) == set(), f"Unsupported keys in dataloader configuration: {set(self.dataloaders.keys()) - set(ALLOWED_DATASET_KEYS)}; only {ALLOWED_DATASET_KEYS} are allowed"

            kwargs = dict(self.dataloaders.get("defaults", {})) | self.dataloaders.get(phase, {})
        else:
            kwargs = self.dataloaders

        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self.get_dataloader('train')
    
    def val_dataloader(self):
        return self.get_dataloader('val')
    
    def test_dataloader(self):
        return self.get_dataloader('test')
    
    def predict_dataloader(self):
        return self.get_dataloader('predict')
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        tf = self.batch_transforms.get("pre", None)

        if tf is not None:
            return tf(batch)
        
        return batch
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        tf = self.batch_transforms.get("post", None)

        if tf is not None:
            return tf(batch)
        
        return batch
