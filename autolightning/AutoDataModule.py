from typing import Dict
import inspect

import lightning as L

from torch.utils.data import DataLoader

from autolightning.utils import build_dataloader_kwargs, create_stage_transforms, build_transform
from torch_mate.data.utils import Transformed, PreLoaded

STAGES = ['train', 'val', 'test', 'predict']
MOMENTS = ["pre", "post"]
PRE_LOAD_MOMENT = "pre_load"


class AutoDataModule(L.LightningDataModule):

    def __init__(self, cfg: Dict):
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

        self.save_hyperparameters(self.configure_configuration(cfg))

        self._common_pre_transforms, self._common_post_transforms = [self.get_common_transform(m) for m in MOMENTS]
        self._common_pre_target_transforms, self._common_post_target_transforms = [self.get_common_target_transform(m) for m in MOMENTS]
        self._pre_transfer_batch_transform, self._post_transfer_batch_transform = [self.get_batch_transform(m) for m in MOMENTS]
        
        self._pre_load_transform = self.get_common_transform(PRE_LOAD_MOMENT)
        self._pre_load_target_transform = self.get_common_target_transform(PRE_LOAD_MOMENT)

        self._has_pre_load_transform = self._pre_load_transform is not None or self._pre_load_target_transform is not None

        if "pre_load" not in self.hparams.dataset.get("extra", {}) and self._has_pre_load_transform:
            raise ValueError("Either a regular or a target transform is provided for a pre-loaded dataset, but the dataset is not configured to be pre-loaded at any point")

    def configure_configuration(self, cfg: Dict):
        return cfg

    def get_common_transform(self, moment: str):
        return build_transform(self.hparams.dataset.get("transforms", {}).get(moment, []))
    
    def get_common_target_transform(self, moment: str):
        return build_transform(self.hparams.dataset.get("target_transforms", {}).get(moment, []))
    
    def get_batch_transform(self, moment: str):
        return build_transform(self.hparams.dataset.get("batch_transforms", {}).get(moment, []))

    def get_transform(self, stage: str):
        if "transforms" in self.hparams.dataset:
            return create_stage_transforms(
                self.hparams.dataset["transforms"].get(stage, None),
                self._common_pre_transforms if stage in STAGES else None,
                self._common_post_transforms if stage in STAGES else None
            )
        
        return None
    
    def get_target_transform(self, stage: str):
        if "target_transforms" in self.hparams.dataset:
            return create_stage_transforms(
                self.hparams.dataset["target_transforms"].get(stage, None),
                self._common_pre_target_transforms if stage in STAGES else None,
                self._common_post_target_transforms if stage in STAGES else None
            )
        
        return None
    
    def get_dataloader_kwargs(self, stage: str):
        dataloaders_cfg = self.hparams.get("dataloaders", {})
        
        return build_dataloader_kwargs(
            dataloaders_cfg,
            stage
        )

    def get_dataset(self, phase: str):
        raise NotImplementedError
    
    def get_transformed_dataset(self, phase: str):
        dataset = self.get_dataset(phase)

        # API for this entry in the configuration dict is up for change imo.
        if "pre_load" in self.hparams.dataset.get("extra", {}):
            if phase in self.hparams.dataset["extra"]["pre_load"]:
                if self._has_pre_load_transform:
                    dataset = Transformed(dataset, self._pre_load_transform, self._pre_load_target_transform)

                dataset = PreLoaded(dataset)

        transform = self.get_transform(phase)
        target_transform = self.get_target_transform(phase)

        if transform is None and target_transform is None:
            return dataset
        
        return Transformed(dataset, transform, target_transform)
    
    def get_dataloader(self, phase: str):
        dataset = self.get_transformed_dataset(phase)
        kwargs = self.get_dataloader_kwargs(phase)

        # Check if dataset is list or tuple and if the first element is a class
        if isinstance(dataset, (list, tuple)) and inspect.isclass(dataset[0]):
            return [DataLoader(ds, **kwargs) for ds in dataset]
        
        # Lightning only supports dict of dataloaders during training
        if phase == 'train' and isinstance(dataset, dict):
            return {k: DataLoader(v, **kwargs) for k, v in dataset.items()}

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
        if self._pre_transfer_batch_transform is not None:
            return self._pre_transfer_batch_transform(batch)
        
        return batch
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if self._post_transfer_batch_transform is not None:
            return self._post_transfer_batch_transform(batch)
        
        return batch
