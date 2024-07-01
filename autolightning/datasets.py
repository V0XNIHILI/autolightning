from typing import Dict

import torch
from torch.utils.data import random_split
import torchvision.datasets

from autolightning import AutoDataModule
from autolightning.utils import get_class_and_init


class MagicData(AutoDataModule):
    def __init__(self, cfg: Dict, root: str, **kwargs):
        """Support any dataset with the following call signature:

        ```python
        dataset = Dataset(root, train=True, **kwargs)
        ```

        Configuration:
        ```yaml
        cfg.dataset.cfg:
            # Name of the dataset class.
            name: torchvision.datasets.MNIST # or any other dataset class, or simply "MNIST" 
            # Percentage of the training set to use for validation (float). Defaults to 0.2.
            val_percentage (float): 0.1
        cfg.dataset.kwargs: # or use the **kwargs directly
            root: "./data"
            # any other arguments to pass to the dataset class
        ```

        Args:
            cfg (Dict): Configuration dictionary.
        """

        super().__init__(cfg)

        self.root = root
        self.name_and_config = {"name": self.hparams.dataset["cfg"]["name"], "cfg": kwargs}

    def prepare_data(self) -> None:
        get_class_and_init(torchvision.datasets, self.name_and_config, self.root, False)
        get_class_and_init(torchvision.datasets, self.name_and_config, self.root, True)

    def setup(self, stage: str):
        if stage == 'fit' or stage == 'validate':
            full_set = get_class_and_init(torchvision.datasets, self.name_and_config, self.root, True)
            
            val_percentage = self.hparams.dataset.get("cfg", {}).get("val_percentage", 0.2)

            train_set, val_set = random_split(
                full_set, [1 - val_percentage, val_percentage], generator=torch.Generator().manual_seed(self.hparams.seed) if self.hparams.get("seed", None) else None
            )

            if stage == 'fit':
                self.train_set = train_set

            self.val_set = val_set
        elif stage == 'test':
            self.test_set = get_class_and_init(torchvision.datasets, self.name_and_config, self.root, False)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def get_dataset(self, phase: str):
        if phase == 'train':
            return self.train_set
        elif phase == 'val':
            return self.val_set
        elif phase == 'test':
            return self.test_set
        
        raise ValueError(f"Unsupported phase: {phase}")
