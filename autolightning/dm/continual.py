from typing import Optional

from torch.utils.data import DataLoader, ChainDataset

from ..types import Phase, TransformValue


class ContinualDataMixin:
    def __init__(self, datasets, n_experiences: Optional[int] = None, experience_transforms: Optional[TransformValue] = None, epochs_per_dataset: int = 1, chain_val_datasets: bool = True, **kwargs):
        if n_experiences is not None:
            assert experience_transforms is None, "If n_experiences is provided, experience_transforms must be None"
        
        if experience_transforms is not None:
            assert n_experiences is None, "If experience_transforms is provided, n_experiences must be None"

        super().__init__(**kwargs)

        # datasets is a list of torch.util.Dataset
        self.datasets = datasets
        self.n_experiences = n_experiences
        self.curr_index = 0
        self.epochs_per_dataset = epochs_per_dataset
        self.chain_val_datasets = chain_val_datasets

    def get_dataset(self, phase: Phase):
        dataset = super().get_dataset(phase)

    def train_dataloader(self):
        return DataLoader(self.datasets[self.curr_index])

    def val_dataloader(self):
        if self.chain_val_datasets:
            return DataLoader(ChainDataset(self.datasets[:self.curr_index+1]))
        
        return [DataLoader(ds) for ds in self.datasets[:self.curr_index+1]]

    def test_dataloader(self):
        if self.chain_val_datasets:
            return DataLoader(ChainDataset(self.datasets[:self.curr_index+1]))

        return [DataLoader(ds) for ds in self.datasets[:self.curr_index+1]]

    def on_epoch_end(self):
        super().on_epoch_end()
        
        if self.trainer.current_epoch % self.epochs_per_dataset == 0:
            self.curr_index += 1

# apply transform to all classes, split classes in chunks,
# RETURN TASK_ID!
reload_dataloaders_every_epoch=True
