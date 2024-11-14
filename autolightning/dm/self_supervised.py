from torch.utils.data import Dataset

from torch_mate.data.utils import Siamese
from torch_mate.data.utils import Triplet

from .. import AutoDataModule
from ..types import Unpack, AutoDataModuleKwargs


class SiameseDataMixin:
    def __init__(self, same_prob: float = 0.5, **kwargs: Unpack[AutoDataModuleKwargs]):
        super().__init__(**kwargs)

        self.same_prob = same_prob


class SiameseData(SiameseDataMixin, AutoDataModule):
    def get_transformed_dataset(self, phase: str):
        dataset = super().get_transformed_dataset(phase)

        return Siamese(dataset, same_prob=self.same_prob)


class TripletData(AutoDataModule):
    def get_transformed_dataset(self, phase: str):
        dataset = super().get_transformed_dataset(phase)

        return Triplet(dataset)
