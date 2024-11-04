from torch_mate.data.utils import FewShot as FS

from .. import AutoDataModule
from ..types import Unpack, AutoDataModuleKwargs


class FewShotMixin:
    def __init__(self,
                 ways: int = 5,
                 shots: int = 1,
                 train_ways: int = -1,
                 query_shots: int = -1,
                 query_ways: int = -1,
                 train_query_shots: int = -1,
                 keep_original_labels: bool = False,
                 shuffle_labels: bool = False,
                 **kwargs: Unpack[AutoDataModuleKwargs]):
        super().__init__(**kwargs)
        
        self.ways = ways
        self.shots = shots
        self.train_ways = train_ways
        self.query_shots = query_shots
        self.query_ways = query_ways
        self.train_query_shots = train_query_shots
        self.keep_original_labels = keep_original_labels
        self.shuffle_labels = shuffle_labels

    def get_transformed_dataset(self, phase: str):
        dataset = super().get_transformed_dataset(phase)

        ways = self.ways
        query_shots = self.query_shots

        if phase == 'train':
            if self.train_ways != -1:
                ways = self.train_ways

            if self.train_query_shots != -1:
                query_shots = self.train_query_shots

        return FS(dataset, n_way=ways, k_shot=self.shots, query_shots=query_shots, query_ways=self.query_ways, keep_original_labels=self.keep_original_labels, shuffle_labels=self.shuffle_labels)


class FewShot(FewShotMixin, AutoDataModule):
    pass
