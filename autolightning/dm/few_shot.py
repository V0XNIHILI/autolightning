from torch_mate.data.utils import FewShot


class FewShotMixin:
    def __init__(self, ways: int = 5, shots: int = 1, train_ways: int = -1, query_shots: int = -1, query_ways: int = -1, train_query_shots: int = -1, **kwargs):
        super().__init__(**kwargs)
        
        self.ways = ways
        self.shots = shots
        self.train_ways = train_ways
        self.query_shots = query_shots
        self.query_ways = query_ways
        self.train_query_shots = train_query_shots

    def get_transformed_dataset(self, phase: str):
        dataset = super().get_transformed_dataset(phase)

        ways = self.ways
        query_shots = self.query_shots

        if phase == 'train':
            if self.train_ways != -1:
                ways = self.train_ways

            if self.train_query_shots != -1:
                query_shots = self.train_query_shots

        return FewShot(dataset, ways, self.shots, query_shots, self.query_ways)
