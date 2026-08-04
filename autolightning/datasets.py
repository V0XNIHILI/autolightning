from torch.utils.data import Dataset

from . import AutoDataModule
from .types import AutoDataModuleKwargsNoDatasetPrepareSplit, Unpack


class RootDownloadTrain(AutoDataModule):
    def __init__(
        self,
        name: str,
        root: str,
        download: bool = False,
        val_percentage: float = 0.1,
        **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit],
    ):
        super().__init__(
            dataset={
                "class_name": f"torchvision.datasets.{name}" if "." not in name else name,
                "args": dict(
                    defaults=dict(root=root, train=True, download=download),
                    test=dict(train=False),
                ),
            },
            random_split={
                "train": 1 - val_percentage,
                "val": val_percentage
            },
            requires_prepare=download,
            **kwargs,
        )


class MNIST(RootDownloadTrain):
    def __init__(
        self,
        root: str,
        download: bool = False,
        val_percentage: float = 0.1,
        **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit],
    ):
        super().__init__(
            name="MNIST",
            root=root,
            download=download,
            val_percentage=val_percentage,
            **kwargs,
        )


class CIFAR10(RootDownloadTrain):
    def __init__(
        self,
        root: str,
        download: bool = False,
        val_percentage: float = 0.1,
        **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit],
    ):
        super().__init__(
            name="CIFAR10",
            root=root,
            download=download,
            val_percentage=val_percentage,
            **kwargs,
        )


class FashionMNIST(RootDownloadTrain):
    def __init__(
        self,
        root: str,
        download: bool = False,
        val_percentage: float = 0.1,
        **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit],
    ):
        super().__init__(
            name="FashionMNIST",
            root=root,
            download=download,
            val_percentage=val_percentage,
            **kwargs,
        )


class CIFAR100(RootDownloadTrain):
    def __init__(self, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit]):
        super().__init__(name="CIFAR100", root=root, download=download, val_percentage=val_percentage, **kwargs)


class _DummyDataset(Dataset):
    def __init__(self, n=10, tag="default", **kwargs):
        self.n = n
        self.tag = tag
        self.kwargs = kwargs

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i
