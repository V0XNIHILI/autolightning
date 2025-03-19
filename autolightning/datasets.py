from . import AutoDataModule
from .types import AutoDataModuleKwargsNoDatasetPrepareSplit, Unpack


class RootDownloadTrain(AutoDataModule):
    def __init__(self, name: str, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit]):
        super().__init__(
            dataset={
                "class": f"torchvision.datasets.{name}" if "." not in name else name,
                "args": dict(defaults=dict(root=root,
                                    train=True,
                                    download=download),
                        test=dict(train=False)),
            },
            random_split={
                "source": "defaults",
                "dest": dict(train=1-val_percentage,
                             val=val_percentage),
            },
            requires_prepare=download,
            **kwargs)


class MNIST(RootDownloadTrain):
    def __init__(self, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit]):
        super().__init__(name="MNIST", root=root, download=download, val_percentage=val_percentage, **kwargs)


class CIFAR10(RootDownloadTrain):
    def __init__(self, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit]):
        super().__init__(name="CIFAR10", root=root, download=download, val_percentage=val_percentage, **kwargs)


class FashionMNIST(RootDownloadTrain):
    def __init__(self, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatasetPrepareSplit]):
        super().__init__(name="FashionMNIST", root=root, download=download, val_percentage=val_percentage, **kwargs)
