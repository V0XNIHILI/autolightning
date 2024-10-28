from . import AutoDataModule
from .types import AutoDataModuleKwargsNoDatsetPrepareSplit, Unpack


def _get_auto_data_module_kwargs(dataset: str, root: str, download: bool, val_percentage: float):
    kwargs = {}

    kwargs["dataset"] = {
        "class": f"torchvision.datasets.{dataset}" if "." not in dataset else dataset,
        "args": dict(defaults=dict(root=root,
                                    train=True,
                                    download=download),
                        test=dict(train=False)),
    }

    kwargs["random_split"] = {
        "source": "defaults",
        "dest": dict(train=1-val_percentage,
                     val=val_percentage),
    }

    kwargs["requires_prepare"] = download

    return kwargs


class RootDownloadTrain(AutoDataModule):
    def __init__(self, name: str, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatsetPrepareSplit]):
        super().__init__(**_get_auto_data_module_kwargs(name, root, download, val_percentage), **kwargs)


class MNIST(RootDownloadTrain):
    def __init__(self, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatsetPrepareSplit]):
        super().__init__(name="MNIST", root=root, download=download, val_percentage=val_percentage, **kwargs)


class CIFAR10(RootDownloadTrain):
    def __init__(self, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatsetPrepareSplit]):
        super().__init__(name="CIFAR10", root=root, download=download, val_percentage=val_percentage, **kwargs)


class FashionMNIST(RootDownloadTrain):
    def __init__(self, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs: Unpack[AutoDataModuleKwargsNoDatsetPrepareSplit]):
        super().__init__(name="FashionMNIST", root=root, download=download, val_percentage=val_percentage, **kwargs)
