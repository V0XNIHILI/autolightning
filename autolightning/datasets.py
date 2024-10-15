from autolightning import AutoDataModule


class MNIST(AutoDataModule):
    def __init__(self, root: str, download: bool = False, val_percentage: float = 0.1, **kwargs):
        super().__init__(
            dataset={
                "class": "torchvision.datasets.MNIST",
                "args": dict(defaults=dict(root=root,
                                           train=True,
                                           download=download),
                             test=dict(train=False)),
            },
            random_split=dict(source="defaults", dest=dict(train=1-val_percentage, val=val_percentage)),
            **kwargs
        )
