# autolightning

## Main idea

My goal here is to achieve zero-code, pure from-configuration-only training of PyTorch models using PyTorch Lightning. This is achieved by using a configuration dictionary that specifies the model, the dataset, the data loaders, etc. The configuration is then used to build all required objects. Currently, this leads to an average lines-of-code reduction of 15% compared to a standard PyTorch Lightning, while improving customizability + reproducibility and maintaining the same flexibility as the original code.

## Example usage

### Define the configuration

To define a complete configuration, you can use the following top-level keys:

```python
cfg = {
    "learner": {...},
    "criterion": {...},
    "lr_scheduler": {...}, # Optional
    "model": {...},
    "optimizer": {...},
    "training": {...}, # Optional
    "seed": ..., # Optional
    "dataset": {...}, # Optional
    "dataloaders": {...}, # Optional
}

```

For example, to train a LeNet5 on MNIST with early stopping and learning rate stepping, the configuration can be defined like below (note that I use `DotMap` here to define the configuration, but you can use any other dictionary-like object):

```python
from dotmap import DotMap

cfg = DotMap()

# Specify the learner and its configuration
# Without having any dots in the name, the import will be done from
# the `torch_mate.lightning.lm` module
cfg.learner.name = "SupervisedLearner"
cfg.learner.cfg = DotMap({
    # Indicate whether classification accuracy should be computed
    "classification": True,
    # Optionally specify for which ks the top-k accuracy should be computed
    # "topk": [1, 5],
})

# Select the criterion and its configuration
cfg.criterion.name = 'CrossEntropyLoss'
# Optionally specify the configuration for the criterion:
# cfg.criterion.cfg = DotMap({
#     'reduction': 'mean'
# })

# Optionally specify the learning rate scheduler and its configuration
cfg.lr_scheduler.scheduler = DotMap({
    "name": "StepLR",
    "cfg": DotMap({
        "step_size": 2,
        "verbose": True,
    }),
})

# Specify the model and its configuration
cfg.model.name = 'torch_mate.models.LeNet5BNMaxPool'
cfg.model.cfg = DotMap({
    'num_classes': 10
})
# Optionally specify a compilation configuration
cfg.model.extra.compile = DotMap({
    'name': 'torch.compile'
})

# Specify the optimizer and its configuration
cfg.optimizer.name = 'Adam'
cfg.optimizer.cfg = DotMap({"lr": 0.007})

# Specify the training configuration (passed directly to the PyTorch
# Lightning Trainer). The `early_stopping` configuration is optional
# and will be used to configure the early stopping callback.
cfg.training = DotMap({
    'max_epochs': 100,
    'early_stopping': DotMap({
        'monitor': 'val/loss',
        'patience': 10,
        'mode': 'min'
    }),
})

# Set the seed for reproducibility
cfg.seed = 4223747124

# Specify the dataset and its configuration.
# Without having any dots in the name, the import will be done from
# the `torch_mate.lightning.datasets` module
cfg.dataset.name = 'MagicData'
cfg.dataset.cfg = DotMap({
    "name": "MNIST", # Can also be torchvision.datasets.MNIST for example
    "val_percentage": 0.1
})
cfg.dataset.kwargs = DotMap({
    "root": './data',
    "download": True
})

# Specify the transforms and their configuration
# Note that you can specify .pre (common pre-transform), .train
# .val/.test/.predict (specific transforms for each split) and
# .post (common post-transform). The complete transforms will
# then be built automatically. The same goes for target_transforms
# via: cfg.dataset.target_transforms
cfg.dataset.transforms.pre = [
    DotMap({'name': 'ToTensor'}),
    DotMap({'name': 'Resize', 'cfg': {'size': (28, 28)}}),
]

# Optionally, specify a pre-device and post-device transfer
# batch transform via: cfg.dataset.batch_transforms.pre and
# cfg.dataset.batch_transforms.post in the same manner
# as for the other transforms.

# Specify the data loaders and their configuration (where default
# is the fallback configuration for all data loaders)
cfg.dataloaders = DotMap({
    'default': DotMap({
        'num_workers': 4,
        'prefetch_factor': 16,
        'persistent_workers': True,
        'batch_size': 256,
    }),
    'train': DotMap({
        'batch_size': 512
    })
})

cfg = cfg.toDict()
```

The complete configuration dictionary will then look like this:

```python
{'learner': {'name': 'SupervisedLearner', 'cfg': {'classification': True}},
 'criterion': {'name': 'CrossEntropyLoss'},
 'lr_scheduler': {'scheduler': {'name': 'StepLR',
   'cfg': {'step_size': 2, 'verbose': True}}},
 'model': {'name': 'torch_mate.models.LeNet5BNMaxPool',
  'cfg': {'num_classes': 10}},
 'optimizer': {'name': 'Adam', 'cfg': {'lr': 0.007}},
 'training': {'max_epochs': 100,
  'early_stopping': {'monitor': 'val/loss', 'patience': 10, 'mode': 'min'}},
 'seed': 4223747124,
 'dataset': {'name': 'MNISTData',
  'kwargs': {'root': './data'},
  'transforms': {'pre': [{'name': 'ToTensor'},
    {'name': 'Resize', 'cfg': {'size': (28, 28)}}]}},
 'dataloaders': {'default': {'num_workers': 4,
   'prefetch_factor': 16,
   'persistent_workers': True,
   'batch_size': 256},
  'train': {'batch_size': 512}}}
```

Note that the configuration can also contain references to classes directly, without the relative import path. This is practical for example when you define a model class in the same file as the configuration. For example:

```python
class LeNet5BNMaxPool(nn.Module):
    def __init__(self, num_classes: int):
        super(LeNet5BNMaxPool, self).__init__()
        ...

    def forward(self, x):
        ...


cfg["model"]["name"] = LeNet5BNMaxPool
```

### Get the model, data and trainer

```python
from lightning.pytorch.loggers import WandbLogger

from autolightning import config_all

trainer, model, data = config_all(cfg,
    # Specify all keyworded arguments that are not part of the 
    # `cfg.training` dictionary for the PyTorch Lightning Trainer
    {
        "enable_progress_bar": True,
        "accelerator": "mps",
        "devices": 1,
        "logger": WandbLogger(project="test_wandb_lightning")
    }
)
```

### Train the model

```python
trainer.fit(model, data)
```

## Customization

### For models

#### Hooks overview

In case you want to add or override behavior of the defaults selected by TorchMate, this can be done by using hooks. TorchMate adds a few new hooks, next to the ones provided by PyTorch Lightning:

- `configure_configuration(self, cfg: Dict)`
    - Return the configuration that should be used. This configuration can be accessed at `self.hparams`.
- `config_model(self)`
    - Return the model that should be trained. This model can be access with `get_model(self)`.
- `compile_model(self, model: nn.Module)`
    - Compile the model and return it. This is called after the model is built and can be used to add change the compile behavior.
- `configure_criteria(self)`
    - Return the criteria that should be used.
- `configure_optimizers_only(self)`
    - Return the optimizers that should be used.
- `configure_schedulers(self, optimizers: List[optim.Optimizer])`
    - Return the schedulers that should be used.
- `shared_step(self, batch, batch_idx, phase: str)` 
    - Function that is called by `training_step(...)`, `validation_step(...)`,  `test_step(...)` and `predict_step(...)` from the`AutoModule` with the fitting stage argument (`train`/`val`/`test`/`predict`)

#### Example hook usage

```python
import torch.nn as nn

from autolightning import AutoModule

class MyModel(AutoModule):
    def config_model(self):
        # Can put any logic here and can access the configuration
        # via self.hparams
        return nn.Linear(100, 10)

    def configure_criteria(self):
        return nn.MSELoss()

    def shared_step(self, batch, batch_idx, phase: str):
        X, y = batch
        model = self.get_model()
        criterion = self.criteria

        loss = criterion(model(X), y)

        self.log(f"{phase}/loss", loss)

        return loss
```

### For data

#### Hooks overview

Similar to models, you can customize the data loading behavior by using hooks. TorchMate adds the following new hooks:

- `configure_configuration(self, cfg: Dict)`
- `get_common_transform(self, moment: str)`
- `get_common_target_transform(self, moment: str)`
- `get_common_batch_transform(self, moment: str)`
- `get_transform(self, stage: str)`
- `get_target_transform(self, stage: str)`
- `get_batch_transform(self, moment: str)`
- `get_dataloader_kwargs(self, stage: str)`
- `get_dataset(self, phase: str)`
- `get_transformed_dataset(self, phase: str)`
- `get_dataloader(self, phase: str)`

#### Example hook usage

```python
import torch.nn as nn

from autolightning import AutoDataModule

class MyDataModule(AutoDataModule):
    def get_dataset(self, split: str):
        # Can put any logic here and can access the configuration
        # via self.hparams
        return MyDataset(split)
```