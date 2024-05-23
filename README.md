# autolightning

The goal of this project is to achieve zero-code, from-configuration-only training of PyTorch models using PyTorch Lightning. This is achieved by using a configuration dictionary that specifies the model, the dataset, the data loaders, etc. The configuration is then used to build all required objects. Currently, this leads to an average lines-of-code reduction of 15% compared to a standard PyTorch Lightning, while improving customizability + reproducibility and maintaining the same flexibility as the original code.

## Example usage

### 1. Define the configuration

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

For example, to train a LeNet5 on MNIST with early stopping and learning rate stepping, the configuration can be defined in one of the following ways:

#### Option 1: A (regular) dictionary

Note that I use `DotMap` here to define the configuration, but you can use any other dictionary-like object, or use tools like OmegaConf or Hydra to define the configuration.

```python
from dotmap import DotMap

cfg = DotMap()

# Specify the learner and its configuration
# Without having any dots in the name, the import will be done from
# the `autolightning.lm` module
cfg.learner.name = "SupervisedLearner"
cfg.learner.cfg = {
    # Indicate whether classification accuracy should be computed
    "classification": True,
    # Optionally specify for which ks the top-k accuracy should be computed
    # "topk": [1, 5],
}

# Select the criterion and its configuration
cfg.criterion.name = 'CrossEntropyLoss'
# Optionally specify the configuration for the criterion:
# cfg.criterion.cfg = {
#     'reduction': 'mean'
# }

# Optionally specify the learning rate scheduler and its configuration
cfg.lr_scheduler.scheduler = {
    "name": "StepLR",
    "cfg": {
        "step_size": 2,
        "verbose": True,
    },
}

# Specify the model and its configuration
cfg.model.name = 'torch_mate.models.LeNet5BNMaxPool'
cfg.model.cfg.num_classes = 10

# Optionally specify a compilation configuration
cfg.model.extra.compile = {
    'name': 'torch.compile'
}

# Specify the optimizer and its configuration
cfg.optimizer.name = 'Adam'
cfg.optimizer.cfg = {"lr": 0.007}

# Specify the training configuration (passed directly to the PyTorch
# Lightning Trainer). The `early_stopping` configuration is optional
# and will be used to configure the early stopping callback.
cfg.training = {
    'max_epochs': 100,
    'early_stopping': {
        'monitor': 'val/loss',
        'patience': 10,
        'mode': 'min'
    },
}

# Set the seed for reproducibility
cfg.seed = 4223747124

# Specify the dataset and its configuration.
# Without having any dots in the name, the import will be done from
# the `autolightning.datasets` module
cfg.dataset.name = 'MagicData'
cfg.dataset.cfg = {
    "name": "MNIST", # Can also be torchvision.datasets.MNIST for example
    "val_percentage": 0.1
}
cfg.dataset.kwargs = {
    "root": './data',
    "download": True
}

# Specify the transforms and their configuration
# Note that you can specify .pre (common pre-transform), .train
# .val/.test/.predict (specific transforms for each split) and
# .post (common post-transform). The complete transforms will
# then be built automatically. The same goes for target_transforms
# via: cfg.dataset.target_transforms
cfg.dataset.transforms.pre = [
    {'name': 'ToTensor'},
    {'name': 'Resize', 'cfg': {'size': (28, 28)}},
]

# Optionally, specify a pre-device and post-device transfer
# batch transform via: cfg.dataset.batch_transforms.pre and
# cfg.dataset.batch_transforms.post in the same manner
# as for the other transforms.

# Specify the data loaders and their configuration (where default
# is the fallback configuration for all data loaders)
cfg.dataloaders = {
    'default': {
        'num_workers': 4,
        'prefetch_factor': 16,
        'persistent_workers': True,
        'batch_size': 256,
    },
    'train': {
        'batch_size': 512
    }
}
```

The complete configuration dictionary will then look like this:

```python
>>> cfg.toDict()
{'learner': {'name': 'SupervisedLearner', 'cfg': {'classification': True}}, 'criterion': {'name': 'CrossEntropyLoss'}, 'lr_scheduler': {'scheduler': {'name': 'StepLR', 'cfg': {'step_size': 2, 'verbose': True}}}, 'model': {'name': 'torch_mate.models.LeNet5BNMaxPool', 'cfg': {'num_classes': 10}, 'extra': {'compile': {'name': 'torch.compile'}}}, 'optimizer': {'name': 'Adam', 'cfg': {'lr': 0.007}}, 'training': {'max_epochs': 100, 'early_stopping': {'monitor': 'val/loss', 'patience': 10, 'mode': 'min'}}, 'seed': 4223747124, 'dataset': {'name': 'MagicData', 'cfg': {'name': 'MNIST', 'val_percentage': 0.1}, 'kwargs': {'root': './data', 'download': True}, 'transforms': {'pre': [{'name': 'ToTensor'}, {'name': 'Resize', 'cfg': {'size': (28, 28)}}]}}, 'dataloaders': {'default': {'num_workers': 4, 'prefetch_factor': 16, 'persistent_workers': True, 'batch_size': 256}, 'train': {'batch_size': 512}}}
```

Note that the configuration can also contain references to classes directly, without the relative import path. This is practical for example when you define a model class in the same file as the configuration. For example:

```python
class LeNet5BNMaxPool(nn.Module):
    def __init__(self, num_classes: int):
        super(LeNet5BNMaxPool, self).__init__()
        ...

    def forward(self, x):
        ...


cfg.model.name = LeNet5BNMaxPool
```

Finally, serialize the resulting configuration to a dictionary:

```python
cfg = cfg.toDict()
```

#### Option 2: a YAML file

#### Option 3: OmegaConf

#### Option 4: Hydra

### 2. Get the model, data and trainer

#### Option 1: Let `autolightning` do the work

```python
from lightning.pytorch.loggers import WandbLogger

from autolightning import config_all

# Next to config_all, you can also use config_model, config_data and config_model_data to only configure specific parts
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

#### Option 2: Create the objects separately based on fixed classes

```python
from autolightning.lm import SupervisedLearner
from autolightning.datasets import MagicData

from lightning import Trainer

# Create the model, data and trainer
model = SupervisedLearner(cfg)
data = MagicData(cfg)
trainer = Trainer(**cfg.training)
```

#### Option 3: Use the AutoLightning CLI (based on PyTorch Lightning CLI)

```python
# TO DO!
```

#### Option 3: Use the original PyTorch Lightning CLI based on fixed classes

This way, you can reuse the features of the Lightning CLI and use it to store the main configuration separate (`cfg` in this case) from the trainer configuration (can be provided via the `--config` flag), which often contains environment-specific settings like GPU indices, etc. To see how this can be done, see [here](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html) for a crisp overview of the PyTorch Lightning CLI.

```python
# file: main.py

from autolightning import pre_cli

from autolightning.lm import SupervisedLearner
from autolightning.datasets import MagicData

from lightning.pytorch.cli import LightningCLI

def cli_main():
    cfg = ... # Load or set the configuration

    LightningCLI(
        pre_cli(SupervisedLearner, cfg), # All arguments after cfg are available to be set in the CLI
        pre_cli(MagicData, cfg), # Same goes for the data module
        trainer_defaults=cfg["training"] if "training" in cfg else None,
        seed_everything_default=cfg["seed"] if "seed" in cfg else True,
    )

if __name__ == "__main__":
    cli_main()
```

### 3. Train the model

If you have the model, data and trainer, you can train the model using the following code:

```python
trainer.fit(model, data)
```

In case you used the CLI, you can run the following command:

```bash
python main.py fit --config ./config.yaml
```

Where `config.yaml` for example looks like this:

```yaml
trainer:
  logger:
    - class_path: WandbLogger
      init_args:
        project: test_autolightning
  callbacks:
    - class_path: ModelCheckpoint
      init_args:
        dirpath: ./nets
        monitor: val/accuracy
        save_top_k: 1
  accelerator: gpu
  check_val_every_n_epoch: 1
  log_every_n_steps: 20
```

## Customization

### For models

#### Hooks overview

In case you want to add or override behavior of the defaults selected by autolightning, this can be done by using hooks. autolightning adds a few new hooks, next to the ones provided by PyTorch Lightning:

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

Similar to models, you can customize the data loading behavior by using hooks. autolightning adds the following new hooks:

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