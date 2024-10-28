# autolightning

**The authors of this project would like to thank the [Lightning](https://lightning.ai/) team for their amazing work on [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), which this project is extensively based on.**

The goal of this project is to achieve zero-code, from-configuration-only training of PyTorch models using PyTorch Lightning. This is achieved by using a configuration dictionary that specifies the model, the dataset, the data loaders, etc. The configuration is then used to build all required objects. Currently, this leads to an average lines-of-code reduction of 15% compared to a standard PyTorch Lightning, while improving customizability + reproducibility and maintaining the same flexibility as the original code.

## Installation

To install the package, you can use the following command:

```bash
pip install git+https://github.com/V0XNIHILI/autolightning.git@main
```

Or, if you want to install the package in editable mode, you can use the following command:

```bash
git clone git@github.com:V0XNIHILI/autolightning.git
cd autolightning
# Make sure you have pip 23 or higher
pip install -e .
```

## Example usage

### In a notebook or script

#### Supervised learning

To run supervised learning on MNIST, with a simple FC layer, the following code is all you need:

```python
from autolightning.lm import Classifier
from autolightning.datasets import MNIST

from functools import partial

import torch
import torch.nn as nn

from torchvision import transforms

net = nn.Linear(28*28, 7)

model = Classifier(
    net=net, 
    optimizer=torch.optim.Adam(net.parameters(), lr=0.003)
)

data = MNIST(
    root="data",
    dataloaders=dict(batch_size=128)
    transforms=[transforms.ToTensor(), nn.Flatten(start_dim=0)]
)
```

Then train using the same PyTorch Lightning trainer that you were used too:

```python
from lightning.pytorch import Trainer

trainer = Trainer(
    max_epochs=10
)

trainer.fit(model, data)
```

### From the CLI (based on PyTorch Lightning CLI)

Note that the `autolightning` CLI tool supports all the same arguments as the regular PyTorch Lightning CLI (as the `AutoCLI` is a subclass of the `LightningCLI`) but allows for two key differences:

1. Configurations can also be specified as Python files (instead of in YAML files)
2. The AutoCLI has additional `torch` flags that can be set in a configuration file to configure the PyTorch backend regarding debugging and performance. For example:
    ```yaml
    ...
    torch:
    autograd:
        set_detect_anomaly: False
        profiler:
        profile: False
        emit_nvtx: False
    set_float32_matmul_precision: high
    backends:
        cuda:
        matmul:
            allow_tf32: True
        cudnn:
        allow_tf32: True
        benchmark: True
    ```

You can also split hyperparameter configuration from per-machine specific configuration by moving the latter into a separate (YAML) config file, for example:

```yaml
# filename: local.yaml
# --------------------

data:
  # Don't forget to remove these keys from main.py!
  root: "../datasets/data"
  download: True
trainer:
  logger:
    - class_path: WandbLogger
      init_args:
        project: name_of_wandb_project
        log_model: true
  callbacks:
    - class_path: ModelCheckpoint
      init_args:
        monitor: val/accuracy
        mode: max
        save_on_train_epoch_end: false
        save_top_k: 1
  accelerator: gpu
  check_val_every_n_epoch: 10
  devices: [3]
```

To run training on the combined configuration (where only the values in `main.py` are stored as hyperparemeters):

```bash
autolightning fit -c main.py  -c local.yaml
```

#### Option 4: Use the original PyTorch Lightning CLI
<details>
    <summary>Option 4.2 (not recommended): Enable model, data and trainer configuration from CLI</summary>

In this way, you store all the training, model and data configuration in one file. However, to stay consistent with the original Lightning CLI API, we use variable interpolation to avoid duplicate values in the YAML file (to enable this, we set `parser_kwargs={"parser_mode": "omegaconf"}`).

```python
# file: main.py

from autolightning import ConfigurableLightningModule, ConfigurableLightningDataModule

from lightning.pytorch.cli import LightningCLI

def cli_main():
    LightningCLI(
        ConfigurableLightningModule,
        ConfigurableLightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"}
    )

if __name__ == "__main__":
    cli_main()
```

```yaml
# file: config.yaml

trainer:
  max_epochs: ${model.init_args.cfg.training.max_epochs}
  ...
model:
  class_path: ${model.init_args.cfg.learner.name}
  init_args:
    # All variables in the this cfg variable below will be saved as hyperparameters,
    # and can be accessed in the model via self.hparams. None of the other variables
    # in this file will be saved as hyperparameters.
    cfg:
      criterion:
      dataloaders:
      dataset:
        name: your_module.YourDataModule
      learner:
        name: autolightning.lm.SupervisedLearner
      lr_scheduler:
      model:
      optimizer:
      seed: 4223747124
      training:
        max_epochs: 100
    ...
data:
  class_path: ${model.init_args.cfg.dataset.name}
  init_args:
    cfg: ${model.init_args.cfg}
    ...
seed_everything: ${model.init_args.cfg.seed}
```
</details>

### 3. Train the model

If you have the model, data and trainer instantiated, you can train the model using the following code:

```python
trainer.fit(model, data)
```

Or if you want to use the AutoLightning CLI, you can run the following command:

```bash
autolighting fit --config main.py --config local.yaml
```

Finally, in case you used the original PyTorch Lightning CLI in your own code, you can run the following command:

```bash
python main.py fit --config ./config.yaml
```

## Customization

### For models

#### Hooks overview

In case you want to add or override behavior of the defaults selected by autolightning, this can be done by using hooks. autolightning adds a few new hooks, next to the ones provided by PyTorch Lightning:

- `parameters_for_optimizer(self, recurse: bool = True)`
    - Return the parameters that should be used for the optimizer.
- `register_optimizer(self, module: nn.Module, optimizer: Optional[OptimizerCallable] = None, lr_scheduler: Optional[LRSchedulerCallable] = None)`
    - Register the optimizer and learning rate scheduler that should be used.
- `register_metric(self, name: str, metric: MetricType)`
    - Register a metric that should be used.
- `register_metrics(self, metrics: Dict[str, MetricType])`
    - Register multiple metrics that should be used.
- `enable_prog_bar(self, phase: Phase)`
    - Return whether the progress bar should be enabled.
- `shared_step(self, phase: Phase, *args, **kwargs)`
    - Function that is called by `shared_logged_step(...)` from the `AutoModule` with the fitting phase argument (`train`/`val`/`test`/`predict`)
- `shared_logged_step(self, phase: Phase, *args: Any, **kwargs: Any):` 
    - Function that is called by `training_step(...)`, `validation_step(...)`,  `test_step(...)` and `predict_step(...)` from the`AutoModule` with the fitting phase argument (`train`/`val`/`test`/`predict`)

#### Example hook usage

```python
import torch.nn as nn

from autolightning import AutoModule

class MyModel(AutoModule):
    def shared_step(self, batch, batch_idx, phase: str):
        X, y = batch

        loss = self.criterion(self.net(X), y)

        return loss
```

### For data

#### Hooks overview

Similar to models, you can customize the data loading behavior by using hooks. autolightning adds the following new hooks:

- `get_transform(self, stage: str)`
    - Return the transform that should be used for the given stage.
- `get_target_transform(self, stage: str)`
    - Return the target transform that should be used for the given stage.
- `get_dataset(self, phase: Phase)`
    - Return the dataset that should be used for the given phase.
- `get_transformed_dataset(self, phase: Phase)`
    - Return the dataset that should be used for the given phase, after applying the transforms.
- `get_dataloader(self, phase: Phase)`
    - Return the dataloader that should be used for the given phase.

#### Example hook usage

```python
import torch.nn as nn

from autolightning import AutoDataModule

class MyDataModule(AutoDataModule):
    def get_dataset(self, phase):
        # Can put any logic here and can access the configuration
        # via self.hparams
        return MyDataset(train=phase == 'train')
```