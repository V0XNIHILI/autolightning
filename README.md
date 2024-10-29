# autolightning ⚡️

**The authors of this project would like to thank the [Lightning](https://lightning.ai/) team for their amazing work on [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), which this project is fully based on.**

The goal of this project is to achieve zero-code, from-configuration-only training of PyTorch models using PyTorch Lightning. This is achieved by using a configuration dictionary that specifies the model, the dataset, the data loaders, etc. The configuration is then used to build all required objects. Currently, this leads to an average lines-of-code reduction of 15% compared to a standard PyTorch Lightning, while improving customizability + reproducibility and maintaining the same flexibility as the original code.

## Built-in training methods

### Supervised learning

* Regular supervised learning ([`Supervised`](./autolightning/lm/supervised.py))
* Classification ([`Classifier`](./autolightning/lm/classifier.py))

### Self-supervised learning

* Siamese networks ([`Siamese`](./autolightning/lm/self_supervised.py))
* Triplet networks ([`Triplet`](./autolightning/lm/self_supervised.py))

### Knowledge distillation

* Knowledge distillation supporting an optional student head and a student regressor net ([`Distilled`](./autolightning/lm/distillation.py)), useful distillation losses provided in [`distillation_losses.py`](autolightning/nn/distillation_losses.py)

### Quantization-aware training (QAT)

* Performing QAT with [Brevitas](https://github.com/Xilinx/brevitas) ([`BrevitasSupervised`, `BrevitasClassifier`, `BrevitasPrototypical`](./autolightning/lm/qat.py))

### Few-shot learning

* Prototypical learning ([`Prototypical`](./autolightning/lm/prototypical.py))

### Continual learning

Works in progress!

## Built-in datasets

* Any dataset that has the call signature `(root: str, download: bool, train: bool)` using [`RootDownloadTrain`](./autolightning/datasets.py), a few of these are included by default:
  * [`MNIST`](./autolightning/datasets.py)
  * [`CIFAR10`](./autolightning/datasets.py)
  * [`FashionMNIST`](./autolightning/datasets.py)
* Few-shot dataset using [`FewShotMixin`](./autolightning/dm/few_shot.py)

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
2. The AutoCLI has additional `torch` flags that can be set in a configuration file to configure the PyTorch backend regarding debugging and performance.
3. By default, the CLI only allows subclasses of `AutoModule` and `AutoDataModule` to be used as Lightning modules.

To train an MLP on MNIST, you can write a YAML configuration file like the following:

```yaml
# filename: config.yaml
# --------------------

model:
  class_path: autolightning.lm.Classifier
  init_args:
    net:
        class_path: torchvision.ops.MLP
        init_args:
            in_channels: 784
            hidden_channels: [100, 10]
data:
  class_path: autolightning.datasets.MNIST
  init_args:
    root: data/
    download: true
    transforms:
      post:
        - class_path: torchvision.transforms.ToTensor
        - class_path: torchvision.transforms.Normalize
          init_args:
            mean: [0.1307]
            std: [0.3081]
        - class_path: torch.nn.Flatten
          init_args:
            start_dim: 0
    dataloaders:
      defaults:
        batch_size: 1024
      train:
        shuffle: true
      val:
        drop_last: false
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 1e-3
seed_everything: 437952406
trainer:
  check_val_every_n_epoch: 10
  max_epochs: 100
  accelerator: cpu
  logger:
    - class_path: autolightning.loggers.AutoWandbLogger
      init_args:
        project: name_of_my_project
```

Then, to add the additional torch flags, you can append the following to the configuration file:

```yaml
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

To run training on the configuration file, you can use the following command:

```bash
autolightning fit -c config.yaml
```

#### Extra options

Alternatively, you can also load a pre-trained model from a state dict, compile a model or disable gradients for a module:

##### Loading a pre-trained model

```yaml
model:
  class_path: autolightning.lm.Classifier
  init_args:
    net:
        class_path: autolightning.load
        init_args:
          module:
            class_path: torchvision.ops.MLP
            init_args:
                in_channels: 784
                hidden_channels: [100, 10]
          file_path: path/to/state_dict.pth
...
```

##### Compiling a model

```yaml
model:
  class_path: autolightning.lm.Classifier
  init_args:
    net:
        class_path: autolightning.compile
        init_args:
          module:
            class_path: torchvision.ops.MLP
            init_args:
                in_channels: 784
                hidden_channels: [100, 10]
          compiler_path: torch.compile
...
```

##### Disabling gradients

```yaml
model:
  class_path: autolightning.lm.Classifier
  init_args:
    net:
        class_path: autolightning.disable_grad
        init_args:
          module:
            class_path: torchvision.ops.MLP
            init_args:
                in_channels: 784
                hidden_channels: [100, 10]
...
```

#### Best practice

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

To run training on the combined configuration (where the values in `main.py` are hyperparameters and the values in `local.yaml` are machine-specific):

```bash
autolightning fit -c main.py  -c local.yaml
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

---

© (2024) Douwe den Blanken, Delft, the Netherlands
