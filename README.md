# autolightning ⚡️

**The authors of this project would like to thank the [Lightning](https://lightning.ai/) team for their amazing work on [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), which this project is fully based on.**

## Overview

`autolightning` provides a zero-code, configuration-driven approach to training PyTorch models using PyTorch Lightning. By specifying models, datasets, transforms, optimizers, and more through configuration dictionaries, you can significantly reduce boilerplate code while maintaining flexibility, improving reproducibility, and enhancing customization options.

Current benchmarks show an average **15% reduction in lines of code** compared to standard PyTorch Lightning implementations, without compromising functionality.

## Key Features

- [**Custom CLI**](#custom-cli): `autolightning`'s CLI application avoids having to create separate `LightningCLI`s for each new project
- [**Config-driven model training**](#config-driven-model-training): Using the CLI, train models with minimal code using YAML or Python config files
- [**Comprehensive transform pipelines**](#transform-pipeline): Easily define complex transform pipelines for datasets
- [**Flexible optimizer and scheduler configuration**](#optimizer-and-scheduler-configuration): Define optimizers and schedulers in the configuration file
- [**Additional (torch) runtime flags**](#additional-runtime-flags): Enable PyTorch performance optimizations or model watching with Weights & Biases from the command line
- [**Built-in dataset splitting**](#built-in-dataset-splitting): Random split and cross-validation support from the command line / with a configuration file
- [**Hyperparameter optimization & model watching**](#hyperparameter-sweeps): Use Weights & Biases, Ray Tune, or Optuna for hyperparameter sweeps. Track model gradient results with Weights & Biases from the command line
- [**Standardized training methods**](#standardized-training-methods): Pre-made modules for supervised learning, self-supervised learning, knowledge distillation, and more
- [**Config-file utilities**](#config-file-utilities): Load pre-trained models, compile models, or freeze model parameters from a configuration file

## Installation

Standard installation:
```bash
pip install git+https://github.com/V0XNIHILI/autolightning.git@main
```

Development installation:
```bash
git clone git@github.com:V0XNIHILI/autolightning.git
cd autolightning
# Make sure you have pip 23 or higher
pip install -e .
```

## Quick Start: Supervised Learning on MNIST

### Using Python Code

```python
from autolightning.lm import Classifier
from autolightning.datasets import MNIST

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.ops import MLP

# Define model
net = MLP(784, [100, 10])

# Create classifier with optimizer
model = Classifier(
    net=net, 
    optimizer=torch.optim.Adam(net.parameters(), lr=1e-3)
)

# Create data module with transforms
data = MNIST(
    root="data",
    dataloaders=dict(batch_size=128),
    transforms=[transforms.ToTensor(), nn.Flatten(start_dim=0)]
)

# Train with standard Lightning trainer
from lightning.pytorch import Trainer

trainer = Trainer(max_epochs=10)
trainer.fit(model, data)
```

### Using Configuration (YAML)

Create a `config.yaml`:

```yaml
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
        - class_path: torch.nn.Flatten
          init_args:
            start_dim: 0
    dataloaders:
      batch_size: 128
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 1e-3
seed_everything: 437952406
trainer:
  max_epochs: 10
```

Then, run training with the following command:

```bash
autolightning fit -c config.yaml
```

All hyperparameters will be automatically logged to your selected logger.

## Key Features in Detail

### Custom CLI

The `autolightning` command line interface is based is a derived class from PyTorch Lightning's `LightningCLI` and supports all of its features while adding additional functionality. It supports, among others:

- Hyperparameter sweeps
- Python configuration files
- Additional torch
- Support for WandB model watching
- Support for WandB sweeps

### Config-driven model training

#### Automated Logging of Hyperparameters

All hyperparameters are automatically logged to the logger of your choice (e.g. Weights & Biases, TensorBoard, etc.) when using the `autolightning` command line interface. This enables easy tracking of experiments and results.

#### Python Config Files

Supporting Python files for configuration files has the key advantages that it is posssible to have conditional logic or for-loops used in the configuration.

```bash
autolightning fit -c config.py
```

By default, the `AutoCLI` looks for a dictionary named `config` when you pass Python config script:

```python
config = {
  "model": ...,
  "data": ...,
}
```

Alternatively, you can also specify a custom configuration name:

```python
CONFIG_NAME = "my_own_config"

my_own_config = {
  "model": ...,
  "data": ...,
}
```

Finally, it is also possible to use a config function:

```python
def config():
  return {
    "model": ...,
    "data": ...,
  }
```

### Transform Pipeline

`autolightning` provides a comprehensive transform pipeline that supports:

- **Dataset transforms**: Applied when loading the dataset
- **Pre-load transforms**: Applied before dataset is loaded into memory
- **Phase-specific transforms**: Different transforms for train/val/test splits
- **Batch transforms**: Applied during batch transfer to/from device

Example:

```python
data = AutoDataModule(
    dataset=CIFAR10("data", train=True, download=True),
    transforms={
        "pre_load": SimpleTransform(1),  # Applied before loading to memory
        "pre": [Transform1(), Transform2()],  # Applied before phase transform
        "train": TrainTransform(),  # Only applied to training data
        "val": ValTransform(),  # Only applied to validation data
        "post": PostTransform()  # Applied after phase transform
    },
    target_transforms={
        "pre_load": LabelTransform(),
        "train": LabelTransform(),
        "val": LabelTransform(),
        "post": LabelTransform()
    },
    batch_transforms={
        "before": BatchTransformBeforeGPU(),
        "after": BatchTransformAfterGPU()
    },
    pre_load=True  # Enable pre-loading
)
```

### Optimizer and Scheduler Configuration

By default when using `LightningCLI`, you can specify the optimizer and scheduler in the configuration file in the following way:

```yaml
data: ...
model: ...
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1e-4
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max: 100
    eta_min: 0.0
```

However, this is not very flexible for more advanced use cases. Hence, in `autolightning`, you can specify the optimizer and scheduler as arguments to the model class:

```yaml
data: ...
model:
  ...
  optimizer:
    class_path: autolightning.optim
    dict_kwargs:
      optimizer_path: AdamW
      lr: 1e-4
  lr_scheduler:
    class_path: autolightning.sched
    dict_kwargs:
      scheduler_path: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 100
      eta_min: 0.0
```

This allows for more flexibility in the configuration file. For example, to specify different optimizers for different parts of the model:

```yaml
model:
  net:
    class_path: nn.ModuleDict
    init_args:
      modules:
        encoder:
          class_path: torchvision.ops.MLP
          init_args:
            in_channels: 784
            hidden_channels: [100, 10]
        decoder:
          class_path: torchvision.ops.MLP
          init_args:
            in_channels: 10
            hidden_channels: [100, 784]
  optimizer:
    encoder:
      class_path: autolightning.optim
      dict_kwargs:
        optimizer_path: AdamW
        lr: 1e-4
    decoder:
      class_path: autolightning.optim
      dict_kwargs:
        optimizer_path: SGD
        lr: 1e-3
  lr_scheduler:
    class_path: autolightning.sched
    dict_kwargs:
      scheduler_path: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 100
      eta_min: 0.0
```

Or, with a list of modules instead of a dict:

```yaml
model:
  net:
    class_path: nn.ModuleList
    init_args:
      modules:
        - class_path: torchvision.ops.MLP
          init_args:
            in_channels: 784
            hidden_channels: [100, 10]
        - class_path: torchvision.ops.MLP
          init_args:
            in_channels: 10
            hidden_channels: [100, 784]
  optimizer:
    - class_path: autolightning.optim
      dict_kwargs:
        optimizer_path: AdamW
        lr: 1e-4
    - class_path: autolightning.optim
      dict_kwargs:
        optimizer_path: SGD
        lr: 1e-3
  lr_scheduler:
    class_path: autolightning.sched
    dict_kwargs:
      scheduler_path: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 100
      eta_min: 0.0
```

Note that it is also easy to [change the learning rate scheduler configuration](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers):

```yaml
model:
  ...
  optimizer: ...
  lr_scheduler:
    interval: step
    scheduler:
      class_path: autolightning.sched
      dict_kwargs:
        scheduler_path: torch.optim.lr_scheduler.CosineAnnealingLR
        T_max: 100
        eta_min: 0.0
```

### Additional runtime flags

To enable PyTorch performance optimizations or model watching with Weights & Biases, enable one or more of the following flags in your configuration:

```yaml
torch:
  autograd:
    set_detect_anomaly: false
    profiler:
      profile: false
      emit_nvtx: false
  set_float32_matmul_precision: highest
  backends:
    cuda:
      matmul:
        allow_tf32: true
    cudnn:
      allow_tf32: true
      benchmark: true
wandb:
  watch:
    enable: false  # Track gradients and parameters during training
```

### Built-in Dataset Splitting

#### Random Split

Split a dataset into train/val/test with specified ratios:

```yaml
data:
  class_path: autolightning.datasets.CIFAR10
  init_args:
    random_split:
      train: 0.7
      val: 0.2
      test: 0.1
    seed: 42  # For reproducibility
```

#### Cross-Validation

K-fold cross-validation can be easily performed on any dataset:

```python
from autolightning.lm import Classifier
from autolightning.datasets import MNIST

n_folds = 5

data = lambda i: MNIST(
    root="data",
    dataloaders=dict(batch_size=128),
    transforms=[transforms.ToTensor(), nn.Flatten(start_dim=0)],
    # Specify the cross validation setup
    cross_val=dict(n_folds=n_folds, fold_idx=i)
)

for fold_idx in range(n_folds):
    net = nn.Linear(28*28, 10)
    model = Classifier(
        net=net, 
        optimizer=torch.optim.Adam(net.parameters(), lr=0.003)
    )
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, data(fold_idx))
```

In a configuration file:

```yaml
data:
  class_path: autolightning.datasets.MNIST
  init_args:
    cross_val:
      n_folds: 5
      fold_idx: 0
```

Then, run training with the following command:

```bash
for idx in {0..4}; do
    autolightning fit -c config.yaml --data.init_args.cross_val.fold_idx $idx
done
```

### Hyperparameter Sweeps

#### Using Weights & Biases

```python
sweep_configuration = {
    'method': 'grid',
    'name': 'Example Sweep',
    'command': ['python', '${program}', '--config', yaml_conf, '${args_no_hyphens}'],
    'program': "autolightning",
    'parameters': {
        'model.init_args.learning_rate': {
            'values': [0.001, 0.01, 0.1]
        }
    },
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    }
}
```

Use the `AutoWandbLogger` in your configuration to track results.

#### Using Ray Tune or Optuna

Supported through configuration hooks:

```python
from ray import tune
from autolightning import auto_main

def tune_function(config):
    # Inject config parameters
    my_config = {...}  # Your base config
    my_config["model"]["init_args"]["learning_rate"] = config["learning_rate"]
    
    # Run training
    auto_main(config=my_config, subcommand="fit")

# Define search space
search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-1)
}

# Run hyperparameter search
analysis = tune.run(tune_function, config=search_space)
```

## Standardized Training Methods

### Supervised Learning
- [**`Supervised`**](./autolightning/lm/supervised.py): General supervised learning
- [**`Classifier`**](./autolightning/lm/classifier.py): Classification tasks

### Self-supervised Learning
- [**`Triplet`**](./autolightning/lm/self_supervised.py): Siamese networks for similarity learning
- [**`Triplet`**](./autolightning/lm/self_supervised.py): Triplet networks for ranking tasks

### Knowledge Distillation
- [**`Distilled`**](./autolightning/lm/distilled.py): Knowledge distillation with optional student head and regressor

### Quantization-Aware Training
- [**`BrevitasSupervised`, `BrevitasClassifier`, `BrevitasPrototypical`**](./autolightning/lm/brevitas.py): QAT with Brevitas

### Few-shot Learning
- [**`Prototypical`**](./autolightning/lm/prototypical.py): Prototypical networks for few-shot learning

### Coming Soon
- **Continual Learning**: Stay tuned!
- **In-context Learning**: Under development!

## Built-in Datasets

- [**`MNIST`**](./autolightning/datasets.py): Classic handwritten digit classification
- [**`CIFAR10`**](./autolightning/datasets.py): 10-class image classification
- [**`FashionMNIST`**](./autolightning/datasets.py): Fashion items classification
- [**`RootDownloadTrain`**](./autolightning/datasets.py): Wrapper for datasets with (root, download, train) parameters
- [**`FewShot` and `FewShotMixin`**](./autolightning/dm/few_shot.py): Convert any `AutoDataset` to a format suitable for few-shot learning (for example in combination with [**`Prototypical`**](./autolightning/lm/prototypical.py))

### Config-file utilities

#### Loading Pre-trained Models

```yaml
model:
  class_path: autolightning.lm.Classifier
  init_args:
    net:
      class_path: autolightning.load
      init_args:
        file_path: path/to/state_dict.pth
        module:
          class_path: torchvision.ops.MLP
          init_args:
              in_channels: 784
              hidden_channels: [100, 10]
```

#### Compiling Models

```yaml
model:
  class_path: autolightning.lm.Classifier
  init_args:
    net:
      class_path: autolightning.compile
      init_args:
        compiler_path: torch.compile
        module:
          class_path: torchvision.ops.MLP
          init_args:
              in_channels: 784
              hidden_channels: [100, 10]
```

#### Freezing Model Parameters

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
```

## Best Practices

### Separating Configuration

Split your configuration into hyperparameters and machine-specific settings:

**main.yaml** (hyperparameters):
```yaml
model:
  class_path: autolightning.lm.Classifier
  init_args:
    net:
      class_path: torchvision.ops.MLP
      init_args:
          in_channels: 784
          hidden_channels: [100, 10]
```

**local.yaml** (machine-specific):
```yaml
data:
  root: "../datasets/data"
  download: True
trainer:
  accelerator: gpu
  devices: [3]
```

Run with combined configuration:
```bash
autolightning fit -c main.yaml -c local.yaml
```

## Customization

### Model Hooks

autolightning adds several hooks to the standard Lightning model:

- **`parameters_for_optimizer(self, recurse: bool = True)`**: Control which parameters are used for optimization
- **`register_optimizer(self, module: nn.Module, optimizer, lr_scheduler)`**: Register optimizers and schedulers
- **`register_metric(self, name: str, metric)`**: Register evaluation metrics
- **`shared_step(self, phase: Phase, *args, **kwargs)`**: Core step function for all phases
- **`shared_logged_step(self, phase: Phase, *args: Any, **kwargs: Any)`**: Logging-aware step function

Example custom model:
```python
import torch.nn as nn
from autolightning import AutoModule

class MyModel(AutoModule):
    def shared_step(self, phase, batch, batch_idx):
        X, y = batch
        loss = self.criterion(self.net(X), y)
        return loss
```

### Data Module Hooks

Customize data loading with the following hooks:

- **`get_transform(self, stage: str)`**: Get the transform for a specific stage
- **`get_target_transform(self, stage: str)`**: Get the target transform for a specific stage
- **`get_dataset(self, phase: Phase)`**: Get the dataset for a specific phase
- **`get_transformed_dataset(self, phase: Phase)`**: Get the transformed dataset
- **`get_dataloader(self, phase: Phase)`**: Get the dataloader for a specific phase

Example custom data module:
```python
from autolightning import AutoDataModule

class MyDataModule(AutoDataModule):
    def get_dataset(self, phase):
        return MyDataset(train=phase == 'train')
```

## Advanced Usage

### Running with a Config File in a Script

```python
import yaml
from autolightning.main import auto_main

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

auto_main(
  # Load config from one or more files
  config=config, # Can be a list of dictionaries
  subcommand="fit"
)

```

### Intantiating the Model, Trainer and Data Module

```python
import yaml
from autolightning.main import auto_main

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load config from one or more files
trainer, model, datamodule = auto_main(
    config=config,
    run=False 
)
```

### Loading Just the Data Module

```python
from autolightning.main import auto_data

# Load and initialize just the data module
datamodule = auto_data(
    config={"class_path": "autolightning.datasets.MNIST", "init_args": {...}}
)
```

---

© (2025) Douwe den Blanken, Delft, the Netherlands
