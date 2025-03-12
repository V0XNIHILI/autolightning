# autolightning ⚡️

**The authors of this project would like to thank the [Lightning](https://lightning.ai/) team for their amazing work on [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), which this project is fully based on.**

## Overview

`autolightning` provides a zero-code, configuration-driven approach to training PyTorch models using PyTorch Lightning. By specifying models, datasets, transforms, optimizers, and more through configuration dictionaries, you can significantly reduce boilerplate code while maintaining flexibility, improving reproducibility, and enhancing customization options.

Current benchmarks show an average **15% reduction in lines of code** compared to standard PyTorch Lightning implementations, without compromising functionality.

## Key Features

- **Config-driven model training**: Train models with minimal code using YAML, Python dictionaries, or Python files
- **Comprehensive transform pipelines**: Full support for data transformations including batch transforms
- **Experiment management**: Seamless integration with WandB and other logging tools
- **Hyperparameter optimization**: Support for WandB sweeps, Ray Tune, and Optuna
- **Standardized training methods**: Pre-made modules for supervised learning, self-supervised learning, knowledge distillation, and more
- **Built-in dataset support**: Simplified access to common datasets with customizable transforms
- **K-fold cross-validation**: Easy implementation of cross-validation strategies
- **Custom CLI**: Enhanced autolightning command supporting all PyTorch Lightning features plus additional torch flags

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

Then, run training with:

```bash
autolightning fit -c config.yaml
```

All hyperparameters will be automatically logged to your selected logger.

## Advanced Features

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

### Cross-Validation

K-fold cross-validation can be easily performed:

```python
from autolightning.lm import Classifier
from autolightning.datasets import MNIST

n_splits = 5

data = lambda i: MNIST(
    root="data",
    dataloaders=dict(batch_size=128),
    transforms=[transforms.ToTensor(), nn.Flatten(start_dim=0)],
    # Specify the cross validation setup
    cross_val=dict(n_splits=n_splits, fold=i)
)

for fold_idx in range(n_splits):
    net = nn.Linear(28*28, 10)
    model = Classifier(
        net=net, 
        optimizer=torch.optim.Adam(net.parameters(), lr=0.003)
    )
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, data(fold_idx))
```

In configuration:

```yaml
data:
  class_path: autolightning.datasets.MNIST
  init_args:
    cross_val:
      n_splits: 5
      fold: 0
```

### Random Split

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

Use the AutoWandbLogger in your configuration to track results.

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

## Pre-made Training Methods

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

## Model Loading and Modification

### Loading Pre-trained Models

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

### Compiling Models

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

### Freezing Model Parameters

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

### Running from a Config File in a Script

```python
from autolightning.main import auto_main

# Load config from files
trainer, model, datamodule = auto_main(
    config=["config.yaml", "local.yaml"],
    subcommand="fit",
    run=True  # Set to False to just initialize without running
)

# Or directly from a dictionary
config = {
    "model": {...},
    "data": {...},
    "trainer": {...}
}

trainer, model, datamodule = auto_main(
    config=config,
    subcommand="fit",
    run=True
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