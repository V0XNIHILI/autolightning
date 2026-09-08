import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from collections import OrderedDict
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

from autolightning.utils import (
    _import_module,
    load,
    compile,
    disable_grad,
    remove_n_layers,
    optim,
    sched,
    init_kwargs,
    LIGHTNING_STATE_DICT_KEYS
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ModelWithSubmodule(nn.Module):
    """Model with nested submodules for testing."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 10)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


def test_import_module():
    """Tests that _import_module correctly imports modules and functions."""
    # Test importing a module with full path
    nn_module = _import_module("torch.nn")
    assert nn_module is torch.nn

    torch_module = _import_module("torch")
    assert torch_module is torch
    
    # Test importing a function with full path
    linear_func = _import_module("torch.nn.Linear")
    assert linear_func is nn.Linear
    
    # Test using default module
    optim_func = _import_module("Adam", default_module="torch.optim")
    assert optim_func is torch.optim.Adam


@pytest.fixture
def mock_state_dict():
    """Create a mock state dict for testing load function."""
    return OrderedDict({
        "layer1.weight": torch.randn(5, 10),
        "layer1.bias": torch.randn(5),
        "layer2.weight": torch.randn(2, 5),
        "layer2.bias": torch.randn(2),
    })


@pytest.fixture
def mock_lightning_state_dict(mock_state_dict):
    """Create a mock Lightning checkpoint state dict."""
    lightning_dict = {key: "mock_value" for key in LIGHTNING_STATE_DICT_KEYS}
    lightning_dict["state_dict"] = mock_state_dict
    return lightning_dict


@pytest.fixture
def mock_nested_state_dict():
    """Create a mock state dict with nested submodule structure."""
    return OrderedDict({
        "encoder.0.weight": torch.randn(8, 10),
        "encoder.0.bias": torch.randn(8),
        "encoder.2.weight": torch.randn(5, 8),
        "encoder.2.bias": torch.randn(5),
        "decoder.0.weight": torch.randn(8, 5),
        "decoder.0.bias": torch.randn(8),
        "decoder.2.weight": torch.randn(10, 8),
        "decoder.2.bias": torch.randn(10),
    })


def test_load_regular_state_dict(mock_state_dict):
    """Tests loading a regular state dict into a model."""
    model = SimpleModel()
    
    # Mock torch.load to return our mock state dict
    with patch('torch.load', return_value=mock_state_dict):
        result = load(model, 'dummy_path.pt', submodule_path=None)
    
    # Check that the model was returned
    assert result is model
    
    # Check that the model parameters match the state dict
    for name, param in model.named_parameters():
        assert torch.all(param == mock_state_dict[name])


def test_load_lightning_state_dict(mock_lightning_state_dict):
    """Tests loading a Lightning checkpoint state dict into a model."""
    model = SimpleModel()
    
    # Mock torch.load to return our mock lightning state dict
    with patch('torch.load', return_value=mock_lightning_state_dict):
        result = load(model, 'dummy_path.pt', submodule_path=None)
    
    # Check that the model was returned
    assert result is model
    
    # Check that the model parameters match the state dict inside the lightning dict
    for name, param in model.named_parameters():
        assert torch.all(param == mock_lightning_state_dict["state_dict"][name])


def test_load_with_submodule_path(mock_nested_state_dict):
    """Tests loading part of a state dict using submodule path."""
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 5)
    )
    
    # Mock torch.load to return our mock nested state dict
    with patch('torch.load', return_value=mock_nested_state_dict):
        result = load(model, 'dummy_path.pt', submodule_path='encoder')
    
    # Check that the model was returned
    assert result is model
    
    # Check that only the encoder parameters were loaded
    for i, (name, param) in enumerate(model.named_parameters()):
        if i % 2 == 0:  # weights
            layer_idx = i // 2
            encoder_key = f"encoder.{layer_idx}.weight"
            assert torch.all(param == mock_nested_state_dict[encoder_key])
        else:  # biases
            layer_idx = (i-1) // 2
            encoder_key = f"encoder.{layer_idx}.bias"
            assert torch.all(param == mock_nested_state_dict[encoder_key])


def test_compile():
    """Tests compiling a module with a custom compiler function."""
    model = SimpleModel()
    
    # Define a simple mock compiler function
    def mock_compiler(module, dummy_arg=None):
        # Just return the module for testing
        module.was_compiled = True
        module.compiler_arg = dummy_arg
        return module
    
    # Mock _import_module to return our mock compiler
    with patch('autolightning.utils._import_module', return_value=mock_compiler):
        result = compile(model, 'mock_compiler', {'dummy_arg': 'test'})
    
    # Check the module was compiled with our mock compiler
    assert result is model
    assert hasattr(result, 'was_compiled')
    assert result.was_compiled
    assert result.compiler_arg == 'test'


def test_disable_grad():
    """Tests disabling gradients for all parameters in a model."""
    model = SimpleModel()
    
    # Initially all parameters should require gradients
    for param in model.parameters():
        assert param.requires_grad
    
    # Disable gradients
    result = disable_grad(model)
    
    # Check the model was returned
    assert result is model
    
    # Check all parameters have requires_grad=False
    for param in model.parameters():
        assert not param.requires_grad


def test_remove_n_layers_positive():
    """Tests removing first N layers with positive index."""
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Remove first 2 layers
    result = remove_n_layers(model, 2)
    
    # Check we get a sequential
    assert isinstance(result, nn.Sequential)
    
    # Check we have 3 layers left
    assert len(result) == 3
    
    # Check the remaining layers are the correct ones
    assert isinstance(result[0], nn.Linear)
    assert result[0].in_features == 8
    assert result[0].out_features == 5


def test_remove_n_layers_negative():
    """Tests removing last N layers with negative index."""
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Remove last 2 layers
    result = remove_n_layers(model, -2)
    
    # Check we get a sequential
    assert isinstance(result, nn.Sequential)
    
    # Check we have 3 layers left
    assert len(result) == 3
    
    # Check the remaining layers are the correct ones
    assert isinstance(result[0], nn.Linear)
    assert result[0].in_features == 10
    assert result[0].out_features == 8


def test_optim_function():
    """Tests the optim helper creates correct optimizer callables."""
    # Test with default module (torch.optim)
    adam_fn = optim("Adam", lr=0.01, betas=(0.8, 0.99))
    
    # Create a model and apply the optimizer
    model = SimpleModel()
    optimizer = adam_fn(model.parameters())
    
    # Check we got the right optimizer with the right params
    assert isinstance(optimizer, Adam)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.99)
    
    # Test with explicit full path
    sgd_fn = optim("torch.optim.SGD", lr=0.1, momentum=0.9)
    
    # Apply the optimizer
    optimizer = sgd_fn(model.parameters())
    
    # Check we got the right optimizer with the right params
    assert isinstance(optimizer, SGD)
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[0]["momentum"] == 0.9


def test_sched_function():
    """Tests the sched helper creates correct scheduler callables."""
    # Test with default module (torch.optim.lr_scheduler)
    step_lr_fn = sched("StepLR", step_size=10, gamma=0.1)
    
    # Create a model, optimizer, and apply the scheduler
    model = SimpleModel()
    optimizer = Adam(model.parameters())
    scheduler = step_lr_fn(optimizer)
    
    # Check we got the right scheduler with the right params
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == 10
    assert scheduler.gamma == 0.1
    
    # Test with explicit full path
    step_lr_fn = sched("torch.optim.lr_scheduler.StepLR", step_size=5, gamma=0.5)
    
    # Apply the scheduler
    scheduler = step_lr_fn(optimizer)
    
    # Check we got the right scheduler with the right params
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == 5
    assert scheduler.gamma == 0.5


def test_init_kwargs_simple():
    """Tests init_kwargs with simple (non-class) configurations."""
    config = {
        "a": 1,
        "b": "test",
        "c": [1, 2, 3],
        "d": {"x": 1, "y": 2}
    }
    
    # Should return the same structure without instantiating anything
    result = init_kwargs(config)
    assert result == config


def test_init_kwargs_with_class():
    """Tests init_kwargs instantiates classes from class_path configurations."""
    config = {
        "model": {
            "class_path": "torch.nn.Linear",
            "init_args": {
                "in_features": 10,
                "out_features": 5
            }
        }
    }
    
    # Mock instantiate_class to verify it's called correctly
    with patch('autolightning.utils.instantiate_class') as mock_instantiate:
        mock_instantiate.return_value = "MOCKED_CLASS"
        result = init_kwargs(config)
    
    # Check instantiate_class was called with the class config
    mock_instantiate.assert_called_once()
    args, kwargs = mock_instantiate.call_args
    assert args == tuple()
    assert kwargs == config["model"]
    
    # Check the result has the mocked class
    assert result == {"model": "MOCKED_CLASS"}


def test_init_kwargs_nested():
    """Tests init_kwargs with nested configurations including classes."""
    config = {
        "model": {
            "encoder": {
                "class_path": "torch.nn.Linear",
                "init_args": {"in_features": 10, "out_features": 5}
            },
            "decoder": {
                "class_path": "torch.nn.Linear",
                "init_args": {"in_features": 5, "out_features": 10}
            }
        },
        "training": {
            "optimizer": {
                "class_path": "torch.optim.Adam",
                "init_args": {"lr": 0.01}
            },
            "epochs": 10
        }
    }
    
    # Mock instantiate_class to verify it's called correctly
    with patch('autolightning.utils.instantiate_class') as mock_instantiate:
        # Return different values for different calls
        mock_instantiate.side_effect = ["ENCODER", "DECODER", "OPTIMIZER"]
        result = init_kwargs(config)
    
    # Check instantiate_class was called 3 times
    assert mock_instantiate.call_count == 3
    
    # Check the nested structure was preserved with instantiated classes
    assert result == {
        "model": {
            "encoder": "ENCODER",
            "decoder": "DECODER"
        },
        "training": {
            "optimizer": "OPTIMIZER",
            "epochs": 10
        }
    }


def test_init_kwargs_with_lists_and_tuples():
    """Tests init_kwargs with lists and tuples containing class configurations."""
    config = {
        "models": [
            {
                "class_path": "torch.nn.Linear",
                "init_args": {"in_features": 10, "out_features": 5}
            },
            {
                "class_path": "torch.nn.ReLU",
                "init_args": {}
            }
        ],
        "config_tuple": (
            {
                "class_path": "torch.nn.Conv2d",
                "init_args": {"in_channels": 3, "out_channels": 16, "kernel_size": 3}
            },
            "not_a_class"
        )
    }
    
    # Mock instantiate_class to verify it's called correctly
    with patch('autolightning.utils.instantiate_class') as mock_instantiate:
        # Return different values for different calls
        mock_instantiate.side_effect = ["LINEAR", "RELU", "CONV2D"]
        result = init_kwargs(config)
    
    # Check instantiate_class was called 3 times
    assert mock_instantiate.call_count == 3
    
    # Check lists and tuples were preserved with instantiated classes
    assert result == {
        "models": ["LINEAR", "RELU"],
        "config_tuple": ("CONV2D", "not_a_class")
    }