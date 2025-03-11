import pytest
import torch.nn as nn
import torch.optim as optim
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import StepLR

from autolightning import AutoModule


# Dummy Components
class DummyNet(nn.Linear):
    def __init__(self):
        super().__init__(1, 1)

    def forward(self, x):
        return self(x)
    
    def shared_step(self, phase, batch, batch_idx):
        x, y = batch
        output = self(x)
        return (output, y)


class DummyCriterion(nn.Module):
    def forward(self, input, target):
        return input - target


class DummyMetric(Metric):
    def update(self, preds, target):
        pass

    def compute(self):
        return 0


# Fixtures for different module types
@pytest.fixture
def dummy_module_list():
    return nn.ModuleList([
        nn.Linear(10, 5),
        nn.Linear(5, 1)
    ])


@pytest.fixture
def dummy_module_dict():
    return nn.ModuleDict({
        'encoder': nn.Linear(10, 5),
        'decoder': nn.Linear(5, 1)
    })


# Callable Fixtures
@pytest.fixture
def dummy_metric_callable():
    def metric(input, target):
        return 0
    return metric


@pytest.fixture
def dummy_optimizer_callable():
    def optimizer(params):
        return optim.SGD(params, lr=0.1)
    return optimizer


@pytest.fixture
def dummy_lr_scheduler_callable():
    def scheduler(optimizer):
        return StepLR(optimizer, step_size=1)
    return scheduler


# AutoModule Fixture
@pytest.fixture
def dummy_net():
    return DummyNet()


@pytest.fixture
def dummy_module(dummy_net):
    return AutoModule(net=dummy_net)


@pytest.fixture
def dummy_criterion():
    return DummyCriterion()


# Tests
def test_net_only():
    nets = [None, DummyNet()]
    for net in nets:
        module = AutoModule(net=net)
        assert module.net == net

        if net:
            assert list(net.parameters()) == list(module.parameters_for_optimizer())


def test_parameters_for_optimizer(dummy_net):
    # Set all parameters to not require grad
    for param in dummy_net.parameters():
        param.requires_grad = False

    module = AutoModule(net=dummy_net, exclude_no_grad=True)
    assert list(module.parameters_for_optimizer()) == []

    module = AutoModule(net=dummy_net, exclude_no_grad=False)
    assert module.exclude_no_grad == False
    assert list(module.parameters_for_optimizer()) == list(dummy_net.parameters())


def test_register_none_optimizer(dummy_module):
    assert dummy_module.optimizers_schedulers == {}
    assert dummy_module.configure_optimizers() is None


def test_register_optimizer(dummy_net, dummy_module):
    dummy_module.register_optimizer(dummy_net, optim.SGD(dummy_net.parameters(), lr=0.1))

    assert isinstance(dummy_module.optimizers_schedulers[dummy_net][0], optim.SGD)
    assert dummy_module.optimizers_schedulers[dummy_net][1] is None

    configured_optimizers = dummy_module.configure_optimizers()
    assert isinstance(configured_optimizers, optim.SGD)


def test_register_callable_optimizer(dummy_net, dummy_optimizer_callable, dummy_module):
    dummy_module.register_optimizer(dummy_net, dummy_optimizer_callable)

    assert callable(dummy_module.optimizers_schedulers[dummy_net][0])
    assert dummy_module.optimizers_schedulers[dummy_net][1] is None

    configured_optimizers = dummy_module.configure_optimizers()
    assert isinstance(configured_optimizers, optim.SGD)
    assert configured_optimizers.param_groups[0]["lr"] == 0.1
    assert configured_optimizers.param_groups[0]["params"] == list(dummy_net.parameters())


def test_register_optimizer_scheduler(dummy_net, dummy_module):
    optimizer = optim.SGD(dummy_net.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1)
    dummy_module.register_optimizer(dummy_net, optimizer, scheduler)

    assert isinstance(dummy_module.optimizers_schedulers[dummy_net][0], optim.SGD)
    assert isinstance(dummy_module.optimizers_schedulers[dummy_net][1], StepLR)

    configured_optimizers = dummy_module.configure_optimizers()
    assert isinstance(configured_optimizers["optimizer"], optim.SGD)
    assert isinstance(configured_optimizers["lr_scheduler"], StepLR)


def test_register_optimizer_scheduler_callables(dummy_net, dummy_optimizer_callable, dummy_lr_scheduler_callable, dummy_module):
    dummy_module.register_optimizer(dummy_net, dummy_optimizer_callable, dummy_lr_scheduler_callable)

    assert callable(dummy_module.optimizers_schedulers[dummy_net][0])
    assert callable(dummy_module.optimizers_schedulers[dummy_net][1])

    configured_optimizers = dummy_module.configure_optimizers()
    assert isinstance(configured_optimizers["optimizer"], optim.SGD)
    assert isinstance(configured_optimizers["lr_scheduler"], StepLR)


def test_register_optimizer_callable_scheduler(dummy_net, dummy_lr_scheduler_callable, dummy_module):
    optimizer = optim.SGD(dummy_net.parameters(), lr=0.1)
    dummy_module.register_optimizer(dummy_net, optimizer, dummy_lr_scheduler_callable)

    assert isinstance(dummy_module.optimizers_schedulers[dummy_net][0], optim.SGD)
    assert callable(dummy_module.optimizers_schedulers[dummy_net][1])

    configured_optimizers = dummy_module.configure_optimizers()
    assert isinstance(configured_optimizers["optimizer"], optim.SGD)
    assert isinstance(configured_optimizers["lr_scheduler"], StepLR)


def test_optimizer_with_scheduler_dict_callable(dummy_net, dummy_criterion):
    optimizer = optim.SGD(dummy_net.parameters(), lr=0.1)
    scheduler = {
        "scheduler": lambda opt: optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1),
        "monitor": "val/loss",
        "frequency": 1
    }
    
    module = AutoModule(net=dummy_net, criterion=dummy_criterion, optimizer=optimizer, lr_scheduler=scheduler)
    
    result = module.configure_optimizers()
    
    assert isinstance(result, dict)
    assert isinstance(result["optimizer"], optim.SGD)
    assert isinstance(result["lr_scheduler"]["scheduler"], optim.lr_scheduler.StepLR)
    assert result["lr_scheduler"]["monitor"] == "val/loss"
    assert result["lr_scheduler"]["frequency"] == 1


def test_callable_optimizer_with_scheduler_dict_callable(dummy_net, dummy_criterion):
    optimizer = lambda params: optim.SGD(params, lr=0.1)
    scheduler = {
        "scheduler": lambda opt: optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1),
        "monitor": "val/loss",
        "frequency": 1
    }
    
    module = AutoModule(net=dummy_net, criterion=dummy_criterion, optimizer=optimizer, lr_scheduler=scheduler)
    
    result = module.configure_optimizers()
    
    assert isinstance(result, dict)
    assert isinstance(result["optimizer"], optim.SGD)
    assert result["optimizer"].param_groups[0]["lr"] == 0.1
    assert result["optimizer"].param_groups[0]["params"] == list(dummy_net.parameters())
    assert isinstance(result["lr_scheduler"]["scheduler"], optim.lr_scheduler.StepLR)
    assert result["lr_scheduler"]["monitor"] == "val/loss"
    assert result["lr_scheduler"]["frequency"] == 1


def test_optimizer_with_scheduler_dict(dummy_net, dummy_criterion):
    optimizer = optim.SGD(dummy_net.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    module = AutoModule(net=dummy_net, criterion=dummy_criterion, optimizer=optimizer, lr_scheduler=scheduler)
    
    result = module.configure_optimizers()
    
    assert isinstance(result, dict)
    assert isinstance(result["optimizer"], optim.SGD)
    assert isinstance(result["lr_scheduler"], optim.lr_scheduler.StepLR)


# New tests extending the existing ones
def test_optimizer_list_instances(dummy_net, dummy_criterion):
    """Test with a list of optimizer instances."""
    optimizer1 = optim.SGD([param for name, param in dummy_net.named_parameters() if "bias" in name], lr=0.01)
    optimizer2 = optim.Adam([param for name, param in dummy_net.named_parameters() if "bias" not in name], lr=0.001)
    
    module = AutoModule(net=dummy_net, criterion=dummy_criterion, optimizer=[optimizer1, optimizer2])
    
    result = module.configure_optimizers()
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], optim.SGD)
    assert isinstance(result[1], optim.Adam)


def test_optimizer_list_callables(dummy_net, dummy_criterion):
    """Test with a list of optimizer callables."""
    optimizer = [
        lambda params: optim.SGD(params, lr=0.01),
        lambda params: optim.Adam(params, lr=0.001)
    ]
    
    module = AutoModule(net=dummy_net, criterion=dummy_criterion, optimizer=optimizer)
    
    result = module.configure_optimizers()
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], optim.SGD)
    assert isinstance(result[1], optim.Adam)
    assert result[0].param_groups[0]["lr"] == 0.01
    assert result[1].param_groups[0]["lr"] == 0.001
    assert result[0].param_groups[0]["params"] == list(dummy_net.parameters())
    assert result[1].param_groups[0]["params"] == list(dummy_net.parameters())


def test_optimizer_list_callables_with_module_list(dummy_module_list, dummy_criterion):
    """Test with a list of optimizer callables with a ModuleList."""
    optimizer = [
        lambda params: optim.SGD(params, lr=0.01),
        lambda params: optim.Adam(params, lr=0.001)
    ]
    
    module = AutoModule(net=dummy_module_list, criterion=dummy_criterion)
    module.register_optimizer(dummy_module_list, optimizer)
    
    result = module.configure_optimizers()
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], optim.SGD)
    assert isinstance(result[1], optim.Adam)
    assert result[0].param_groups[0]["lr"] == 0.01
    assert result[1].param_groups[0]["lr"] == 0.001
    assert result[0].param_groups[0]["params"] == list(dummy_module_list[0].parameters())
    assert result[1].param_groups[0]["params"] == list(dummy_module_list[1].parameters())


def test_optimizer_dict_with_moduledict(dummy_module_dict, dummy_criterion):
    """Test with a dictionary of optimizers and a ModuleDict."""
    optimizer = {
        'encoder': lambda params: optim.SGD(params, lr=0.01),
        'decoder': lambda params: optim.Adam(params, lr=0.001)
    }
    
    module = AutoModule(net=dummy_module_dict, criterion=dummy_criterion)
    module.register_optimizer(dummy_module_dict, optimizer)
    
    result = module.configure_optimizers()
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], optim.SGD)
    assert isinstance(result[1], optim.Adam)
    assert result[0].param_groups[0]["lr"] == 0.01
    assert result[1].param_groups[0]["lr"] == 0.001
    assert result[0].param_groups[0]["params"] == list(dummy_module_dict['encoder'].parameters())
    assert result[1].param_groups[0]["params"] == list(dummy_module_dict['decoder'].parameters())


def test_optimizer_dict_with_non_moduledict(dummy_net, dummy_criterion):
    """Test with a dictionary of optimizers and a non-ModuleDict (should raise an error)."""
    optimizer = {
        'layer1': lambda params: optim.SGD(params, lr=0.01),
        'layer2': lambda params: optim.Adam(params, lr=0.001)
    }
    
    module = AutoModule(net=dummy_net, criterion=dummy_criterion)
    module.register_optimizer(dummy_net, optimizer)
    
    with pytest.raises(ValueError, match="Cannot use optimizer dict with non-ModuleDict module"):
        module.configure_optimizers()


def test_list_optimizers_with_scheduler(dummy_net, dummy_criterion, dummy_lr_scheduler_callable):
    """Test with a list of optimizers and a scheduler (should raise an error)."""
    optimizer = [
        lambda params: optim.SGD(params, lr=0.01),
        lambda params: optim.Adam(params, lr=0.001)
    ]
    
    module = AutoModule(net=dummy_net, criterion=dummy_criterion)

    module.register_optimizer(dummy_net, optimizer, dummy_lr_scheduler_callable)
    
    with pytest.raises(AssertionError, match="Cannot use a list of optimizers with a scheduler"):
        module.configure_optimizers()


def test_dict_optimizers_with_scheduler(dummy_module_dict, dummy_criterion, dummy_lr_scheduler_callable):
    """Test with a dict of optimizers and a scheduler (should raise an error)."""
    optimizer = {
        'encoder': lambda params: optim.SGD(params, lr=0.01),
        'decoder': lambda params: optim.Adam(params, lr=0.001)
    }
    
    module = AutoModule(net=dummy_module_dict, criterion=dummy_criterion)

    module.register_optimizer(dummy_module_dict, optimizer, dummy_lr_scheduler_callable)
    
    with pytest.raises(AssertionError, match="Cannot use a dict of optimizers with a scheduler"):
        module.configure_optimizers()


def test_multiple_optimizers_and_schedulers(dummy_module_dict, dummy_criterion):
    """Test with multiple optimizers and schedulers."""
    module = AutoModule(criterion=dummy_criterion, net=dummy_module_dict)

    # Register optimizers and schedulers for encoder and decoder
    module.register_optimizer(
        dummy_module_dict['encoder'],
        lambda params: optim.SGD(params, lr=0.01),
        lambda opt: optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    )
    module.register_optimizer(
        dummy_module_dict['decoder'],
        lambda params: optim.Adam(params, lr=0.001),
        lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    )
    
    result = module.configure_optimizers()
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert len(result[0]) == 2  # optimizers
    assert len(result[1]) == 2  # schedulers
    
    # Check optimizer types
    assert isinstance(result[0][0], optim.SGD)
    assert isinstance(result[0][1], optim.Adam)
    
    # Check scheduler types
    assert isinstance(result[1][0], optim.lr_scheduler.StepLR)
    assert isinstance(result[1][1], optim.lr_scheduler.ExponentialLR)


def test_invalid_optimizer_type(dummy_net, dummy_criterion):
    """Test with an invalid optimizer type."""
    module = AutoModule(net=dummy_net, criterion=dummy_criterion)
    
    module.register_optimizer(dummy_net, "invalid_optimizer")

    with pytest.raises(TypeError, match="Invalid optimizer type"):
        module.configure_optimizers()


def test_scheduler_without_optimizer(dummy_net, dummy_criterion, dummy_lr_scheduler_callable):
    """Test with a scheduler but no optimizer."""
    with pytest.raises(ValueError, match="Cannot register a scheduler when the optimizer is None"):
        AutoModule(net=dummy_net, criterion=dummy_criterion, lr_scheduler=dummy_lr_scheduler_callable)


def test_invalid_scheduler_type(dummy_net, dummy_criterion):
    """Test with an invalid scheduler type."""
    optimizer = optim.SGD(dummy_net.parameters(), lr=0.1)
    
    module = AutoModule(net=dummy_net, criterion=dummy_criterion)
    
    module.register_optimizer(dummy_net, optimizer, "invalid_scheduler")

    with pytest.raises(TypeError, match="Invalid scheduler type"):
        module.configure_optimizers()
