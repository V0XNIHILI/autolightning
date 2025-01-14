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