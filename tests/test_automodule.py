import torch.nn as nn
import torch.optim as optim
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import StepLR

from autolightning import AutoModule


# Define dummy components for testing
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


def dummy_metric_callable(input, target):
    return 0


def dummy_optimizer_callable(params):
    return optim.SGD(params, lr=0.1)


def dummy_lr_scheduler_callable(optimizer):
    return StepLR(optimizer, step_size=1)


def test_net_only():
    nets = [None, DummyNet()]

    for net in nets:
        module = AutoModule(net=net)
        assert module.net == net

        if net != None:
            assert list(net.parameters()) == list(module.parameters_for_optimizer())


def test_parameters_for_optimizer():
    net = DummyNet()

    # Set all parameters to not require grad
    for param in net.parameters():
        param.requires_grad = False

    module = AutoModule(net=net, exclude_no_grad=True)

    assert list(module.parameters_for_optimizer()) == []

    module = AutoModule(net=net, exclude_no_grad=False)

    assert module.exclude_no_grad == False

    assert list(module.parameters_for_optimizer()) == list(net.parameters())


def test_register_none_optimizer():
    net = DummyNet()

    module = AutoModule(net=net)
    assert module.optimizers_schedulers == {}
    
    configured_optimizers = module.configure_optimizers()
    assert configured_optimizers == None


def test_register_optimizer():
    net = DummyNet()

    module = AutoModule(net=net)
    module.register_optimizer(net, optim.SGD(net.parameters(), lr=0.1))

    assert isinstance(module.optimizers_schedulers[net][0], optim.SGD)
    assert module.optimizers_schedulers[net][1] == None

    configured_optimizers = module.configure_optimizers()
    assert isinstance(configured_optimizers, optim.SGD)


def test_register_optimizer_scheduler():
    net = DummyNet()

    module = AutoModule(net=net)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1)
    module.register_optimizer(net, optimizer, scheduler)

    assert isinstance(module.optimizers_schedulers[net][0], optim.SGD)
    assert isinstance(module.optimizers_schedulers[net][1], StepLR)

    configured_optimizers = module.configure_optimizers()

    assert isinstance(configured_optimizers["optimizer"], optim.SGD)
    assert isinstance(configured_optimizers["lr_scheduler"], StepLR)


def test_register_optimizer_scheduler_callables():
    net = DummyNet()

    module = AutoModule(net=net)
    optimizer = dummy_optimizer_callable
    scheduler = dummy_lr_scheduler_callable
    module.register_optimizer(net, optimizer, scheduler)

    assert callable(module.optimizers_schedulers[net][0])
    assert callable(module.optimizers_schedulers[net][1])

    configured_optimizers = module.configure_optimizers()

    assert isinstance(configured_optimizers["optimizer"], optim.SGD)
    assert isinstance(configured_optimizers["lr_scheduler"], StepLR)


def test_register_optimizer_callable_scheduler():
    net = DummyNet()

    module = AutoModule(net=net)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    scheduler = dummy_lr_scheduler_callable
    module.register_optimizer(net, optimizer, scheduler)

    assert isinstance(module.optimizers_schedulers[net][0], optim.SGD)
    assert callable(module.optimizers_schedulers[net][1])

    configured_optimizers = module.configure_optimizers()

    assert isinstance(configured_optimizers["optimizer"], optim.SGD)
    assert isinstance(configured_optimizers["lr_scheduler"], StepLR)


# TODO:
# test configure_optimizers in more detail


# TODO:
# test shared_logged_step + shared_step + forward combinations


# # Define possible inputs
# nets = [None, DummyNet()]
# criterions = [None, DummyCriterion()]
# optimizers = [None, dummy_optimizer_callable, optim.SGD(DummyNet().parameters(), lr=0.1), 
#               [optim.SGD(DummyNet().parameters(), lr=0.1)], 
#               [dummy_optimizer_callable]]
# lr_schedulers = [None, dummy_lr_scheduler_callable]
# compilers = [None, lambda x: x]
# all_metrics = [None, {"accuracy": DummyMetric()}, {"output": dummy_metric_callable}]
# loss_log_keys = ["loss", "custom_loss"]
# log_metrics_options = [True, False]
# exclude_no_grad_options = [True, False]
# disable_prog_bar_options = [True, False]

# # Generate all possible combinations using product
# @pytest.mark.parametrize(
#     "net, criterion, optimizer, lr_scheduler, compiler, metrics, loss_log_key, log_metrics, exclude_no_grad, disable_prog_bar",
#     product(nets, criterions, optimizers, lr_schedulers, compilers, all_metrics, loss_log_keys, log_metrics_options, exclude_no_grad_options, disable_prog_bar_options)
# )
# def test_auto_module_init(net, criterion, optimizer, lr_scheduler, compiler, metrics, loss_log_key, log_metrics, exclude_no_grad, disable_prog_bar):
#     if lr_scheduler != None and optimizer == None:
#         return

#     module = AutoModule(
#         net=net,
#         criterion=criterion,
#         optimizer=optimizer,
#         lr_scheduler=lr_scheduler,
#         compiler=compiler,
#         metrics=metrics,
#         loss_log_key=loss_log_key,
#         log_metrics=log_metrics,
#         exclude_no_grad=exclude_no_grad,
#         disable_prog_bar=disable_prog_bar
#     )
    
#     assert module.net == net
#     assert module.criterion == criterion

#     if metrics == None:
#         assert module.metrics == {}
#     else:
#         assert module.metrics == metrics

#     # if optimizer != None:
#     #      assert module.optimizers_schedulers["*"] == (optimizer, lr_scheduler)
    
#     # module.configure_optimizers()
#     # optimizer_from_module = module.optimizers()
#     # assert optimizer_from_module == optimizer
#     # assert module.lr_scheduler == lr_scheduler
#     assert module.compiler == compiler
#     assert module.loss_log_key == loss_log_key
#     assert module.log_metrics == log_metrics
#     assert module.exclude_no_grad == exclude_no_grad
