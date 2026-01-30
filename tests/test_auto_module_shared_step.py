import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from autolightning import AutoModule


class TestSharedStepOutput(AutoModule):
    """Test implementation of AutoModule with different shared_step outputs"""

    def __init__(self, output_type, **kwargs):
        super().__init__(**kwargs)
        self.output_type = output_type
        # Mock the log and log_dict methods
        self.log = MagicMock()

    def forward(self, x):
        if not hasattr(self, "net") or self.net is None:
            return x
        return self.net(x)

    def shared_step(self, phase, batch, batch_idx):
        x, y = batch
        output = self(x)

        if self.output_type == "tensor":
            # Return a tensor (directly computed loss)
            return torch.tensor(1.0, requires_grad=True)

        elif self.output_type == "tuple":
            # Return a tuple to be passed to criterion
            return (output, y)

        elif self.output_type == "loss_dict":
            # Return a dict with "loss" key
            return {"loss": torch.tensor(1.0, requires_grad=True)}

        elif self.output_type == "criterion_args_dict":
            # Return a dict with "criterion_args" key
            return {"criterion_args": (output, y)}

        elif self.output_type == "loss_and_metrics_dict":
            # Return a dict with loss and metrics
            return {
                "loss": torch.tensor(1.0, requires_grad=True),
                "metric_args": {"accuracy": (output, y), "f1": (output, y)},
            }

        elif self.output_type == "full_dict":
            # Return a dict with loss, metrics, and logging kwargs
            return {
                "loss": torch.tensor(1.0, requires_grad=True),
                "metric_args": {"accuracy": (output, y), "f1": (output, y)},
                "log_kwargs": {"custom_log_kwarg": True}
            }

        return None


class TestMetric(nn.Module):
    """Simple test metric that always returns 0.75"""

    def __init__(self):
        super().__init__()

    def __call__(self, preds, target):
        return torch.tensor(0.75)


@pytest.fixture
def dummy_batch():
    """Simple test batch"""
    x = torch.randn(10, 5)
    y = torch.randint(0, 2, (10,))
    return x, y


@pytest.fixture
def dummy_criterion():
    """Simple test criterion"""
    return nn.MSELoss()


@pytest.fixture
def dummy_metrics():
    """Dictionary of test metrics"""
    return {
        "accuracy": TestMetric(),
        "f1": {"func": TestMetric(), "log_kwargs": {"prog_bar": True}},
    }


def test_shared_logged_step_tensor_loss(dummy_batch):
    """Test shared_logged_step with a tensor loss output"""

    log_loss_key = "lol_testing"

    module = TestSharedStepOutput(output_type="tensor", loss_log_key=log_loss_key)

    phase = "train"
    
    loss = module.shared_logged_step(phase, dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

    # Check that log was called with the loss
    module.log.assert_called_once()
    assert module.log.call_args[0][0] == f"{phase}/{log_loss_key}"
    assert module.log.call_args[0][1].item() == 1.0


def test_shared_logged_step_tuple(dummy_batch):
    """Test shared_logged_step with tuple output"""

    log_loss_key = "lol_testing"
    
    module = TestSharedStepOutput(output_type="tuple", criterion=MagicMock(return_value=torch.tensor(2.0)), loss_log_key=log_loss_key)

    phase = "val"

    loss = module.shared_logged_step(phase, dummy_batch, 0)

    # Check that criterion was called
    module.criterion.assert_called_once()

    # Check that the loss is returned correctly
    assert loss.item() == 2.0

    # Check that log was called with the loss
    module.log.assert_called_once()
    assert module.log.call_args[0][0] == f"{phase}/{log_loss_key}"
    assert module.log.call_args[0][1].item() == 2.0


def test_shared_logged_step_loss_dict(dummy_batch):
    """Test shared_logged_step with dict containing 'loss' key"""

    log_loss_key = "more_testing"

    module = TestSharedStepOutput(output_type="loss_dict", loss_log_key=log_loss_key)

    phase = "train"

    loss = module.shared_logged_step(phase, dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

    # Check that log was called with the loss
    module.log.assert_called_once()
    assert module.log.call_args[0][0] == f"{phase}/{log_loss_key}"
    assert module.log.call_args[0][1].item() == 1.0


def test_shared_logged_step_criterion_args_dict(dummy_batch, dummy_criterion):
    """Test shared_logged_step with dict containing 'criterion_args' key"""
    module = TestSharedStepOutput(
        output_type="criterion_args_dict",
         # Mock the criterion to return a known loss
        criterion=MagicMock(return_value=torch.tensor(2.0)),
        loss_log_key="loss",
    )

    phase = "test"

    loss = module.shared_logged_step(phase, dummy_batch, 0)

    # Check that criterion was called
    module.criterion.assert_called_once()

    # Check that the loss is returned correctly
    assert loss.item() == 2.0

    # Check that log was called with the loss
    module.log.assert_called_once()
    assert module.log.call_args[0][0] == f"{phase}/loss"
    assert module.log.call_args[0][1].item() == 2.0


def test_shared_logged_step_with_metrics(dummy_batch, dummy_metrics):
    """Test shared_logged_step with metrics dictionary"""
    module = TestSharedStepOutput(output_type="loss_and_metrics_dict", metrics=dummy_metrics, loss_log_key="loss")

    loss = module.shared_logged_step("val", dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

     # Check that metrics were logged
    assert module.log.call_count == 3

    assert module.log.call_args_list[0][0][0] == "val/accuracy"
    assert module.log.call_args_list[0][0][1].item() == 0.75

    assert module.log.call_args_list[1][0][0] == "val/f1"
    assert module.log.call_args_list[1][0][1].item() == 0.75

    assert module.log.call_args_list[2][0][0] == "val/loss"
    assert module.log.call_args_list[2][0][1].item() == 1.0


def test_shared_logged_step_full_dict(dummy_batch, dummy_metrics):
    """Test shared_logged_step with full dictionary including log_kwargs"""
    module = TestSharedStepOutput(output_type="full_dict", metrics=dummy_metrics, loss_log_key="loss")

    loss = module.shared_logged_step("train", dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

    # For the loss, it is currently not possible to specify custom log_kwargs
    for i in range(2):
        assert module.log.call_args_list[i][1]['custom_log_kwarg'] is True

     # Check that metrics were logged
    assert module.log.call_count == 3

    assert module.log.call_args_list[0][0][0] == "train/accuracy"
    assert module.log.call_args_list[0][0][1].item() == 0.75

    assert module.log.call_args_list[1][0][0] == "train/f1"
    assert module.log.call_args_list[1][0][1].item() == 0.75

    assert module.log.call_args_list[2][0][0] == "train/loss"
    assert module.log.call_args_list[2][0][1].item() == 1.0


def test_shared_logged_step_disable_loss_logging(dummy_batch):
    """Test shared_logged_step with loss_log_key=None"""
    module = TestSharedStepOutput(
        output_type="tensor",
        loss_log_key=None,  # Disable loss logging
    )

    loss = module.shared_logged_step("train", dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

    # Check that log_dict was not called
    module.log_dict.assert_not_called()


def test_shared_logged_step_none_output(dummy_batch):
    """Test shared_logged_step with None output from shared_step"""
    module = TestSharedStepOutput(output_type="none", loss_log_key="loss")

    loss = module.shared_logged_step("train", dummy_batch, 0)

    # Check that loss is None
    assert loss is None

    # Check that no logging methods were called
    module.log_dict.assert_not_called()
    module.log.assert_not_called()


def test_prog_bar_enable_disable():
    """Test the should_enable_prog_bar method"""
    module = AutoModule(disable_prog_bar=False)
    assert module.should_enable_prog_bar("train") is False
    assert module.should_enable_prog_bar("val") is True
    assert module.should_enable_prog_bar("test") is False

    module = AutoModule(disable_prog_bar=True)
    assert module.should_enable_prog_bar("train") is False
    assert module.should_enable_prog_bar("val") is False
    assert module.should_enable_prog_bar("test") is False


def test_shared_logged_step_in_step_methods(dummy_batch):
    """Test that training_step, validation_step, etc. call shared_logged_step"""

    # Create a TestSharedStepOutput with mocked shared_logged_step
    module = TestSharedStepOutput(output_type="tensor")
    module.shared_logged_step = MagicMock(return_value=torch.tensor(1.0))

    # Call step methods
    train_loss = module.training_step(dummy_batch, 0)
    val_loss = module.validation_step(dummy_batch, 0)
    test_loss = module.test_step(dummy_batch, 0)

    # Check that shared_logged_step was called with the correct phase
    assert module.shared_logged_step.call_count == 3

    calls = module.shared_logged_step.call_args_list
    assert calls[0][0][0] == "train"
    assert calls[1][0][0] == "val"
    assert calls[2][0][0] == "test"

    # Check that the losses were returned correctly
    assert train_loss == torch.tensor(1.0)
    assert val_loss == torch.tensor(1.0)
    assert test_loss == torch.tensor(1.0)


class AutoModuleRealImplementation(AutoModule):
    """A more realistic implementation of AutoModule for integration testing"""

    def forward(self, x):
        return self.net(x)

    def shared_step(self, phase, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Return different formats based on phase
        if phase == "train":
            return loss
        elif phase == "val":
            return {"loss": loss, "metrics": {"my_metric": (output, y)}}
        else:  # test or predict
            return {"loss": loss, "log_dict": {"custom_value": 0.5}}


@pytest.fixture
def real_model():
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
    return model


@pytest.fixture
def real_implementation(real_model, dummy_criterion, dummy_metrics):
    """Create a realistic AutoModule implementation"""
    module = AutoModuleRealImplementation(
        net=real_model,
        criterion=dummy_criterion,
        metrics=dummy_metrics,
        loss_log_key="loss",
    )

    # Mock logging methods
    module.log = MagicMock()
    module.log_dict = MagicMock()

    return module
