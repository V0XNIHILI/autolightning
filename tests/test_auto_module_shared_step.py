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
        self.log_dict = MagicMock()

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
                "metrics": {"accuracy": (output, y)},
            }

        elif self.output_type == "full_dict":
            # Return a dict with loss, metrics, and logging kwargs
            return {
                "loss": torch.tensor(1.0, requires_grad=True),
                "metrics": {"accuracy": (output, y)},
                "log_kwargs": {"on_epoch": True},
                "log_dict": {"custom_metric": 0.95},
            }

        else:
            # Return None
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
        "f1": {"metric": TestMetric(), "log_kwargs": {"prog_bar": True}},
    }


def test_shared_logged_step_tensor_loss(dummy_batch):
    """Test shared_logged_step with a tensor loss output"""
    module = TestSharedStepOutput(output_type="tensor", loss_log_key="loss")

    loss = module.shared_logged_step("train", dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

    # Check that log_dict was called with the loss
    module.log_dict.assert_called_once()
    assert "train/loss" in module.log_dict.call_args[0][0]
    assert module.log_dict.call_args[0][0]["train/loss"].item() == 1.0


def test_shared_logged_step_tuple(dummy_batch, dummy_criterion):
    """Test shared_logged_step with tuple output"""
    module = TestSharedStepOutput(output_type="tuple", criterion=dummy_criterion, loss_log_key="loss")

    # Mock the criterion to return a known loss
    module.criterion = MagicMock(return_value=torch.tensor(2.0))

    loss = module.shared_logged_step("val", dummy_batch, 0)

    # Check that criterion was called
    module.criterion.assert_called_once()

    # Check that the loss is returned correctly
    assert loss.item() == 2.0

    # Check that log_dict was called with the loss
    module.log_dict.assert_called_once()
    assert "val/loss" in module.log_dict.call_args[0][0]
    assert module.log_dict.call_args[0][0]["val/loss"].item() == 2.0


def test_shared_logged_step_loss_dict(dummy_batch):
    """Test shared_logged_step with dict containing 'loss' key"""
    module = TestSharedStepOutput(output_type="loss_dict", loss_log_key="loss")

    loss = module.shared_logged_step("train", dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

    # Check that log_dict was called with the loss
    module.log_dict.assert_called_once()
    assert "train/loss" in module.log_dict.call_args[0][0]
    assert module.log_dict.call_args[0][0]["train/loss"].item() == 1.0


def test_shared_logged_step_criterion_args_dict(dummy_batch, dummy_criterion):
    """Test shared_logged_step with dict containing 'criterion_args' key"""
    module = TestSharedStepOutput(
        output_type="criterion_args_dict",
        criterion=dummy_criterion,
        loss_log_key="loss",
    )

    # Mock the criterion to return a known loss
    module.criterion = MagicMock(return_value=torch.tensor(2.0))

    loss = module.shared_logged_step("test", dummy_batch, 0)

    # Check that criterion was called
    module.criterion.assert_called_once()

    # Check that the loss is returned correctly
    assert loss.item() == 2.0

    # Check that log_dict was called with the loss
    module.log_dict.assert_called_once()
    assert "test/loss" in module.log_dict.call_args[0][0]
    assert module.log_dict.call_args[0][0]["test/loss"].item() == 2.0


def test_shared_logged_step_with_metrics(dummy_batch, dummy_metrics):
    """Test shared_logged_step with metrics dictionary"""
    module = TestSharedStepOutput(output_type="loss_and_metrics_dict", metrics=dummy_metrics, loss_log_key="loss")

    loss = module.shared_logged_step("val", dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

    # Check that log_dict was called with the loss
    module.log_dict.assert_called_once()
    assert "val/loss" in module.log_dict.call_args[0][0]
    assert module.log_dict.call_args[0][0]["val/loss"].item() == 1.0

    # Check that metrics were logged
    assert module.log.call_count == 2
    assert any("val/accuracy" in call[0][0] for call in module.log.call_args_list)
    assert any("val/f1" in call[0][0] for call in module.log.call_args_list)


def test_shared_logged_step_full_dict(dummy_batch, dummy_metrics):
    """Test shared_logged_step with full dictionary including log_kwargs"""
    module = TestSharedStepOutput(output_type="full_dict", metrics=dummy_metrics, loss_log_key="loss")

    loss = module.shared_logged_step("train", dummy_batch, 0)

    # Check that the loss is returned correctly
    assert loss.item() == 1.0

    # Check that log_dict was called with the loss and custom metric
    module.log_dict.assert_called_once()
    log_dict_args = module.log_dict.call_args[0][0]
    assert "train/loss" in log_dict_args
    assert log_dict_args["train/loss"].item() == 1.0
    assert "train/custom_metric" in log_dict_args
    assert log_dict_args["train/custom_metric"] == 0.95

    # Check that log_dict was called with the custom kwargs
    assert module.log_dict.call_args[1].get("on_epoch") is True

    # Check that metrics were logged
    assert module.log.call_count == 2
    assert any("train/accuracy" in call[0][0] for call in module.log.call_args_list)
    assert any("train/f1" in call[0][0] for call in module.log.call_args_list)


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


def test_integration_different_phases(real_implementation, dummy_batch):
    """Integration test for different phases with different output formats"""

    # Train phase returns a tensor
    train_loss = real_implementation.training_step(dummy_batch, 0)
    assert isinstance(train_loss, torch.Tensor)

    # Val phase returns dict with metrics
    val_loss = real_implementation.validation_step(dummy_batch, 0)
    assert isinstance(val_loss, torch.Tensor)

    # Test phase returns dict with log_dict
    test_loss = real_implementation.test_step(dummy_batch, 0)
    assert isinstance(test_loss, torch.Tensor)

    # Check logging calls
    assert real_implementation.log_dict.call_count == 3

    # Check the val phase logged metrics
    assert real_implementation.log.call_count >= 2

    # Check that custom value was logged in test phase
    test_log_dict_call = real_implementation.log_dict.call_args_list[2][0][0]
    assert "test/custom_value" in test_log_dict_call
    assert test_log_dict_call["test/custom_value"] == 0.5
