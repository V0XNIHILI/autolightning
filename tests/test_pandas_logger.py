import pandas as pd
import pytest
from unittest.mock import patch

from autolightning.loggers import PandasLogger


@pytest.fixture
def logger():
    return PandasLogger()


def test_name(logger):
    assert logger.name == "PandasLogger"


def test_version(logger):
    assert logger.version == "0.1"


def test_log_hyperparams(logger):
    params = {"lr": 0.01, "batch_size": 32}
    logger.log_hyperparams(params)
    assert logger.get_hyperparams() == params


def test_log_metrics(logger):
    metrics1 = {"accuracy": 0.9, "loss": 0.1}
    metrics2 = {"accuracy": 0.92, "loss": 0.08}
    step1 = 1
    step2 = 2

    logger.log_metrics(metrics1, step1)
    logger.log_metrics(metrics2, step2)

    expected_df = pd.DataFrame([{**metrics1, "step": step1}, {**metrics2, "step": step2}])
    pd.testing.assert_frame_equal(logger.get_logs(), expected_df)


def test_log_metrics_empty(logger):
    assert logger.get_logs().empty


def test_save(logger):
    with patch.object(logger, "save", return_value=None) as mock_save:
        logger.save()
        mock_save.assert_called_once()


def test_finalize(logger):
    status = "COMPLETED"
    with patch.object(logger, "finalize", return_value=None) as mock_finalize:
        logger.finalize(status)
        mock_finalize.assert_called_once_with(status)
