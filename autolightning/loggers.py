from typing import Dict, Mapping, Optional, Union, Literal

from lightning.pytorch.loggers import (
    CometLogger,
    CSVLogger,
    Logger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .types import Phase


LOG_PHASE_KEYS = {"train", "val", "test", "predict"}
LOG_ORDER_OPTIONS = {"phase_first", "metric_first"}


class LogKeyMixin:
    """This mixin class makes it possible to use a custom log key format for the loggers without having to
    change your logging code in the AutoModule class.
    """

    def __init__(
        self,
        separator: str = "/",
        order: str = "phase_first",
        prefix: str = "",
        postfix: str = "",
        phase_mapping: Dict[str, str] = None,
        metric_mapping: Dict[str, str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.separator = separator
        self.order = order
        self.prefix = prefix
        self.postfix = postfix

        if phase_mapping is not None:
            full_phase_mapping = {phase: phase for phase in LOG_PHASE_KEYS}
            full_phase_mapping.update(phase_mapping.copy())

            self.phase_mapping = full_phase_mapping
        else:
            self.phase_mapping = {}

        self.metric_mapping = {} if metric_mapping is None else metric_mapping

    def get_log_key(self, phase: Union[Phase, Literal["epoch"]], *metrics: str) -> str:
        # TODO: explain why this is necessary?
        if phase == "epoch":
            return "epoch"

        phase_key = self.phase_mapping.get(phase, phase)
        metric = self.separator.join([self.metric_mapping.get(metric, metric) for metric in metrics])

        key = ""

        if self.order == "phase_first":
            key = f"{phase_key}{self.separator}{metric}"
        elif self.order == "metric_first":
            key = f"{metric}{self.separator}{phase_key}"

        if key != "":
            return f"{self.prefix}{key}{self.postfix}"

        raise ValueError(f"Invalid log order: {self.order}; must be either {LOG_ORDER_OPTIONS}")

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        modified_metrics = {self.get_log_key(*key.split("/")): value for key, value in metrics.items()}

        super().log_metrics(key_modified_metrics, step)


class PandasLogger(Logger):
    def __init__(self):
        super().__init__()

        import pandas as pd

        self._df = pd.DataFrame()
        self._hparams = {}

    @property
    def name(self):
        return "PandasLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        self._hparams = params

    @rank_zero_only
    def log_metrics(self, metrics, step):
        import pandas as pd

        self._df = pd.concat([self._df, pd.DataFrame([{**metrics, "step": step}])], ignore_index=True)

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass

    def get_logs(self):
        return self._df

    def get_hyperparams(self):
        return self._hparams


# Note: I originally tried to programmatically generate the classes below using type(),
# but the type checking of jsonargparse did not like it + static analysis tools used
# in IDEs obviously cannot see the generated classes. So, I did it the manual way.


class AutoCometLogger(LogKeyMixin, CometLogger):
    pass


class AutoCSVLogger(LogKeyMixin, CSVLogger):
    pass


class AutoLogger(LogKeyMixin, Logger):
    pass


class AutoMLFlowLogger(LogKeyMixin, MLFlowLogger):
    pass


class AutoNeptuneLogger(LogKeyMixin, NeptuneLogger):
    pass


class AutoTensorBoardLogger(LogKeyMixin, TensorBoardLogger):
    pass


class AutoWandbLogger(LogKeyMixin, WandbLogger):
    pass


class AutoPandasLogger(LogKeyMixin, PandasLogger):
    pass


__all__ = [
    "AutoCometLogger",
    "AutoCSVLogger",
    "AutoLogger",
    "AutoMLFlowLogger",
    "AutoNeptuneLogger",
    "AutoTensorBoardLogger",
    "AutoWandbLogger",
    "PandasLogger",
    "AutoPandasLogger",
]
