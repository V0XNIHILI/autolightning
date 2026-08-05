import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from autolightning.auto_module import AutoModule, _call_with_flexible_args

warnings.filterwarnings("ignore")
torch.manual_seed(0)


def dl(n=8, batch=4):
    x = torch.randn(n, 4)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch)


def trainer(**kw):
    return L.Trainer(
        max_epochs=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        **kw,
    )


class PhaseProbe(AutoModule):
    """Feeds a constant 1.0 into the metric during train and 0.0 during val/test.

    If the metric instance is shared across phases, the epoch value for at least one
    phase will not be its own constant.
    """

    CONSTANTS = {"train": 1.0, "val": 0.0, "test": 0.5}

    def shared_step(self, phase, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        value = torch.full((x.shape[0],), self.CONSTANTS[phase])
        return {"criterion_args": (out, y), "metric_args": {"mean": (value,)}}


class SharedMetricProbe(PhaseProbe):
    """Reproduces the pre-fix behaviour: one metric instance used by every phase."""

    def _build_phase_metrics(self, metrics):
        super()._build_phase_metrics(metrics)

        shared = nn.ModuleDict()
        for name, entry in metrics.items():
            metric = entry["metric"] if isinstance(entry, dict) else entry
            if hasattr(metric, "update"):
                shared[name] = metric

        self._phase_metrics = nn.ModuleDict({f"phase_{p}": shared for p in ("train", "val", "test", "predict")})


def make_probe(cls):
    return cls(
        net=nn.Linear(4, 3),
        criterion=nn.CrossEntropyLoss(),
        optimizer=partial(optim.SGD, lr=0.01),
        metrics={"mean": {"metric": MeanMetric(), "log_kwargs": {"on_step": False, "on_epoch": True}}},
    )


def test_no_leakage_between_phases():
    m = make_probe(PhaseProbe)
    t = trainer()
    t.fit(m, dl(), dl())

    train, val = t.logged_metrics["train/mean"].item(), t.logged_metrics["val/mean"].item()
    print(f"  fixed  -> train/mean={train:.4f}  val/mean={val:.4f}  (expect 1.0 / 0.0)")
    assert abs(train - 1.0) < 1e-6, f"train/mean polluted: {train}"
    assert abs(val - 0.0) < 1e-6, f"val/mean polluted: {val}"


def test_shared_instance_does_leak():
    """Sanity check that the test would actually catch the bug.

    A shared instance fails in two ways at once, so both are asserted here:
      - val's epoch-end reset wipes the buffers before train's epoch-end compute, which
        torchmetrics reports as "compute called before update" and which yields NaN;
      - what does survive is the pooled mean of both phases' inputs, not either phase's own.
    """
    m = make_probe(SharedMetricProbe)
    t = trainer()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.fit(m, dl(), dl())

    reset_warnings = [w for w in caught if "compute" in str(w.message) and "before" in str(w.message)]
    assert reset_warnings, "expected a compute-before-update warning from the shared instance"

    train, val = t.logged_metrics["train/mean"].item(), t.logged_metrics["val/mean"].item()
    print(f"  shared -> train/mean={train:.4f}  val/mean={val:.4f}  (expect NOT 1.0 / 0.0)")
    print(f"  shared -> torchmetrics warned: {reset_warnings[0].message}")
    assert not (abs(train - 1.0) < 1e-6 and abs(val - 0.0) < 1e-6), \
        "shared instance did not leak; the leakage test proves nothing"


def test_clone_independence():
    m = make_probe(PhaseProbe)
    assert m.metrics_for("train")["mean"] is not m.metrics_for("val")["mean"]
    assert m.metrics_for("train")["mean"] is not m.metrics_for("test")["mean"]

    # clones must be real submodules, so .to()/state_dict()/DDP see them
    names = dict(m.named_modules())
    assert "_metrics.phase_train.mean" in names and "_metrics.phase_val.mean" in names
    print("  train/val/test metric instances are distinct and registered")


def test_stateless_metrics_are_shared():
    fn = lambda pred, target: (pred.argmax(-1) == target).float().mean()
    m = PhaseProbe(net=nn.Linear(4, 3), criterion=nn.CrossEntropyLoss(), metrics={"acc": fn})
    assert m.metrics_for("train")["acc"] is m.metrics_for("val")["acc"] is fn
    print("  stateless callables shared across phases (no clone)")


def test_test_phase_isolated():
    m = make_probe(PhaseProbe)
    t = trainer()
    t.fit(m, dl(), dl())
    t.test(m, dl(), verbose=False)

    test_val = t.logged_metrics["test/mean"].item()
    print(f"  test/mean={test_val:.4f} after fit (expect 0.5)")
    assert abs(test_val - 0.5) < 1e-6


class TupleProbe(AutoModule):
    def shared_step(self, phase, batch, batch_idx):
        x, y = batch
        return self.net(x), y


def test_phases_key_restricts_metric():
    m = TupleProbe(
        net=nn.Linear(4, 3),
        criterion=nn.CrossEntropyLoss(),
        optimizer=partial(optim.SGD, lr=0.01),
        metrics={
            "acc": MulticlassAccuracy(num_classes=3),
            "acc_eval": {"metric": MulticlassAccuracy(num_classes=3), "phases": ("val", "test")},
        },
    )
    assert set(m.metrics_for("train")) == {"acc"}
    assert set(m.metrics_for("val")) == {"acc", "acc_eval"}

    t = trainer()
    t.fit(m, dl(), dl())
    assert not any(k.startswith("train/acc_eval") for k in t.logged_metrics)
    assert any(k.startswith("val/acc_eval") for k in t.logged_metrics)
    print("  'phases' key keeps acc_eval out of training entirely")


def test_invalid_phase_rejected():
    try:
        TupleProbe(net=nn.Linear(4, 3), metrics={"a": {"metric": MeanMetric(), "phases": ("trian",)}})
    except ValueError as e:
        print(f"  bad phase rejected: {e}")
        return
    raise AssertionError("invalid phase was accepted")


def test_tuple_vs_list_splatting():
    seen = []
    f = lambda *a, **k: seen.append((a, k))

    _call_with_flexible_args(f, (1, 2))
    _call_with_flexible_args(f, [1, 2])
    _call_with_flexible_args(f, {"a": 1})

    assert seen[0] == ((1, 2), {}), seen[0]
    assert seen[1] == (([1, 2],), {}), seen[1]
    assert seen[2] == ((), {"a": 1}), seen[2]
    print("  tuple -> f(*args), list -> f(args), dict -> f(**args)  [as documented]")


class BinaryTuple(AutoModule):
    def configure_metrics(self):
        from torchmetrics.functional import accuracy
        return {"acc": lambda logits, y: accuracy(logits, y, task="binary")}

    def shared_step(self, phase, batch, batch_idx):
        x, y = batch
        return self.net(x).squeeze(-1), y


def binary_dl(n=16, batch=4):
    x = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch)


def make_binary(metrics):
    import torch.nn.functional as F
    return BinaryTuple(
        net=nn.Linear(4, 1),
        criterion=lambda p, t: F.binary_cross_entropy_with_logits(p, t.float()),
        optimizer=partial(optim.SGD, lr=0.01),
        metrics=metrics,
    )


def test_stateful_metrics_log_during_training():
    """Regression: update()-only + training_step's on_step default = silently nothing logged."""
    from torchmetrics import Specificity, Recall

    m = make_binary({"specificity": Specificity(task="binary"), "sensitivity": Recall(task="binary")})
    t = trainer()
    t.fit(m, binary_dl(), binary_dl())

    for key in ("train/acc", "train/specificity", "train/sensitivity",
                "val/acc", "val/specificity", "val/sensitivity"):
        assert key in t.logged_metrics, f"{key} missing from {sorted(t.logged_metrics)}"
    print(f"  logged: {sorted(k for k in t.logged_metrics if 'loss' not in k)}")


def test_on_step_stateful_uses_forward():
    """With on_step=True a stateful metric must go through forward(), or nothing is logged."""
    from torchmetrics import Specificity

    m = make_binary({
        "spec": {"metric": Specificity(task="binary"), "log_kwargs": {"on_step": True, "on_epoch": False}}
    })
    t = trainer()
    t.fit(m, binary_dl(), binary_dl())

    assert "train/spec" in t.logged_metrics, sorted(t.logged_metrics)
    print(f"  on_step=True stateful metric logged: train/spec={t.logged_metrics['train/spec'].item():.4f}")
