"""CPU coverage for the small trainer/metrics seams retained from Lumine."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.trace_callback import TqdmCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_progress_uses_existing_metrics_and_keeps_lr_readable():
    trainer = SimpleNamespace(
        step_train_metrics={"training/total_loss": 0.125, "training/grad_norm": 2.0, "training/lr": 1e-5},
        step_env_metrics={"mfu": 0.375, "tokens_per_second(M)": 1.25, "max_memory_allocated(GB)": 10.125},
    )
    original_metrics = trainer.step_env_metrics.copy()
    callback = TqdmCallback(trainer)
    callback.data_loader_tqdm = MagicMock()
    callback.on_step_end(TrainerState(global_step=1))
    callback.data_loader_tqdm.set_postfix_str.assert_called_once_with(
        "total_loss: 0.12, grad_norm: 2.00, lr: 1.00e-05, mfu: 37.50%, tokens/s: 1.25M, gpu_mem: 10.1GB",
        refresh=False,
    )
    callback.data_loader_tqdm.update.assert_called_once_with()
    assert trainer.step_env_metrics == original_metrics


def test_progress_omits_unavailable_environment_metrics():
    trainer = SimpleNamespace(step_train_metrics={"training/total_loss": 1.5}, step_env_metrics={})
    callback = TqdmCallback(trainer)
    callback.data_loader_tqdm = MagicMock()
    callback.on_step_end(TrainerState(global_step=1))
    callback.data_loader_tqdm.set_postfix_str.assert_called_once_with("total_loss: 1.50", refresh=False)


@pytest.mark.parametrize("completed, planned, refresh", [(3, 10, True), (10, 10, False), (0, 10, True)])
def test_epoch_progress_finishes_at_successful_batch_count(completed, planned, refresh):
    callback = TqdmCallback(SimpleNamespace())
    callback.data_loader_tqdm = MagicMock(n=completed, total=planned)
    callback.on_epoch_end(TrainerState())
    assert callback.data_loader_tqdm.total == completed
    assert callback.data_loader_tqdm.refresh.call_count == int(refresh)
    callback.data_loader_tqdm.close.assert_called_once_with()


@dataclass
class _DatasetArguments:
    dataset_name = "mock"
    datasets_type: str = "mapping"
    dataloader: object = field(default_factory=lambda: SimpleNamespace(shuffle=False))


@pytest.mark.parametrize("steps", [0, -1, 1, 10])
def test_runtime_rejects_empty_training_schedule_before_dataloader(monkeypatch, steps):
    compute = MagicMock()
    args = SimpleNamespace(
        data=_DatasetArguments(),
        train=SimpleNamespace(seed=42),
        model=SimpleNamespace(accelerator=SimpleNamespace(dp_size=2)),
        compute_train_steps=compute,
        train_steps=steps,
    )
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = args
    trainer.data_transform = None
    monkeypatch.setattr("veomni.trainer.base.build_dataset", lambda **kwargs: list(range(12)))
    if steps <= 0:
        with pytest.raises(ValueError, match="No training steps.*dataset size"):
            trainer._build_dataset()
    else:
        trainer._build_dataset()
        assert trainer.train_steps == steps
    compute.assert_called_once_with(6)
