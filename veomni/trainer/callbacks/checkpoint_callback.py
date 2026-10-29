# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-layer callback that schedules model checkpoint I/O.

This owns the every-N-steps / epochs cadence and the one-shot sidecar export.
*What* is written is :meth:`BaseTrainer.save_dcp` /
:meth:`~BaseTrainer.save_hf_or_lora` / :meth:`~BaseTrainer.load` /
:meth:`~BaseTrainer.save_model_assets`, which fan out to the model handles.
*How* belongs to each model's
:class:`~veomni.models.checkpoint.ModelCheckpointManager`.

DCP and HF/LoRA share this callback because they share a manager; each format
still has its own cadence knobs and its own last-saved step so a DCP write
does not suppress an HF export or the reverse.

Job-level global_state — where the dataloader is, the rng, the meters — is not written
here. It has its own schedule and its own files, in
:mod:`~veomni.trainer.callbacks.global_state_callback`.
"""

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ...checkpoint import layout
from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


class CheckpointCallback(Callback):
    """Schedule DCP / HF / LoRA I/O and the one-shot tokenizer/config export."""

    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        ckpt = args.train.checkpoint
        self.dcp_every_n_steps = ckpt.save_steps
        self.dcp_every_n_epochs = ckpt.save_epochs
        self.save_hf_weights = ckpt.save_hf_weights
        self.hf_every_n_steps = ckpt.hf_save_steps
        self.hf_every_n_epochs = ckpt.hf_save_epochs
        self.update_latest = getattr(ckpt, "update_latest", False)
        self._last_dcp_step: int = -1
        self._last_hf_step: int = -1

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.trainer.save_model_assets()
        self.trainer.load()
        helper.empty_cache()

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.save_hf_weights:
            if state.global_step != self._last_hf_step:
                self._save_hf(state, stage="train_end")
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at train_end (global_step {state.global_step} "
                    f"already saved)."
                )
                self.trainer.wait_for_pending_save()
        else:
            self.trainer.wait_for_pending_save()

    def on_step_end(self, state: TrainerState, **kwargs):
        if self.dcp_every_n_steps and state.global_step % self.dcp_every_n_steps == 0:
            self._save_dcp(state)
        if self.save_hf_weights and self.hf_every_n_steps and state.global_step % self.hf_every_n_steps == 0:
            self._save_hf(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0:
            if state.global_step != self._last_dcp_step:
                self._save_dcp(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate checkpoint save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )
        if self.save_hf_weights and self.hf_every_n_epochs and (state.epoch + 1) % self.hf_every_n_epochs == 0:
            if state.global_step != self._last_hf_step:
                self._save_hf(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )

    def _save_dcp(self, state: TrainerState):
        self.trainer.save_dcp(state)
        self._last_dcp_step = state.global_step

    def _save_hf(self, state: TrainerState, stage: str = "step_end"):
        self.trainer.save_hf_or_lora(state, stage=stage)
        if self.update_latest and self.trainer.args.train.global_rank == 0:
            self._update_latest_hf(state)
        self._last_hf_step = state.global_step

    def _update_latest_hf(self, state: TrainerState) -> None:
        # LoRA exports live in lora_ckpt, not hf_ckpt. Never advertise an adapter
        # as a standalone HF model (including a stale HF export from another run).
        if self.trainer.args.model.lora_config:
            return
        ckpt = self.trainer.args.train.checkpoint
        target = Path(layout.hf_export_dir(layout.step_dir(ckpt.save_path, state.global_step))).absolute()
        if not target.is_dir():
            raise FileNotFoundError(f"Successful HF export directory is missing: {target}")
        alias_dir = Path(ckpt.output_dir).absolute() / layout.HF_EXPORT_DIRNAME
        alias_dir.mkdir(parents=True, exist_ok=True)
        latest = alias_dir / "latest"
        if latest.exists() and not latest.is_symlink():
            raise FileExistsError(f"Refusing to replace a non-symlink checkpoint alias: {latest}")
        # A relative target survives moving the whole output tree. A temporary
        # sibling and replace keep readers on the previous export until success,
        # including when latest is an existing dangling symlink.
        with tempfile.TemporaryDirectory(prefix=".latest-", dir=alias_dir) as temporary:
            pending = Path(temporary) / "latest"
            pending.symlink_to(os.path.relpath(target, alias_dir), target_is_directory=True)
            os.replace(pending, latest)


__all__ = ["CheckpointCallback"]
