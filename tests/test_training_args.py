"""Unit tests for ``src.training.args.build_training_args``.

Pure-CPU; do not load a model or instantiate a Trainer. We only check that
the returned ``TrainingArguments`` reflect the config knobs we expose.
"""

from __future__ import annotations

import pytest

from src.training.args import build_training_args
from src.utils.config import Config


def _base_cfg() -> Config:
    cfg = Config()
    cfg.update(
        {
            "system": {"seed": 7},
            "train": {
                "num_epochs": 2,
                "per_device_batch_size": 8,
                "per_device_eval_batch_size": 16,
                "lr": 1e-3,
                "save_steps": 50,
                "logging_steps": 10,
            },
        }
    )
    return cfg


def test_defaults_match_documented_values(tmp_path) -> None:
    cfg = _base_cfg()
    args = build_training_args(cfg, str(tmp_path), has_eval=False)

    assert args.warmup_ratio == pytest.approx(0.1)
    assert args.weight_decay == pytest.approx(0.01)
    assert args.max_grad_norm == pytest.approx(1.0)
    assert args.lr_scheduler_type.value == "cosine"
    assert args.gradient_checkpointing is False
    assert args.seed == 7


def test_train_overrides_flow_through(tmp_path) -> None:
    cfg = _base_cfg()
    cfg.train.warmup_ratio = 0.25
    cfg.train.weight_decay = 0.05
    cfg.train.max_grad_norm = 0.5
    cfg.train.lr_scheduler_type = "linear"
    cfg.train.gradient_checkpointing = True

    args = build_training_args(cfg, str(tmp_path), has_eval=False)

    assert args.warmup_ratio == pytest.approx(0.25)
    assert args.weight_decay == pytest.approx(0.05)
    assert args.max_grad_norm == pytest.approx(0.5)
    assert args.lr_scheduler_type.value == "linear"
    assert args.gradient_checkpointing is True


def test_has_eval_false_disables_eval_and_best_model(tmp_path) -> None:
    cfg = _base_cfg()
    args = build_training_args(cfg, str(tmp_path), has_eval=False)

    assert args.eval_strategy.value == "no"
    assert args.load_best_model_at_end is False
    assert args.metric_for_best_model is None


def test_has_eval_true_enables_steps_eval_and_best_model(tmp_path) -> None:
    cfg = _base_cfg()
    args = build_training_args(cfg, str(tmp_path), has_eval=True)

    assert args.eval_strategy.value == "steps"
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False


def test_bf16_toggle(tmp_path) -> None:
    cfg = _base_cfg()
    args_off = build_training_args(cfg, str(tmp_path), has_eval=False)
    assert args_off.bf16 is False

    cfg.train.bf16 = True
    args_on = build_training_args(cfg, str(tmp_path), has_eval=False)
    assert args_on.bf16 is True


def test_seed_propagates_from_system(tmp_path) -> None:
    cfg = _base_cfg()
    cfg.system.seed = 1234
    args = build_training_args(cfg, str(tmp_path), has_eval=True)
    assert args.seed == 1234


def test_eval_steps_defaults_to_save_steps_when_omitted(tmp_path) -> None:
    cfg = _base_cfg()
    args = build_training_args(cfg, str(tmp_path), has_eval=True)
    assert args.eval_steps == cfg.train.save_steps

    cfg.train.eval_steps = 25
    args2 = build_training_args(cfg, str(tmp_path), has_eval=True)
    assert args2.eval_steps == 25
