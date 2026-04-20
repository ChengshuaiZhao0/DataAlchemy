"""Shared ``TrainingArguments`` builder + post-train summary extraction.

Used by both ``src.training.pretrain`` and ``src.training.sft`` so that
hyperparameter defaults stay in a single place.
"""

from __future__ import annotations

import os
from typing import Any

from transformers import Trainer, TrainingArguments

from src.utils.config import Config


def _require(cfg_section: Any, key: str) -> Any:
    value = getattr(cfg_section, key, None)
    if value is None:
        raise ValueError(
            f"Missing required config key 'train.{key}'. "
            "Populate it in your YAML (see configs/_base/pretrain.yaml)."
        )
    return value


def build_training_args(
    cfg: Config,
    output_dir: str,
    has_eval: bool = True,
) -> TrainingArguments:
    """Construct ``TrainingArguments`` from ``cfg.train`` + ``cfg.system``.

    Required ``cfg.train`` keys: ``num_epochs``, ``per_device_batch_size``.
    ``per_device_eval_batch_size`` defaults to ``per_device_batch_size``.

    ``has_eval`` controls eval/save behavior (loading the best model only makes
    sense when an eval split exists). When ``has_eval=False``, the following
    ``cfg.train`` keys are passed through to ``TrainingArguments`` but ignored
    by ``Trainer`` (since ``eval_strategy="no"`` short-circuits eval):

    - ``per_device_eval_batch_size``
    - ``eval_steps``
    - ``early_stopping_patience`` (also: ``EarlyStoppingCallback`` is not
      registered upstream, see ``src.training._shared.build_callbacks``)

    Mixed precision: only ``cfg.train.bf16`` is honored. ``fp16`` was removed
    after we standardized on bf16-capable hardware.
    """
    train = cfg.train
    num_epochs = int(_require(train, "num_epochs"))
    batch_size = int(_require(train, "per_device_batch_size"))
    eval_batch_size = int(getattr(train, "per_device_eval_batch_size", batch_size))
    use_bf16 = bool(getattr(train, "bf16", False))
    save_steps = int(getattr(train, "save_steps", 1000))
    eval_steps = int(getattr(train, "eval_steps", save_steps))

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=int(getattr(train, "logging_steps", 200)),
        prediction_loss_only=True,
        learning_rate=float(getattr(train, "lr", 3e-3)),
        lr_scheduler_type=str(getattr(train, "lr_scheduler_type", "cosine")),
        warmup_ratio=float(getattr(train, "warmup_ratio", 0.1)),
        weight_decay=float(getattr(train, "weight_decay", 0.01)),
        max_grad_norm=float(getattr(train, "max_grad_norm", 1.0)),
        bf16=use_bf16,
        gradient_accumulation_steps=int(
            getattr(train, "gradient_accumulation_steps", 1)
        ),
        gradient_checkpointing=bool(
            getattr(train, "gradient_checkpointing", False)
        ),
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        seed=int(getattr(cfg.system, "seed", 42)),
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs"),
    )


def summarize_trainer(trainer: Trainer) -> dict[str, Any]:
    """Pluck the final train/eval losses + best metric from a finished Trainer."""
    summary: dict[str, Any] = {}
    if not trainer.state.log_history:
        return summary
    last_train = next(
        (h for h in reversed(trainer.state.log_history) if "loss" in h), None
    )
    last_eval = next(
        (h for h in reversed(trainer.state.log_history) if "eval_loss" in h), None
    )
    if last_train:
        summary["final_train_loss"] = last_train.get("loss")
    if last_eval:
        summary["final_eval_loss"] = last_eval.get("eval_loss")
    if trainer.state.best_metric is not None:
        summary["best_metric"] = float(trainer.state.best_metric)
    return summary
