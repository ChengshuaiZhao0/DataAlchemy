"""Shared helpers for the pretrain and sft training loops.

Both loops follow the same skeleton — load JSONL → shuffle → optional
``train_ratio`` subsample → tokenize → optional ``validation_ratio`` split.
These utilities keep that skeleton identical across the two entry points.
"""

from __future__ import annotations

from transformers import EarlyStoppingCallback, TrainerCallback

from src.utils.config import Config
from src.utils.logging import get_logger
from src.utils.run_logging import MetricsLoggingCallback

logger = get_logger(__name__)


def positions_cap(model) -> int:
    """Return the model's max sequence length (GPT: n_positions, LLaMA: max_position_embeddings)."""
    cfg = model.config
    n = getattr(cfg, "n_positions", None) or getattr(
        cfg, "max_position_embeddings", None
    )
    if n is None:
        raise ValueError(
            "Model config exposes neither n_positions nor max_position_embeddings."
        )
    return int(n)


def load_and_subsample(data_files: list[str], seed: int, train_ratio: float):
    """Load JSONL files, shuffle, and optionally keep the first ``train_ratio`` fraction."""
    from datasets import load_dataset

    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"dataset.train_ratio must be in (0, 1]; got {train_ratio}")

    dataset = load_dataset("json", data_files={"train": data_files})["train"].shuffle(
        seed=seed
    )
    if train_ratio < 1.0:
        dataset = dataset.select(range(int(len(dataset) * train_ratio)))
        logger.info(
            "Subsampled training set to %d rows (train_ratio=%g)",
            len(dataset),
            train_ratio,
        )
    return dataset


def maybe_split(tokenized, seed: int, val_ratio: float):
    """Split ``tokenized`` into (train, eval); return (train, None) when val_ratio is out of (0, 1)."""
    if 0 < val_ratio < 1:
        split = tokenized.train_test_split(test_size=val_ratio, seed=seed)
        return split["train"], split["test"]
    return tokenized, None


def build_callbacks(
    cfg: Config, has_eval: bool
) -> list[TrainerCallback]:
    """Return the common callback set: metrics logger plus optional early stopping."""
    callbacks: list[TrainerCallback] = [MetricsLoggingCallback(logger=logger)]
    patience = int(getattr(cfg.train, "early_stopping_patience", 3))
    if has_eval and patience > 0:
        threshold = float(getattr(cfg.train, "early_stopping_threshold", 0.0005))
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            )
        )
    return callbacks
