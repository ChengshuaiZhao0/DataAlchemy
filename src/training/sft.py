"""SFT loop reusing the symbolic tokenizer and a pretrained DataAlchemy model.

Supervision is applied only to the completion (``<think>``/``<answer>`` onwards).
Dataset-selection flow mirrors :mod:`src.training.pretrain`:
``load → shuffle → train_ratio subsample → tokenize → validation_ratio split``.
"""

from __future__ import annotations

import os

from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer

from src.models import load_tokenizer_and_model
from src.training._shared import (
    build_callbacks,
    load_and_subsample,
    maybe_split,
    positions_cap,
)
from src.training.args import build_training_args, summarize_trainer
from src.utils.cli import default_num_proc
from src.utils.config import Config
from src.utils.logging import get_logger
from src.utils.run_logging import log_run_footer, log_run_header

logger = get_logger(__name__)


def _tokenize_sft(ds: Dataset, tokenizer, num_proc: int, desc: str) -> Dataset:
    def tokenize_fn(examples):
        input_ids, labels = [], []
        for prompt, completion in zip(
            examples["input"], examples["output"], strict=True
        ):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(
                completion + tokenizer.eos_token, add_special_tokens=False
            )["input_ids"]
            input_ids.append(prompt_ids + completion_ids)
            labels.append([-100] * len(prompt_ids) + completion_ids)
        return {"input_ids": input_ids, "labels": labels}

    return ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc=desc,
    )


def sft(
    data_files: list[str],
    pretrained_model_dir: str,
    model_dir: str,
    cfg: Config,
    log_dir: str | None = None,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    log_dir = log_dir or os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_info = log_run_header(logger, cfg, log_dir)
    data_files = sorted(data_files)
    seed = int(getattr(cfg.system, "seed", 42))

    tokenizer, model = load_tokenizer_and_model(
        pretrained_model_dir,
        cfg.model.arch,
        attn_implementation=str(getattr(cfg.model, "attn_implementation", "sdpa")),
    )
    model.train()
    logger.info("Loaded SFT base model from %s", pretrained_model_dir)

    train_ratio = float(getattr(cfg.dataset, "train_ratio", 1.0))
    dataset = load_and_subsample(data_files, seed=seed, train_ratio=train_ratio)

    tok_cfg = getattr(cfg, "tokenizer", Config())
    num_proc = int(getattr(tok_cfg, "num_proc", default_num_proc()))
    tokenized = _tokenize_sft(dataset, tokenizer, num_proc, "Tokenizing SFT")

    cap = positions_cap(model)
    max_len = max((len(x) for x in tokenized["input_ids"]), default=0)
    logger.info("SFT tokenized: n=%d max_len=%d cap=%d", len(tokenized), max_len, cap)
    if max_len > cap:
        raise ValueError(
            f"SFT sample length ({max_len}) exceeds model cap ({cap})."
        )

    val_ratio = float(getattr(cfg.train, "validation_ratio", 0.0))
    train_tokenized, eval_tokenized = maybe_split(tokenized, seed=seed, val_ratio=val_ratio)

    collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors="pt"
    )

    training_args = build_training_args(
        cfg,
        model_dir,
        has_eval=eval_tokenized is not None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=collator,
        callbacks=build_callbacks(cfg, has_eval=eval_tokenized is not None),
    )
    trainer.train()
    trainer.save_model(os.path.join(model_dir, "final"))
    tokenizer.save_pretrained(os.path.join(model_dir, "final"))

    log_run_footer(logger, run_info, extra=summarize_trainer(trainer))
