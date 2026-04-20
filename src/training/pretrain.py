"""From-scratch pretraining loop."""

from __future__ import annotations

import os

from transformers import DataCollatorForLanguageModeling, Trainer

from src.models import (
    ARCH_LLAMA,
    LLAMA_DEFAULT_FFN_RATIO,
    ModelSpec,
    build_model,
    build_tokenizer,
    num_trainable_parameters,
)
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


def _model_spec_from_config(cfg: Config) -> ModelSpec:
    m = cfg.model
    hidden_size = int(m.hidden_size)
    if m.arch == ARCH_LLAMA:
        intermediate_size = int(
            getattr(m, "intermediate_size", LLAMA_DEFAULT_FFN_RATIO * hidden_size)
        )
    else:
        # GPT2Config ignores intermediate_size; ModelSpec's default is unused.
        intermediate_size = int(getattr(m, "intermediate_size", 0))
    return ModelSpec(
        arch=m.arch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_layer=int(m.n_layer),
        n_head=int(m.n_head),
        n_positions=int(getattr(m, "n_positions", 256)),
        rope_theta=float(getattr(m, "rope_theta", 10000.0)),
        attn_implementation=str(getattr(m, "attn_implementation", "sdpa")),
    )


def pretrain(
    data_files: list[str],
    tokenizer_dir: str,
    model_dir: str,
    cfg: Config,
    log_dir: str | None = None,
) -> None:
    """Train a from-scratch causal LM on ``data_files``."""
    os.makedirs(model_dir, exist_ok=True)
    log_dir = log_dir or os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_info = log_run_header(logger, cfg, log_dir)

    data_files = sorted(data_files)
    seed = int(getattr(cfg.system, "seed", 42))
    train_ratio = float(getattr(cfg.dataset, "train_ratio", 1.0))
    dataset = load_and_subsample(data_files, seed=seed, train_ratio=train_ratio)

    tok_cfg = getattr(cfg, "tokenizer", Config())
    tokenizer = build_tokenizer(
        tokenizer_dir,
        cfg.model.arch,
        corpus_files=data_files,
        vocab_size=int(getattr(tok_cfg, "vocab_size", 64)),
        n_positions=int(getattr(cfg.model, "n_positions", 256)),
    )

    model = build_model(_model_spec_from_config(cfg), tokenizer)
    logger.info(
        "Model %s built with %.2fK trainable params",
        cfg.model.arch,
        num_trainable_parameters(model) / 1e3,
    )

    cap = positions_cap(model)

    def tokenize_fn(examples):
        texts = [
            f"{i} {o}{tokenizer.eos_token}"
            for i, o in zip(examples["input"], examples["output"], strict=True)
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=cap,
            padding=False,
            add_special_tokens=False,
        )

    num_proc = int(getattr(tok_cfg, "num_proc", default_num_proc()))
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    lengths = [len(ids) for ids in tokenized["input_ids"]]
    n_truncated = sum(1 for n in lengths if n >= cap)
    logger.info(
        "Tokenized %d samples; max_len=%d (cap=%d); %d hit cap",
        len(lengths),
        max(lengths) if lengths else 0,
        cap,
        n_truncated,
    )

    val_ratio = float(getattr(cfg.train, "validation_ratio", 0.0))
    train_dataset, eval_dataset = maybe_split(tokenized, seed=seed, val_ratio=val_ratio)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = build_training_args(
        cfg,
        model_dir,
        has_eval=eval_dataset is not None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=build_callbacks(cfg, has_eval=eval_dataset is not None),
    )
    trainer.train()
    trainer.save_model(os.path.join(model_dir, "final"))
    tokenizer.save_pretrained(os.path.join(model_dir, "final"))

    log_run_footer(logger, run_info, extra=summarize_trainer(trainer))
