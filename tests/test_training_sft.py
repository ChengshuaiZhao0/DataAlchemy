"""Slow integration test: chained pretrain -> SFT on the toy corpus."""

from __future__ import annotations

import os

import pytest

from src.models import ARCH_GPT
from src.training.pretrain import pretrain
from src.training.sft import sft
from src.utils.config import Config

pytestmark = pytest.mark.slow


def _pretrain_cfg() -> Config:
    cfg = Config()
    cfg.update(
        {
            "model": {
                "arch": ARCH_GPT,
                "hidden_size": 32,
                "intermediate_size": 64,
                "n_layer": 1,
                "n_head": 2,
                "n_positions": 64,
            },
            "dataset": {},
            "tokenizer": {"vocab_size": 512, "num_proc": 1},
            "system": {"seed": 42},
            "train": {
                "num_epochs": 1,
                "per_device_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "save_steps": 1000,
                "eval_steps": 1000,
                "lr": 3e-3,
                "fp16": False,
                "bf16": False,
                "validation_ratio": 0.1,
                "early_stopping_patience": 0,
            },
        }
    )
    return cfg


def _sft_cfg() -> Config:
    cfg = Config()
    cfg.update(
        {
            "model": {"arch": ARCH_GPT},
            "dataset": {"train_ratio": 0.9},
            "tokenizer": {"num_proc": 1},
            "system": {"seed": 42},
            "train": {
                "num_epochs": 1,
                "per_device_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "save_steps": 1000,
                "eval_steps": 1000,
                "lr": 3e-3,
                "fp16": False,
                "validation_ratio": 0.1,
                "early_stopping_patience": 0,
            },
        }
    )
    return cfg


def test_sft_after_pretrain(tmp_path, toy_rule_files, monkeypatch) -> None:
    pre_cfg = _pretrain_cfg()
    run_log = tmp_path / "run.log"
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", str(run_log))

    import src.utils.logging as lg_mod
    lg_mod._RUN_LOG_PATH = None
    from src.utils.logging import setup_run_log
    setup_run_log(stage="Pretrain", cfg=pre_cfg)

    pretrained_dir = tmp_path / "pre"
    tok_dir = tmp_path / "tok"
    pretrain(
        data_files=list(toy_rule_files.values()),
        tokenizer_dir=str(tok_dir),
        model_dir=str(pretrained_dir),
        cfg=pre_cfg,
    )

    setup_run_log(stage="SFT", cfg=_sft_cfg())
    sft_dir = tmp_path / "sft"
    sft(
        data_files=[toy_rule_files["F1F2"]],
        pretrained_model_dir=str(pretrained_dir / "final"),
        model_dir=str(sft_dir),
        cfg=_sft_cfg(),
    )

    assert run_log.exists() and run_log.stat().st_size > 0
    contents = run_log.read_text()
    assert "[Pretrain]" in contents
    assert "[SFT]" in contents
    assert os.path.isdir(sft_dir / "final")
