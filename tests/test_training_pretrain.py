"""Slow integration test: 1 epoch of pretraining on a toy corpus."""

from __future__ import annotations

import os

import pytest

from src.models import ARCH_GPT
from src.training.pretrain import pretrain
from src.utils.config import Config

pytestmark = pytest.mark.slow


def _build_cfg() -> Config:
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


def test_pretrain_writes_run_log_and_checkpoint(tmp_path, toy_rule_files, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    tok_dir = tmp_path / "tok"
    run_log = tmp_path / "run.log"
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", str(run_log))

    import src.utils.logging as lg_mod
    lg_mod._RUN_LOG_PATH = None
    from src.utils.logging import setup_run_log
    setup_run_log(stage="Pretrain", cfg=_build_cfg())

    cfg = _build_cfg()

    pretrain(
        data_files=list(toy_rule_files.values()),
        tokenizer_dir=str(tok_dir),
        model_dir=str(model_dir),
        cfg=cfg,
    )

    assert run_log.exists() and run_log.stat().st_size > 0
    contents = run_log.read_text()
    assert "===== RUN HEADER =====" in contents
    assert "===== RUN FOOTER =====" in contents
    assert "[Pretrain]" in contents
    assert os.path.isdir(model_dir / "final")
    from src.models import load_tokenizer_and_model
    tok, model = load_tokenizer_and_model(str(model_dir / "final"), ARCH_GPT)
    assert tok.pad_token is not None
    assert model.config.n_embd == 32
