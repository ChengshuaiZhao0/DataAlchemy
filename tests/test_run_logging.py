"""Tests for src.utils.run_logging: header, footer, metrics callback."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

import src.utils.logging as lg_mod
from src.utils.config import Config
from src.utils.logging import get_logger, setup_run_log
from src.utils.run_logging import (
    MetricsLoggingCallback,
    log_run_footer,
    log_run_header,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.delenv("DATAALCHEMY_RUN_LOG", raising=False)
    prev_path = lg_mod._RUN_LOG_PATH
    prev_stage = lg_mod._CURRENT_STAGE
    lg_mod._RUN_LOG_PATH = None
    yield
    lg_mod._RUN_LOG_PATH = prev_path
    lg_mod._CURRENT_STAGE = prev_stage
    for lg in list(logging.Logger.manager.loggerDict.values()):
        if not isinstance(lg, logging.Logger):
            continue
        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler):
                lg.removeHandler(h)
                h.close()


def _make_cfg() -> Config:
    cfg = Config()
    cfg.update(
        {
            "system": {"seed": 42, "experiment": "tests"},
            "model": {"arch": "GPT"},
        }
    )
    return cfg


def test_log_run_header_emits_banner_and_returns_info(tmp_path, monkeypatch) -> None:
    log_path = str(tmp_path / "run.log")
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", log_path)
    setup_run_log(stage="Pretrain", cfg=_make_cfg())
    logger = get_logger("src.tests.run_logging.header")

    info = log_run_header(logger, _make_cfg(), str(tmp_path))
    for h in logger.handlers:
        h.flush()

    assert "config" in info and info["config"]["system"]["seed"] == 42
    assert "env" in info
    assert "_start_monotonic" in info

    contents = open(log_path).read()
    assert "===== RUN HEADER =====" in contents
    assert "config_hash" in contents
    assert "resolved config" in contents


def test_log_run_footer_emits_duration_and_summary(tmp_path, monkeypatch) -> None:
    log_path = str(tmp_path / "run.log")
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", log_path)
    setup_run_log(stage="Pretrain", cfg=_make_cfg())
    logger = get_logger("src.tests.run_logging.footer")

    info = log_run_header(logger, _make_cfg(), str(tmp_path))
    closed = log_run_footer(logger, info, extra={"final_train_loss": 0.25})
    for h in logger.handlers:
        h.flush()

    assert closed["summary"]["final_train_loss"] == 0.25
    assert "_start_monotonic" not in closed
    contents = open(log_path).read()
    assert "===== RUN FOOTER =====" in contents
    assert "final_train_loss" in contents


def test_metrics_callback_emits_one_line_per_event(tmp_path, monkeypatch) -> None:
    log_path = str(tmp_path / "run.log")
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", log_path)
    setup_run_log(stage="Pretrain", cfg=_make_cfg())
    logger = get_logger("src.tests.run_logging.metrics")
    cb = MetricsLoggingCallback(logger=logger)

    state = SimpleNamespace(global_step=10, epoch=0.5)
    cb.on_log(args=None, state=state, control=None, logs={"loss": 1.234, "learning_rate": 3e-3})
    state = SimpleNamespace(global_step=20, epoch=1.0)
    cb.on_log(args=None, state=state, control=None, logs={"eval_loss": 0.5})
    for h in logger.handlers:
        h.flush()

    lines = [ln for ln in open(log_path).read().splitlines() if "step=" in ln]
    assert len(lines) == 2
    assert "step=10" in lines[0] and "split=train" in lines[0]
    assert "step=20" in lines[1] and "split=eval" in lines[1]


def test_src_star_logger_captured_by_setup_run_log(tmp_path, monkeypatch) -> None:
    log_path = str(tmp_path / "run.log")
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", log_path)
    child = get_logger("src.training.pretrain_testproxy")
    setup_run_log(stage="Pretrain", cfg=_make_cfg())

    child.info("src-child-message")
    for h in child.handlers:
        h.flush()
    assert os.path.exists(log_path)
    assert "src-child-message" in open(log_path).read()
