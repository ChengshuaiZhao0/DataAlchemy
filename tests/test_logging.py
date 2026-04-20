"""Tests for the unified per-run logging helpers."""

from __future__ import annotations

import logging
import os

import pytest

import src.utils.logging as lg_mod
from src.utils.logging import get_logger, setup_run_log


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Isolate each test from module-global log path / stage."""
    prev_path = lg_mod._RUN_LOG_PATH
    prev_stage = lg_mod._CURRENT_STAGE
    monkeypatch.delenv("DATAALCHEMY_RUN_LOG", raising=False)
    lg_mod._RUN_LOG_PATH = None
    yield
    lg_mod._RUN_LOG_PATH = prev_path
    lg_mod._CURRENT_STAGE = prev_stage
    for name in (
        "dataalchemy",
        "dataalchemy.test.logging",
        "src.logging_test",
        "transformers",
        "datasets",
        "accelerate",
        "py.warnings",
    ):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler):
                lg.removeHandler(h)
                h.close()


def test_get_logger_is_idempotent() -> None:
    lg1 = get_logger("dataalchemy.test.logging")
    count_before = len(lg1.handlers)
    lg2 = get_logger("dataalchemy.test.logging")
    assert lg1 is lg2
    assert len(lg2.handlers) == count_before


def test_setup_run_log_writes_to_env_path(tmp_path, monkeypatch) -> None:
    log_path = str(tmp_path / "run.log")
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", log_path)
    setup_run_log(stage="Pretrain")
    lg = get_logger("src.logging_test")
    lg.info("hello-from-test")
    for h in lg.handlers:
        h.flush()
    assert os.path.exists(log_path)
    contents = open(log_path).read()
    assert "hello-from-test" in contents
    assert "[Pretrain]" in contents


def test_setup_run_log_format_has_pipes(tmp_path, monkeypatch) -> None:
    log_path = str(tmp_path / "run.log")
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", log_path)
    setup_run_log(stage="Evaluate")
    lg = get_logger("src.logging_test")
    lg.warning("watch out")
    for h in lg.handlers:
        h.flush()
    line = [ln for ln in open(log_path).read().splitlines() if "watch out" in ln][0]
    # Format: "YYYY-MM-DD HH:MM:SS | LEVEL | [stage] message"
    assert " | WARNING | [Evaluate] watch out" in line


def test_setup_run_log_does_not_duplicate_handlers(tmp_path, monkeypatch) -> None:
    log_path = str(tmp_path / "run.log")
    monkeypatch.setenv("DATAALCHEMY_RUN_LOG", log_path)
    setup_run_log(stage="Pretrain")
    setup_run_log(stage="Pretrain")
    lg = logging.getLogger("dataalchemy")
    file_handlers = [
        h for h in lg.handlers
        if isinstance(h, logging.FileHandler)
        and h.baseFilename == log_path
    ]
    assert len(file_handlers) == 1
