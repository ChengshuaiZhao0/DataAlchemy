"""Unified per-run logging.

One file per run at ``logs/{experiment}/run_{YYYYmmdd_HHMMSS}.log`` (overridable
by the ``DATAALCHEMY_RUN_LOG`` env var). Every line carries a stage tag in the
format ``<time> | <LEVEL> | [<stage>] <message>``.

Entry-point scripts call :func:`setup_run_log` once at the top of ``main()``;
thereafter any ``logging.getLogger(...)`` call in the project (or in transformers/
datasets/accelerate) flows through the shared file + stream handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

_FORMAT = "%(asctime)s | %(levelname)s | [%(stage)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CAPTURED_LOGGERS = ("dataalchemy", "transformers", "datasets", "accelerate")
_SRC_PREFIXES = ("src",)

_CURRENT_STAGE: str = "main"
_RUN_LOG_PATH: str | None = None


class _StageFilter(logging.Filter):
    """Inject the current stage tag as a LogRecord attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.stage = _CURRENT_STAGE
        return True


def _formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT, datefmt=_DATEFMT)


def get_logger(
    name: str = "dataalchemy",
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a project logger with a shared stdout handler and stage filter.

    The returned logger is attached to the file handler lazily by
    :func:`setup_run_log`. Calling ``get_logger`` multiple times with the same
    name is idempotent.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_dataalchemy_setup", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(_formatter())
    stream.addFilter(_StageFilter())
    logger.addHandler(stream)

    logger._dataalchemy_setup = True  # type: ignore[attr-defined]
    if _RUN_LOG_PATH is not None:
        _attach_file_handler(logger, _RUN_LOG_PATH)
    return logger


def _attach_file_handler(logger: logging.Logger, path: str) -> None:
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == path:
            return
    fh = logging.FileHandler(path, mode="a")
    fh.setFormatter(_formatter())
    fh.addFilter(_StageFilter())
    logger.addHandler(fh)


def _target_logger_names() -> set[str]:
    names = set(_CAPTURED_LOGGERS)
    for existing in list(logging.Logger.manager.loggerDict.keys()):
        for prefix in _SRC_PREFIXES:
            if existing == prefix or existing.startswith(prefix + "."):
                names.add(existing)
    return names


def _resolve_run_log_path(cfg) -> str:
    env_path = os.environ.get("DATAALCHEMY_RUN_LOG")
    if env_path:
        return env_path
    experiment = "run"
    if cfg is not None:
        experiment = str(getattr(getattr(cfg, "system", object()), "experiment", "run"))
    log_dir = os.path.join("logs", experiment)
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"run_{ts}.log")


def setup_run_log(stage: str, cfg=None) -> str:
    """Bootstrap the per-run log at the top of an entry-point ``main()``.

    Sets the stage tag, resolves the log path (env var or auto-generated),
    and attaches a shared file handler to every project-relevant logger.
    Returns the resolved log path.
    """
    global _CURRENT_STAGE, _RUN_LOG_PATH
    _CURRENT_STAGE = stage
    if _RUN_LOG_PATH is None:
        _RUN_LOG_PATH = _resolve_run_log_path(cfg)
        os.makedirs(os.path.dirname(_RUN_LOG_PATH) or ".", exist_ok=True)

    for name in _target_logger_names():
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        has_stage_filter = any(
            isinstance(f, _StageFilter) for h in logger.handlers for f in h.filters
        )
        if not has_stage_filter:
            for h in logger.handlers:
                h.addFilter(_StageFilter())
        _attach_file_handler(logger, _RUN_LOG_PATH)

    logging.captureWarnings(True)
    py_warnings = logging.getLogger("py.warnings")
    if py_warnings.level == logging.NOTSET:
        py_warnings.setLevel(logging.WARNING)
    _attach_file_handler(py_warnings, _RUN_LOG_PATH)
    for h in py_warnings.handlers:
        h.addFilter(_StageFilter())

    get_logger("dataalchemy").info("===== STAGE START: %s =====", stage)
    return _RUN_LOG_PATH


def current_run_log_path() -> str | None:
    return _RUN_LOG_PATH


def set_stage(stage: str) -> None:
    """Change the current stage tag mid-process (rare; most entry points call
    :func:`setup_run_log` instead)."""
    global _CURRENT_STAGE
    _CURRENT_STAGE = stage


# Initialize the root project logger so module-level ``logger = get_logger()``
# imports work even before an entry point calls setup_run_log().
logger = get_logger()
