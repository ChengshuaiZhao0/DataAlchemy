"""Structured run header/footer and trainer-metrics logging.

Everything lands inside the unified per-run log (see :mod:`src.utils.logging`).
No sidecar files (``run_info.json`` / ``resolved_config.yaml`` /
``metrics.jsonl``) are written — each run is a single log file that contains
the header block, chronological INFO lines, per-step trainer metrics, and a
footer block with duration + summary.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import yaml
from transformers import TrainerCallback

from src.utils.config import Config


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        else:
            env["cuda_devices"] = []
    except Exception:
        env["torch"] = None
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:
        env["transformers"] = None
    try:
        import datasets

        env["datasets"] = datasets.__version__
    except Exception:
        env["datasets"] = None
    return env


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _cfg_to_dict(cfg: Any) -> Any:
    if isinstance(cfg, Config):
        return cfg.dict()
    if isinstance(cfg, dict):
        return {k: _cfg_to_dict(v) for k, v in cfg.items()}
    return cfg


def _header_body_lines(info: dict[str, Any], cfg_yaml: str) -> list[str]:
    env = info.get("env", {})
    lines = [
        "===== RUN HEADER =====",
        f"run_dir        : {info.get('run_dir')}",
        f"argv           : {' '.join(info.get('argv', []))}",
        f"git_sha        : {env.get('git_sha')}",
        f"python         : {env.get('python')}",
        f"torch          : {env.get('torch')}",
        f"cuda_available : {env.get('cuda_available')}",
        f"cuda_devices   : {env.get('cuda_devices')}",
        f"transformers   : {env.get('transformers')}",
        f"datasets       : {env.get('datasets')}",
        f"config_hash    : {info.get('config_hash')}",
        f"start_time     : {info.get('start_time')}",
        "--- resolved config ---",
    ]
    lines.extend(cfg_yaml.rstrip().splitlines())
    lines.append("===== / RUN HEADER =====")
    return lines


def _emit_block(logger: logging.Logger, lines: list[str]) -> None:
    for line in lines:
        logger.info(line)


def log_run_header(
    logger: logging.Logger, cfg: Any, run_dir: str
) -> dict[str, Any]:
    """Emit the run-header block into the unified log.

    Returns an ``info`` dict carrying ``_start_monotonic`` for :func:`log_run_footer`.
    ``run_dir`` is retained in the info dict for reference but no sidecar file
    is written.
    """
    cfg_dict = _cfg_to_dict(cfg)
    cfg_yaml = yaml.safe_dump(cfg_dict, sort_keys=True, default_flow_style=False)
    config_hash = hashlib.sha256(cfg_yaml.encode("utf-8")).hexdigest()[:16]

    info: dict[str, Any] = {
        "run_dir": os.path.abspath(run_dir),
        "argv": sys.argv,
        "config": cfg_dict,
        "config_hash": config_hash,
        "env": _collect_env(),
        "start_time": _now_iso(),
        "_start_monotonic": time.monotonic(),
    }
    _emit_block(logger, _header_body_lines(info, cfg_yaml))
    return info


def log_run_footer(
    logger: logging.Logger,
    info: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit the run-footer block with duration + summary."""
    info = dict(info or {})
    start_mono = info.pop("_start_monotonic", None)
    info["end_time"] = _now_iso()
    if start_mono is not None:
        info["duration_seconds"] = round(time.monotonic() - float(start_mono), 3)
    if extra:
        info.setdefault("summary", {}).update(extra)

    summary = info.get("summary") or {}
    lines = [
        "===== RUN FOOTER =====",
        f"end_time         : {info['end_time']}",
        f"duration_seconds : {info.get('duration_seconds')}",
    ]
    if summary:
        lines.append("--- summary ---")
        for k, v in summary.items():
            lines.append(f"{k:<16} : {v}")
    lines.append("===== / RUN FOOTER =====")
    _emit_block(logger, lines)
    return info


class MetricsLoggingCallback(TrainerCallback):
    """Emit every Trainer ``on_log`` event into the unified run log.

    One line per logging event, formatted as::

        step=200 epoch=200.00 split=train loss=2.3389 lr=5.97e-04 grad_norm=0.496

    No JSONL sidecar file is written; a future analysis script can parse these
    lines from the run log directly.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("src.utils.run_logging")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        split = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        parts = [
            f"step={int(state.global_step)}",
            f"epoch={state.epoch:.2f}" if state.epoch is not None else "epoch=None",
            f"split={split}",
        ]
        for k, v in logs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4g}")
            elif isinstance(v, (int, bool)) or v is None:
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={v}")
        self.logger.info(" ".join(parts))
