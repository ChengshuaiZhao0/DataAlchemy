"""Shared CLI bootstrap for the four scripts under ``scripts/``.

Each script previously duplicated argparse + ``load_config`` + ``set_seed`` +
device selection. ``parse_and_setup`` collapses those into one call.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import torch

from src.utils.config import Config, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("dataalchemy.cli")


def parse_and_setup(
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> tuple[argparse.Namespace, Config, torch.device]:
    """Parse ``--config`` + dotted overrides, seed RNGs, return (args, cfg, device).

    ``add_args`` may register extra flags on the parser before parsing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    if add_args is not None:
        add_args(parser)
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides=overrides)
    logger.info("Config:\n%s", cfg.dict())

    set_seed(
        int(getattr(cfg.system, "seed", 42)),
        deterministic=bool(getattr(cfg.system, "deterministic", True)),
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and bool(getattr(cfg.system, "use_cuda", True))
        else "cpu"
    )
    logger.info("Using device %s", device)
    return args, cfg, device


def default_num_proc(cap: int = 8) -> int:
    """Sensible default for ``datasets.map(num_proc=...)``."""
    return min(cap, os.cpu_count() or cap)
