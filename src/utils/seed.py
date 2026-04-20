"""Seeding for reproducibility across python, numpy, and torch.

Delegates RNG seeding to ``transformers.set_seed`` (which already covers
random / numpy / torch / cuda) and only adds the cuDNN determinism toggle
on top.
"""

from __future__ import annotations

import torch
import transformers


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """Seed all RNGs and toggle cuDNN deterministic mode."""
    transformers.set_seed(seed=seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
