"""Tests for the global seeding helper."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.seed import set_seed


def _draw_sequence() -> tuple[list, list, torch.Tensor]:
    py = [random.random() for _ in range(5)]
    npy = list(np.random.rand(5))
    tor = torch.rand(5)
    return py, npy, tor


def test_set_seed_is_reproducible_for_python_numpy_torch() -> None:
    set_seed(1234)
    py1, np1, t1 = _draw_sequence()

    set_seed(1234)
    py2, np2, t2 = _draw_sequence()

    assert py1 == py2
    assert np.allclose(np1, np2)
    assert torch.equal(t1, t2)


def test_different_seeds_produce_different_sequences() -> None:
    set_seed(0)
    py0, np0, t0 = _draw_sequence()
    set_seed(1)
    py1, np1, t1 = _draw_sequence()
    assert py0 != py1
    assert not np.allclose(np0, np1)
    assert not torch.equal(t0, t1)