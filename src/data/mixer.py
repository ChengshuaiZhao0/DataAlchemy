"""Deterministic mixing of pretraining datasets.

Merged from the legacy ``mix_dataset.py`` and ``mix_pretrain_data.py`` scripts.
Both of those were one-offs hardcoded to specific files; here we expose one
reusable helper.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Sequence


def _reservoir_sample(path: str, k: int, rng: random.Random) -> List[str]:
    """Uniformly sample ``k`` lines from ``path`` in a single streaming pass."""
    reservoir: List[str] = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i < k:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = line
    return reservoir


def mix_files(
    sources: Sequence[str],
    weights: Sequence[float],
    output_path: str,
    total_lines: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> int:
    """Mix lines from several files in the given proportions.

    Parameters
    ----------
    sources     : paths to the input files.
    weights     : non-negative mixture weights (must sum to > 0).
    output_path : destination file.
    total_lines : total number of lines to emit; defaults to min(len(src)).
    seed        : shuffle seed.

    Returns the number of lines written.
    """
    if len(sources) != len(weights):
        raise ValueError("sources and weights must have the same length")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("sum of weights must be > 0")

    rng = random.Random(seed)

    if total_lines is None:
        total_lines = min(_count_lines(p) for p in sources)

    sampled: List[str] = []
    for src, w in zip(sources, weights):
        take = int(total_lines * w / total_weight)
        if take == 0:
            continue
        chunk = _reservoir_sample(src, take, rng)
        if len(chunk) < take:
            raise ValueError(
                f"requested {take} lines from {src} which has only {len(chunk)}"
            )
        sampled.extend(chunk)

    if shuffle:
        rng.shuffle(sampled)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(sampled)
    return len(sampled)


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def sweep_mix(
    primary: str,
    secondary: str,
    ratios: Iterable[float],
    output_pattern: str,
    total_lines: int | None = None,
    seed: int = 42,
) -> Dict[float, str]:
    """For each ``p`` in ``ratios`` produce a file mixing primary:secondary at ``p:1-p``.

    ``output_pattern`` is a format string that receives ``p`` as ``{p}``
    (underscore-escaped, e.g. ``"data/mix_{p}.txt"`` -> ``"data/mix_0p3.txt"``).
    Returns the mapping ``{p: path}``.
    """
    out: Dict[float, str] = {}
    for p in ratios:
        path = output_pattern.format(p=f"{p:.1f}".replace(".", "p"))
        mix_files(
            sources=(primary, secondary),
            weights=(p, 1.0 - p),
            output_path=path,
            total_lines=total_lines,
            seed=seed,
        )
        out[p] = path
    return out
