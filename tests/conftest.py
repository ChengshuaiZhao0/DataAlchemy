"""Shared pytest fixtures for the DataAlchemy test suite.

Fixtures favour small, CPU-only, fully-in-memory artefacts so that the whole
non-``slow`` suite finishes in a few seconds.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture
def deterministic_rng() -> random.Random:
    return random.Random(0)


@pytest.fixture
def toy_dataset_files(tmp_path) -> dict[str, str]:
    """Generate three small toy JSONL datasets (l=3) with `DatasetGenerator`."""
    from src.data.generator import DatasetGenerator

    gen = DatasetGenerator(element_length=3)
    paths: dict[str, str] = {}
    for name, rules in (
        ("F1F1", ["[F1]", "[F1]"]),
        ("F1F2", ["[F1]", "[F2]"]),
        ("F2F1", ["[F2]", "[F1]"]),
    ):
        path = str(tmp_path / f"{name}.jsonl")
        gen.generate(rules, path)
        paths[name] = path
    return paths


# Backward-compat fixture alias for callers that haven't been renamed yet.
@pytest.fixture
def toy_rule_files(toy_dataset_files) -> dict[str, str]:
    return toy_dataset_files


@pytest.fixture
def toy_model_spec():
    from src.models import ARCH_GPT, ModelSpec

    return ModelSpec(
        arch=ARCH_GPT,
        hidden_size=32,
        intermediate_size=64,
        n_layer=1,
        n_head=2,
        n_positions=64,
    )


@pytest.fixture
def gpt_tokenizer(tmp_path, toy_rule_files):
    """A ByteLevel-BPE tokenizer trained once on the toy corpus."""
    from src.models import ARCH_GPT, build_tokenizer

    corpus: list[str] = list(toy_rule_files.values())
    tokenizer_dir = str(tmp_path / "tokenizer")
    return build_tokenizer(
        tokenizer_dir, ARCH_GPT, corpus_files=corpus, vocab_size=512
    )


@pytest.fixture
def toy_gpt_model(toy_model_spec, gpt_tokenizer):
    from src.models import build_model

    return build_model(toy_model_spec, gpt_tokenizer)
