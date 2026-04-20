"""Atoms and elements as defined in Section 4.1 of the paper.

An *atom* is a character drawn from a finite alphabet ``A = {A, B, ..., Z}``.
An *element* is an ordered sequence of atoms of length ``l``.
"""

from __future__ import annotations

import string
from collections.abc import Iterable

ALPHABET: str = string.ascii_uppercase
ALPHABET_SIZE: int = len(ALPHABET)


def is_atom(char: str) -> bool:
    return len(char) == 1 and char in ALPHABET


def element_to_str(element: Iterable[str]) -> str:
    return " ".join(element)


def str_to_element(text: str) -> list[str]:
    return text.strip().split()
