"""Base and compositional transformations.

Definitions (paper Section 4.2):
    f_rot   (f_1) : ROT-n shift applied atom-wise.
    f_pos   (f_2) : cyclic shift of the element by n positions.
    f_reverse(f_3): reverse the element.

Any compositional transformation ``f_S = f_{s_k} o ... o f_{s_1}`` is
represented by a :class:`Composition`, built from the base transformations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from src.atoms import ALPHABET, ALPHABET_SIZE
from src.constants import F_POS_TOKEN, F_REVERSE_TOKEN, F_ROT_TOKEN


class Transformation(ABC):
    """A deterministic map Element -> Element."""

    name: str
    token: str

    @abstractmethod
    def __call__(self, element: Sequence[str]) -> list[str]:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.token})"


@dataclass
class RotTransformation(Transformation):
    """``f_rot`` — ROT-n cipher. Shift each atom by ``n`` positions."""

    n: int = 13
    name: str = "f_rot"
    token: str = F_ROT_TOKEN

    def __call__(self, element: Sequence[str]) -> list[str]:
        out: list[str] = []
        for char in element:
            idx = ALPHABET.find(char)
            if idx == -1:
                out.append(char)
            else:
                out.append(ALPHABET[(idx + self.n) % ALPHABET_SIZE])
        return out


@dataclass
class PosTransformation(Transformation):
    """``f_pos`` — cyclic shift the element by ``n`` positions to the left."""

    n: int = 1
    name: str = "f_pos"
    token: str = F_POS_TOKEN

    def __call__(self, element: Sequence[str]) -> list[str]:
        element = list(element)
        if not element:
            return []
        k = self.n % len(element)
        return element[k:] + element[:k]


@dataclass
class ReverseTransformation(Transformation):
    """``f_reverse`` — reverse the element in place (paper's ``f_3``)."""

    name: str = "f_reverse"
    token: str = F_REVERSE_TOKEN

    def __call__(self, element: Sequence[str]) -> list[str]:
        return list(element)[::-1]


class Composition(Transformation):
    """A left-to-right composition ``f = f_k o ... o f_1``.

    Transformations in ``self.steps`` are applied in order, so
    ``Composition([f1, f2])(x) == f2(f1(x))``.
    """

    def __init__(self, steps: Sequence[Transformation]) -> None:
        if not steps:
            raise ValueError("Composition must contain at least one transformation")
        self.steps: list[Transformation] = list(steps)
        self.name = " o ".join(s.name for s in self.steps)
        self.token = " ".join(s.token for s in self.steps)

    def __call__(self, element: Sequence[str]) -> list[str]:
        out = list(element)
        for step in self.steps:
            out = step(out)
        return out

    def intermediates(self, element: Sequence[str]) -> list[list[str]]:
        """Return the intermediate elements after each step (for CoT traces)."""
        out = list(element)
        trace: list[list[str]] = []
        for step in self.steps:
            out = step(out)
            trace.append(list(out))
        return trace


# Convenient aliases so callers can write ``f_rot(13)`` instead of the class name.
f_rot = RotTransformation
f_pos = PosTransformation
f_reverse = ReverseTransformation
f1 = RotTransformation
f2 = PosTransformation
f3 = ReverseTransformation


_TOKEN_TO_CLS = {
    F_ROT_TOKEN: RotTransformation,
    F_POS_TOKEN: PosTransformation,
    F_REVERSE_TOKEN: ReverseTransformation,
}


def transformation_from_token(token: str, rot_n: int = 13, pos_n: int = 1) -> Transformation:
    """Instantiate a base transformation from its transformation token.

    ``rot_n`` / ``pos_n`` are ignored for :class:`ReverseTransformation`.
    """
    cls = _TOKEN_TO_CLS.get(token)
    if cls is None:
        raise KeyError(
            f"Unknown transformation token: {token!r}. "
            f"Expected one of {list(_TOKEN_TO_CLS)}."
        )
    if cls is RotTransformation:
        return RotTransformation(n=rot_n)
    if cls is PosTransformation:
        return PosTransformation(n=pos_n)
    return ReverseTransformation()


def composition_from_tokens(
    tokens: Sequence[str], rot_n: int = 13, pos_n: int = 1
) -> Composition:
    """Build a :class:`Composition` from an iterable of transformation tokens.

    Accepts either ``["[F1]", "[F2]"]`` or space-separated ``"[F1] [F2]"``.
    """
    if isinstance(tokens, str):
        tokens = tokens.split()
    return Composition(
        [transformation_from_token(t, rot_n=rot_n, pos_n=pos_n) for t in tokens]
    )
