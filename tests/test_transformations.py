"""Unit tests for the symbolic transformations (paper Section 4.2)."""

from __future__ import annotations

import pytest

from src.constants import F_POS_TOKEN, F_REVERSE_TOKEN, F_ROT_TOKEN
from src.transformations import (
    Composition,
    PosTransformation,
    ReverseTransformation,
    RotTransformation,
    composition_from_tokens,
    f_pos,
    f_reverse,
    f_rot,
    transformation_from_token,
)

APPLE = list("APPLE")


def test_f_rot_13_on_apple() -> None:
    assert f_rot(13)(APPLE) == list("NCCYR")


def test_f_rot_identity_at_n_zero_and_full_cycle() -> None:
    assert f_rot(0)(APPLE) == APPLE
    assert f_rot(26)(APPLE) == APPLE


def test_f_rot_preserves_length_and_wraps_alphabet() -> None:
    out = f_rot(1)(list("XYZ"))
    assert out == list("YZA")
    assert len(out) == 3


def test_f_pos_shifts_one_left() -> None:
    assert f_pos(1)(APPLE) == list("PPLEA")


def test_f_pos_full_cycle_is_identity() -> None:
    assert f_pos(len(APPLE))(APPLE) == APPLE
    assert f_pos(0)(APPLE) == APPLE


def test_f_pos_empty_element() -> None:
    assert f_pos(1)([]) == []


def test_f_reverse_involution() -> None:
    r = f_reverse()
    assert r(APPLE) == list("ELPPA")
    assert r(r(APPLE)) == APPLE


def test_composition_applies_left_to_right() -> None:
    comp = Composition([f_rot(13), f_pos(1)])
    assert comp(APPLE) == f_pos(1)(f_rot(13)(APPLE))
    assert comp(APPLE) == list("CCYRN")


def test_composition_intermediates_record_each_step() -> None:
    comp = Composition([f_rot(13), f_pos(1)])
    trace = comp.intermediates(APPLE)
    assert trace == [list("NCCYR"), list("CCYRN")]
    assert len(trace) == len(comp.steps)


def test_composition_preserves_length_for_base_transformations() -> None:
    comp = Composition([f_rot(13), f_pos(2), f_reverse()])
    for n in range(1, 8):
        element = list("ABCDEFGH"[:n])
        assert len(comp(element)) == n


def test_composition_empty_raises() -> None:
    with pytest.raises(ValueError):
        Composition([])


def test_double_reverse_is_identity() -> None:
    comp = Composition([f_reverse(), f_reverse()])
    assert comp(APPLE) == APPLE


def test_transformation_from_token_returns_expected_class() -> None:
    assert isinstance(transformation_from_token(F_ROT_TOKEN), RotTransformation)
    assert isinstance(transformation_from_token(F_POS_TOKEN), PosTransformation)
    assert isinstance(transformation_from_token(F_REVERSE_TOKEN), ReverseTransformation)


def test_transformation_from_token_applies_defaults() -> None:
    rot = transformation_from_token(F_ROT_TOKEN)
    pos = transformation_from_token(F_POS_TOKEN)
    assert rot.n == 13
    assert pos.n == 1


def test_transformation_from_token_accepts_custom_n() -> None:
    rot = transformation_from_token(F_ROT_TOKEN, rot_n=5)
    pos = transformation_from_token(F_POS_TOKEN, pos_n=3)
    assert rot.n == 5
    assert pos.n == 3


def test_transformation_from_token_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        transformation_from_token("[unknown]")


def test_composition_from_tokens_accepts_string_and_list() -> None:
    from_list = composition_from_tokens([F_ROT_TOKEN, F_POS_TOKEN])
    from_str = composition_from_tokens(f"{F_ROT_TOKEN} {F_POS_TOKEN}")
    assert from_list(APPLE) == from_str(APPLE)
