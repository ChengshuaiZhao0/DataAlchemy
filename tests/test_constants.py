"""Tests for the symbolic-token constants."""

from __future__ import annotations

from src import constants as C


def test_transformation_tokens_match_paper_appendix_b() -> None:
    assert C.TRANSFORMATION_TOKENS == ["[F1]", "[F2]", "[F3]"]
    assert C.F_ROT_TOKEN == "[F1]"
    assert C.F_POS_TOKEN == "[F2]"
    assert C.F_REVERSE_TOKEN == "[F3]"
    # Backward-compat alias must be the same list, not a copy.
    assert C.RULE_TOKENS is C.TRANSFORMATION_TOKENS


def test_sep_token_is_defined() -> None:
    # Regression for the legacy bug where `SEP_TOKEN` was imported from
    # constants but never declared there.
    assert hasattr(C, "SEP_TOKEN")
    assert isinstance(C.SEP_TOKEN, str) and C.SEP_TOKEN


def test_special_tokens_are_unique_non_empty_strings() -> None:
    specials = [
        C.BOS_TOKEN,
        C.PAD_TOKEN,
        C.EOS_TOKEN,
        C.UNK_TOKEN,
        C.MASK_TOKEN,
        C.SEP_TOKEN,
        C.THINK_TOKEN,
        C.ANSWER_TOKEN,
    ]
    assert all(isinstance(t, str) and t for t in specials)
    assert len(set(specials)) == len(specials)
