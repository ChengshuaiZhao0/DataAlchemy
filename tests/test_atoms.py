"""Tests for the atom alphabet utilities."""

from __future__ import annotations

from src.atoms import ALPHABET, ALPHABET_SIZE, element_to_str, is_atom, str_to_element


def test_alphabet_is_26_uppercase_letters() -> None:
    assert len(ALPHABET) == 26
    assert ALPHABET_SIZE == 26
    assert ALPHABET == "".join(sorted(set(ALPHABET)))
    assert ALPHABET.isalpha() and ALPHABET.isupper()


def test_is_atom_true_for_single_uppercase_letters() -> None:
    for c in ALPHABET:
        assert is_atom(c)


def test_is_atom_false_for_non_atoms() -> None:
    for bad in ("a", "AB", "", "1", "?", "AA", " "):
        assert not is_atom(bad)


def test_element_string_roundtrip() -> None:
    element = ["A", "P", "P", "L", "E"]
    text = element_to_str(element)
    assert text == "A P P L E"
    assert str_to_element(text) == element
    # Idempotence on whitespace padding.
    assert str_to_element("  A P P L E  \n") == element
