"""Unit tests for dataset-spec parsers in src.utils.config."""

from __future__ import annotations

import pytest

from src.utils.config import parse_test_spec, parse_train_spec


def test_parse_train_spec_single_path() -> None:
    assert parse_train_spec("data/a.jsonl") == ["data/a.jsonl"]


def test_parse_train_spec_multiple_paths_comma() -> None:
    assert parse_train_spec("data/a.jsonl,data/b.jsonl") == [
        "data/a.jsonl",
        "data/b.jsonl",
    ]


def test_parse_train_spec_trims_whitespace() -> None:
    assert parse_train_spec(" data/a.jsonl , data/b.jsonl ") == [
        "data/a.jsonl",
        "data/b.jsonl",
    ]


def test_parse_train_spec_rejects_list() -> None:
    with pytest.raises(TypeError):
        parse_train_spec(["data/a.jsonl"])  # type: ignore[arg-type]


def test_parse_train_spec_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_train_spec(" ")


def test_parse_test_spec_one_line() -> None:
    assert parse_test_spec(["data/a.jsonl"]) == [["data/a.jsonl"]]


def test_parse_test_spec_multi_line() -> None:
    assert parse_test_spec(["data/a.jsonl", "data/b.jsonl"]) == [
        ["data/a.jsonl"],
        ["data/b.jsonl"],
    ]


def test_parse_test_spec_comma_within_line() -> None:
    assert parse_test_spec(["data/a.jsonl,data/b.jsonl", "data/c.jsonl"]) == [
        ["data/a.jsonl", "data/b.jsonl"],
        ["data/c.jsonl"],
    ]


def test_parse_test_spec_rejects_string() -> None:
    with pytest.raises(TypeError):
        parse_test_spec("data/a.jsonl")  # type: ignore[arg-type]


def test_parse_test_spec_rejects_empty_list() -> None:
    with pytest.raises(TypeError):
        parse_test_spec([])
