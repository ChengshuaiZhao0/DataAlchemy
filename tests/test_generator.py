"""Tests for DatasetGenerator — JSONL schema and CoT correctness."""

from __future__ import annotations

import json

from src.constants import ANSWER_TOKEN, F_REVERSE_TOKEN, F_ROT_TOKEN, THINK_TOKEN
from src.data.generator import DatasetGenerator
from src.transformations import composition_from_tokens

REQUIRED_FIELDS = {
    "input", "output", "element", "transformation",
    "instruction", "reasoning", "answer",
}


def _read_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_generator_writes_exhaustive_corpus(tmp_path) -> None:
    out = tmp_path / "F1F2.jsonl"
    gen = DatasetGenerator(element_length=3)
    written = gen.generate(["[F1]", "[F2]"], str(out))
    records = _read_records(str(out))
    assert written == len(records) == 26**3


def test_every_record_has_seven_fields(tmp_path) -> None:
    out = tmp_path / "F1F2.jsonl"
    DatasetGenerator(element_length=2).generate(["[F1]", "[F2]"], str(out))
    for record in _read_records(str(out)):
        assert set(record) == REQUIRED_FIELDS
        assert record["instruction"] == THINK_TOKEN  # 2-step CoT
        assert record["reasoning"]


def test_record_invariants(tmp_path) -> None:
    out = tmp_path / "F1F2.jsonl"
    DatasetGenerator(element_length=2).generate(["[F1]", "[F2]"], str(out))
    for record in _read_records(str(out))[:20]:
        # input == element + " " + transformation + " " + instruction
        assert record["input"] == (
            f"{record['element']} {record['transformation']} {record['instruction']}"
        )
        # full rendered line round-trips
        rendered = f"{record['input']} {record['output']}"
        assert THINK_TOKEN in rendered
        assert ANSWER_TOKEN in rendered


def test_line_count_respects_downsample(tmp_path) -> None:
    out = tmp_path / "downsample.jsonl"
    written = DatasetGenerator(element_length=3).generate(
        ["[F1]", "[F2]"], str(out), downsample_to=50
    )
    assert written <= 50
    assert len(_read_records(str(out))) == written


def test_intermediate_reasoning_matches_composition(tmp_path) -> None:
    out = tmp_path / "F1F2.jsonl"
    DatasetGenerator(element_length=2).generate(["[F1]", "[F2]"], str(out))
    comp = composition_from_tokens(["[F1]", "[F2]"])

    for record in _read_records(str(out))[:10]:
        element = record["element"].split()
        trace = comp.intermediates(element)
        # The reasoning field starts with the first intermediate.
        reasoning_tokens = record["reasoning"].split()
        assert reasoning_tokens[: len(trace[0])] == trace[0]
        # The answer matches the final element after the full composition.
        assert record["answer"].split() == trace[-1]


def test_single_transformation_shortcut_has_no_think(tmp_path) -> None:
    out = tmp_path / "F1_only.jsonl"
    DatasetGenerator(element_length=2).generate_single_transformation(
        F_ROT_TOKEN, str(out)
    )
    for record in _read_records(str(out)):
        assert record["instruction"] == ANSWER_TOKEN
        assert record["reasoning"] == ""
        assert record["output"] == record["answer"]


def test_reverse_rule_writes_dataset_without_error(tmp_path) -> None:
    # Regression for the legacy `reverse_cipher` bug in the old generator.
    out = tmp_path / "F3F3.jsonl"
    gen = DatasetGenerator(element_length=2)
    written = gen.generate([F_REVERSE_TOKEN, F_REVERSE_TOKEN], str(out))
    assert written == 26**2
