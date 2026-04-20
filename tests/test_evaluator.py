"""Tests for the evaluator helpers."""

from __future__ import annotations

import json

import pytest

from src.constants import ANSWER_TOKEN
from src.evaluation.evaluator import Evaluation, EvalResult, write_results_json


def test_write_results_json_full_only_schema(tmp_path) -> None:
    result = EvalResult(
        full={
            "total_samples": 10,
            "exact_match_accuracy": 0.5,
            "avg_edit_distance": 0.25,
            "avg_bleu_score": 0.75,
        }
    )
    path = tmp_path / "out" / "eval.json"
    write_results_json(result, str(path))

    payload = json.loads(path.read_text())
    assert payload["num_samples"] == 10
    assert payload["full"]["exact_match_accuracy"] == pytest.approx(0.5)
    assert payload["full"]["avg_edit_distance"] == pytest.approx(0.25)
    assert payload["full"]["avg_bleu_score"] == pytest.approx(0.75)
    assert "reasoning" not in payload
    assert "answer" not in payload


def test_write_results_json_includes_reasoning_and_answer(tmp_path) -> None:
    split = {
        "total_samples": 10,
        "exact_match_accuracy": 0.5,
        "avg_edit_distance": 0.25,
        "avg_bleu_score": 0.75,
    }
    result = EvalResult(full=split, reasoning=split, answer=split)
    path = tmp_path / "eval.json"
    write_results_json(result, str(path))

    payload = json.loads(path.read_text())
    for split_name in ("full", "reasoning", "answer"):
        assert payload[split_name]["exact_match_accuracy"] == pytest.approx(0.5)
        assert payload[split_name]["avg_edit_distance"] == pytest.approx(0.25)
        assert payload[split_name]["avg_bleu_score"] == pytest.approx(0.75)


def _write_jsonl(path, rows) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_evaluation_reads_predictions_jsonl(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt": "p1", "predict": "apple", "label": "apple"},
            {"prompt": "p2", "predict": "banana", "label": "orange"},
        ],
    )
    result = Evaluation(split_reasoning=False, num_workers=1).run(str(path))
    assert result.full["total_samples"] == 2
    assert result.full["exact_match_accuracy"] == pytest.approx(0.5)
    assert result.reasoning is None
    assert result.answer is None


def test_evaluation_split_reasoning(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    pred = f"think_a {ANSWER_TOKEN} final_a"
    label = f"think_a {ANSWER_TOKEN} final_a"
    _write_jsonl(path, [{"prompt": "p", "predict": pred, "label": label}])
    result = Evaluation(split_reasoning=True, num_workers=1).run(str(path))
    assert result.full["exact_match_accuracy"] == pytest.approx(1.0)
    assert result.reasoning is not None
    assert result.answer is not None
    assert result.reasoning["exact_match_accuracy"] == pytest.approx(1.0)
    assert result.answer["exact_match_accuracy"] == pytest.approx(1.0)


def test_evaluation_parallel_matches_serial(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = []
    for i in range(50):
        pred = f"tok_{i % 7} tok_{(i * 3) % 11} {ANSWER_TOKEN} out_{i % 5}"
        label = f"tok_{i % 7} tok_{(i * 3) % 13} {ANSWER_TOKEN} out_{i % 5}"
        rows.append({"prompt": f"p{i}", "predict": pred, "label": label})
    _write_jsonl(path, rows)

    serial = Evaluation(split_reasoning=True, num_workers=1).run(str(path))
    parallel = Evaluation(split_reasoning=True, num_workers=2).run(str(path))

    for split in ("full", "reasoning", "answer"):
        a = getattr(serial, split)
        b = getattr(parallel, split)
        assert a["total_samples"] == b["total_samples"]
        assert a["exact_match_accuracy"] == pytest.approx(b["exact_match_accuracy"])
        assert a["avg_edit_distance"] == pytest.approx(b["avg_edit_distance"])
        assert a["avg_bleu_score"] == pytest.approx(b["avg_bleu_score"], abs=1e-9)
