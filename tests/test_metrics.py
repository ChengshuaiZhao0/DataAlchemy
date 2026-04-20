"""Tests for the evaluation metrics accumulator."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import EvaluationMetrics


def test_exact_match_all_and_none() -> None:
    assert EvaluationMetrics.exact_match(["NCCYR"], ["NCCYR"]) == [1]
    assert EvaluationMetrics.exact_match(["NCCYR"], ["APPLE"]) == [0]
    assert EvaluationMetrics.exact_match(
        ["a", "b", "c"], ["a", "x", "c"]
    ) == [1, 0, 1]


def test_normalized_edit_distance_bounds() -> None:
    assert EvaluationMetrics.normalized_edit_distance(["apple"], ["apple"]) == [0.0]
    assert EvaluationMetrics.normalized_edit_distance(["apple"], ["zzzzz"]) == [1.0]
    assert EvaluationMetrics.normalized_edit_distance([""], [""]) == [0.0]


def test_normalized_edit_distance_partial() -> None:
    eds = EvaluationMetrics.normalized_edit_distance(["apple"], ["appld"])
    assert 0.0 < eds[0] < 1.0


@pytest.fixture
def metrics() -> EvaluationMetrics:
    return EvaluationMetrics()


def test_empty_state_returns_zeroes(metrics: EvaluationMetrics) -> None:
    m = metrics.get_metrics()
    assert m["total_samples"] == 0
    assert m["exact_match_accuracy"] == 0.0
    assert m["avg_edit_distance"] == 0.0
    assert m["avg_bleu_score"] == 0.0


def test_update_batch_returns_batch_metrics(metrics: EvaluationMetrics) -> None:
    out = metrics.update_batch(["apple", "banana"], ["apple", "orange"])
    assert out["batch_size"] == 2
    assert 0.0 <= out["exact_match_accuracy"] <= 1.0
    assert 0.0 <= out["avg_edit_distance"] <= 1.0
    assert 0.0 <= out["avg_bleu_score"] <= 1.0


def test_update_batch_accumulates_across_calls(metrics: EvaluationMetrics) -> None:
    metrics.update_batch(["apple"], ["apple"])
    metrics.update_batch(["banana"], ["orange"])
    m = metrics.get_metrics()
    assert m["total_samples"] == 2
    assert m["exact_match_accuracy"] == pytest.approx(0.5)


def test_reset_clears_state(metrics: EvaluationMetrics) -> None:
    metrics.update_batch(["apple"], ["apple"])
    metrics.reset()
    m = metrics.get_metrics()
    assert m["total_samples"] == 0
