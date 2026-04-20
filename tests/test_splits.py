"""Tests for the task-generalization scenario builder."""

from __future__ import annotations

import pytest

from src.data.splits import Scenario, rules_for_scenario

TRAIN_2 = ["[F1]", "[F2]"]
HELD_OUT = ["[F3]"]


def test_scenario_enum_has_expected_members() -> None:
    assert {s.value for s in Scenario} == {"ID", "CMP", "POOD", "OOD"}


@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("ID", ["[F1]", "[F2]"]),
        ("CMP", ["[F2]", "[F1]"]),
        ("POOD", ["[F1]", "[F3]"]),
        ("OOD", ["[F3]", "[F3]"]),
    ],
)
def test_rules_for_two_rule_train(scenario: str, expected) -> None:
    assert rules_for_scenario(scenario, TRAIN_2, HELD_OUT) == expected


def test_scenario_accepts_enum_member() -> None:
    assert rules_for_scenario(Scenario.ID, TRAIN_2, HELD_OUT) == TRAIN_2


def test_cmp_requires_at_least_two_train_rules() -> None:
    with pytest.raises(ValueError):
        rules_for_scenario("CMP", ["[F1]"], HELD_OUT)


def test_pood_and_ood_require_held_out_rule() -> None:
    with pytest.raises(ValueError):
        rules_for_scenario("POOD", TRAIN_2, [])
    with pytest.raises(ValueError):
        rules_for_scenario("OOD", TRAIN_2, [])


def test_empty_train_rules_always_rejected() -> None:
    with pytest.raises(ValueError):
        rules_for_scenario("ID", [], HELD_OUT)


def test_ood_length_matches_train_length() -> None:
    out = rules_for_scenario("OOD", ["[F1]", "[F2]", "[F1]"], HELD_OUT)
    assert out == ["[F3]", "[F3]", "[F3]"]


def test_pood_swaps_only_last_train_rule() -> None:
    out = rules_for_scenario("POOD", ["[F1]", "[F2]", "[F1]"], HELD_OUT)
    assert out == ["[F1]", "[F2]", "[F3]"]
