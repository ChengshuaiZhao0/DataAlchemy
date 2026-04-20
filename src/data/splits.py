"""Scenario definitions for task generalization (paper Section 5).

    ID   : transformation seen during training (in-distribution)
    CMP  : permutation of ID rules (same transformations, new composition)
    POOD : one rule swapped for an unseen rule (partially out-of-distribution)
    OOD  : all rules unseen during training
"""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence


class Scenario(str, Enum):
    ID = "ID"
    CMP = "CMP"
    POOD = "POOD"
    OOD = "OOD"


def rules_for_scenario(
    scenario: Scenario | str,
    train_rules: Sequence[str],
    held_out_rules: Sequence[str],
) -> List[str]:
    """Return a single evaluation rule list for the requested scenario.

    Examples
    --------
    >>> rules_for_scenario("ID", ["[F1]", "[F2]"], ["[F3]"])
    ['[F1]', '[F2]']
    >>> rules_for_scenario("CMP", ["[F1]", "[F2]"], ["[F3]"])
    ['[F2]', '[F1]']
    >>> rules_for_scenario("POOD", ["[F1]", "[F2]"], ["[F3]"])
    ['[F1]', '[F3]']
    >>> rules_for_scenario("OOD", ["[F1]", "[F2]"], ["[F3]"])
    ['[F3]', '[F3]']
    """
    scen = Scenario(scenario) if isinstance(scenario, str) else scenario
    train = list(train_rules)
    held = list(held_out_rules)
    if not train:
        raise ValueError("train_rules cannot be empty")

    if scen is Scenario.ID:
        return list(train)
    if scen is Scenario.CMP:
        if len(train) < 2:
            raise ValueError("CMP requires at least two train rules")
        return list(reversed(train))
    if scen is Scenario.POOD:
        if not held:
            raise ValueError("POOD requires at least one held-out rule")
        return list(train[:-1]) + [held[0]]
    if scen is Scenario.OOD:
        if not held:
            raise ValueError("OOD requires at least one held-out rule")
        return [held[0]] * len(train)
    raise ValueError(f"Unknown scenario: {scenario}")
