"""Format-noise injection for the format-generalization experiments (Section 7).

Modes (paper terminology)::

    insert  : with prob p, insert an <unk> before the token
    delete  : with prob p, drop the token
    modify  : with prob p, replace the token with <unk>
    hybrid  : with prob p, sample one of {insert, delete, modify}

Domains (paper Figure 7) — perturb only that field of the JSONL record::

    element        : input element atoms
    transformation : the run of transformation tokens [F*]
    instruction    : the single output-start marker (<think> or <answer>)
    all            : all three above

Operates on JSONL records produced by ``DatasetGenerator``. Each record's
``output`` / ``reasoning`` / ``answer`` fields are gold targets — never
perturbed; only ``element`` / ``transformation`` / ``instruction`` and the
re-derived ``input`` are touched.
"""

from __future__ import annotations

import json
import os
import random
from enum import Enum
from typing import Any, Dict, List, Sequence

from src.constants import UNK_TOKEN


class NoiseMode(str, Enum):
    INSERT = "insert"
    DELETE = "delete"
    MODIFY = "modify"
    HYBRID = "hybrid"

    @classmethod
    def from_string(cls, mode: "NoiseMode | str") -> "NoiseMode":
        if isinstance(mode, NoiseMode):
            return mode
        if mode in _LEGACY_MODE_ALIASES:
            return _LEGACY_MODE_ALIASES[mode]
        return cls(mode)


# Legacy mode names kept for backwards-compatible config loading.
_LEGACY_MODE_ALIASES = {
    "add": NoiseMode.INSERT,
    "del": NoiseMode.DELETE,
    "modify": NoiseMode.MODIFY,
    "all": NoiseMode.HYBRID,
}

# Old per-position domain names. Both used to mean a single transformation-
# token index; under the structural classifier they collapse to
# "transformation". `prompt` is the pre-rename label for `instruction`.
_LEGACY_DOMAIN_ALIASES = {
    "relation": "transformation",
    "action": "transformation",
    "prompt": "instruction",
}

_VALID_DOMAINS = {"all", "element", "transformation", "instruction"}


def _coerce_mode(mode: NoiseMode | str) -> NoiseMode:
    return NoiseMode.from_string(mode)


def _coerce_domain(domain: str) -> str:
    return _LEGACY_DOMAIN_ALIASES.get(domain, domain)


def _perturb_tokens(
    tokens: Sequence[str],
    mode: NoiseMode,
    p: float,
    rng: random.Random,
) -> List[str]:
    """Apply the chosen mode to every token in ``tokens`` with probability p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("Probability p must be in [0, 1]")
    out: List[str] = []
    for token in tokens:
        r = rng.random()
        if mode is NoiseMode.INSERT:
            if r < p:
                out.append(UNK_TOKEN)
            out.append(token)
        elif mode is NoiseMode.DELETE:
            if r >= p:
                out.append(token)
        elif mode is NoiseMode.MODIFY:
            out.append(UNK_TOKEN if r < p else token)
        elif mode is NoiseMode.HYBRID:
            if r < p:
                op = rng.choice(["insert", "delete", "modify"])
                if op == "insert":
                    out.append(UNK_TOKEN)
                    out.append(token)
                elif op == "delete":
                    continue
                else:
                    out.append(UNK_TOKEN)
            else:
                out.append(token)
    return out


def _perturb_field(
    record: Dict[str, Any],
    field: str,
    mode: NoiseMode,
    p: float,
    rng: random.Random,
) -> None:
    """Perturb ``record[field]``'s tokens in place."""
    record[field] = " ".join(
        _perturb_tokens(str(record[field]).split(), mode, p, rng)
    )


def _recompose_input(record: Dict[str, Any]) -> None:
    """Recompose ``record["input"]`` from element / transformation / instruction."""
    record["input"] = " ".join(
        s for s in (
            record["element"], record["transformation"], record["instruction"]
        ) if s
    )


def apply_noise_to_record(
    record: Dict[str, Any],
    mode: NoiseMode | str,
    p: float,
    domain: str,
    rng: random.Random,
) -> Dict[str, Any]:
    """Return a new record with the requested domain perturbed."""
    mode_e = _coerce_mode(mode)
    domain = _coerce_domain(domain)
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Unknown domain: {domain!r}")

    out = dict(record)  # gold target fields pass through unchanged
    if domain == "all":
        for field in ("element", "transformation", "instruction"):
            _perturb_field(out, field, mode_e, p, rng)
    else:
        _perturb_field(out, domain, mode_e, p, rng)
    _recompose_input(out)
    return out


def generate_noisy_dataset(
    source_paths: Sequence[str],
    output_path: str,
    mode: NoiseMode | str,
    p: float,
    domain: str = "all",
    seed: int = 42,
) -> int:
    """Read JSONL records from ``source_paths``, perturb the requested domain.

    Writes JSONL to ``output_path``. ``output`` / ``reasoning`` / ``answer``
    are preserved verbatim — they describe the original gold target.
    """
    rng = random.Random(seed)
    mode_e = _coerce_mode(mode)
    domain = _coerce_domain(domain)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0
    with open(output_path, "w") as out:
        for path in source_paths:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    noisy = apply_noise_to_record(record, mode_e, p, domain, rng)
                    out.write(json.dumps(noisy) + "\n")
                    written += 1
    return written
