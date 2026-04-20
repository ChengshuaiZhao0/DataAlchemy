"""Tests for format-perturbation noise injection (paper Section 7)."""

from __future__ import annotations

import json
import random

import pytest

from src.constants import ANSWER_TOKEN, THINK_TOKEN, UNK_TOKEN
from src.data.noise import (
    NoiseMode,
    apply_noise_to_record,
    generate_noisy_dataset,
)


def _record(
    element="A B C D",
    transformation="[F1] [F2]",
    instruction=THINK_TOKEN,
    reasoning="N O P Q [F2]",
    answer="O P Q N",
) -> dict:
    return {
        "element": element,
        "transformation": transformation,
        "instruction": instruction,
        "reasoning": reasoning,
        "answer": answer,
        "input": f"{element} {transformation} {instruction}",
        "output": f"{reasoning} {ANSWER_TOKEN} {answer}".strip(),
    }


def test_legacy_mode_aliases_resolve() -> None:
    assert NoiseMode.from_string("add") is NoiseMode.INSERT
    assert NoiseMode.from_string("del") is NoiseMode.DELETE
    assert NoiseMode.from_string("modify") is NoiseMode.MODIFY
    assert NoiseMode.from_string("all") is NoiseMode.HYBRID


def test_canonical_mode_names_resolve() -> None:
    assert NoiseMode.from_string("insert") is NoiseMode.INSERT
    assert NoiseMode.from_string("delete") is NoiseMode.DELETE
    assert NoiseMode.from_string("hybrid") is NoiseMode.HYBRID


@pytest.mark.parametrize("mode", ["insert", "delete", "modify", "hybrid"])
def test_p_zero_is_identity(mode: str) -> None:
    rec = _record()
    out = apply_noise_to_record(rec, mode, p=0.0, domain="all", rng=random.Random(0))
    assert out == rec


def test_p_one_modify_replaces_targeted_field_with_unk() -> None:
    rec = _record()
    out = apply_noise_to_record(
        rec, "modify", p=1.0, domain="element", rng=random.Random(0)
    )
    assert all(tok == UNK_TOKEN for tok in out["element"].split())
    assert out["transformation"] == rec["transformation"]
    assert out["instruction"] == rec["instruction"]


def test_apply_noise_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValueError):
        apply_noise_to_record(
            _record(), "modify", p=2.0, domain="element", rng=random.Random(0)
        )


def test_domain_element_only_perturbs_element() -> None:
    rec = _record()
    out = apply_noise_to_record(
        rec, "modify", p=1.0, domain="element", rng=random.Random(0)
    )
    # Element fully replaced by <unk>.
    assert all(tok == UNK_TOKEN for tok in out["element"].split())
    # Other structural fields untouched.
    assert out["transformation"] == rec["transformation"]
    assert out["instruction"] == rec["instruction"]
    # Gold target fields preserved verbatim.
    assert out["output"] == rec["output"]
    assert out["reasoning"] == rec["reasoning"]
    assert out["answer"] == rec["answer"]


def test_domain_transformation_only_perturbs_transformation() -> None:
    rec = _record()
    out = apply_noise_to_record(
        rec, "modify", p=1.0, domain="transformation", rng=random.Random(0)
    )
    assert all(tok == UNK_TOKEN for tok in out["transformation"].split())
    assert out["element"] == rec["element"]
    assert out["instruction"] == rec["instruction"]


def test_domain_instruction_only_perturbs_instruction_marker() -> None:
    """For CoT records, `instruction` targets <think>."""
    rec = _record()  # instruction defaults to <think>
    out = apply_noise_to_record(
        rec, "modify", p=1.0, domain="instruction", rng=random.Random(0)
    )
    assert out["instruction"] == UNK_TOKEN
    assert out["element"] == rec["element"]
    assert out["transformation"] == rec["transformation"]


def test_domain_instruction_perturbs_answer_marker_for_non_cot() -> None:
    """No <think> → `instruction` targets <answer>."""
    rec = _record(instruction=ANSWER_TOKEN, reasoning="", answer="N O P Q")
    rec["output"] = "N O P Q"
    out = apply_noise_to_record(
        rec, "modify", p=1.0, domain="instruction", rng=random.Random(0)
    )
    assert out["instruction"] == UNK_TOKEN


def test_input_recomposed_from_perturbed_fields() -> None:
    rec = _record()
    out = apply_noise_to_record(
        rec, "modify", p=1.0, domain="element", rng=random.Random(0)
    )
    assert out["input"] == (
        f"{out['element']} {out['transformation']} {out['instruction']}"
    )


def test_legacy_domain_aliases_resolve_to_transformation() -> None:
    rec = _record()
    expected = apply_noise_to_record(rec, "modify", p=1.0, domain="transformation",
                                      rng=random.Random(0))
    for legacy in ("relation", "action"):
        out = apply_noise_to_record(rec, "modify", p=1.0, domain=legacy,
                                     rng=random.Random(0))
        assert out == expected


def test_legacy_prompt_alias_resolves_to_instruction() -> None:
    rec = _record()
    expected = apply_noise_to_record(
        rec, "modify", p=1.0, domain="instruction", rng=random.Random(0)
    )
    out = apply_noise_to_record(
        rec, "modify", p=1.0, domain="prompt", rng=random.Random(0)
    )
    assert out == expected


def test_unknown_domain_raises() -> None:
    with pytest.raises(ValueError):
        apply_noise_to_record(
            _record(), "modify", p=0.5, domain="bogus", rng=random.Random(0)
        )


def test_generate_noisy_dataset_round_trips_jsonl(tmp_path) -> None:
    src = tmp_path / "clean.jsonl"
    records = [_record(), _record(element="X Y Z W")]
    src.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out = tmp_path / "noisy.jsonl"

    written = generate_noisy_dataset(
        source_paths=[str(src)],
        output_path=str(out),
        mode="modify",
        p=0.0,  # identity, so we can assert verbatim equality
        seed=123,
    )
    assert written == 2
    out_records = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert out_records == records


def test_generate_noisy_dataset_preserves_gold_target_fields(tmp_path) -> None:
    src = tmp_path / "clean.jsonl"
    records = [_record() for _ in range(5)]
    src.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out = tmp_path / "noisy.jsonl"

    generate_noisy_dataset(
        source_paths=[str(src)],
        output_path=str(out),
        mode="modify",
        p=1.0,
        domain="element",
        seed=0,
    )
    perturbed_records = [
        json.loads(line) for line in out.read_text().splitlines() if line
    ]
    for original, perturbed in zip(records, perturbed_records, strict=True):
        assert perturbed["output"] == original["output"]
        assert perturbed["reasoning"] == original["reasoning"]
        assert perturbed["answer"] == original["answer"]
