"""Exhaustive dataset generator for arbitrary compositional transformations.

Each line of output is a JSON object with the following fields::

    {
      "input":          "A B C D [F1] [F2] <think>",
      "output":         "N O P Q [F2] <answer> O P Q N",
      "element":        "A B C D",
      "transformation": "[F1] [F2]",
      "instruction":    "<think>",   # or <answer> for non-CoT lines
      "reasoning":      "N O P Q [F2]",  # empty string when not use_cot
      "answer":         "O P Q N"
    }

Invariants:

    input == f"{element} {transformation} {instruction}"
    full rendered line == f"{input} {output}"
"""

from __future__ import annotations

import json
import os
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence

from src.atoms import ALPHABET
from src.constants import ANSWER_TOKEN, THINK_TOKEN
from src.transformations import (
    Composition,
    composition_from_tokens,
)


class DatasetGenerator:
    """Generates exhaustive symbolic CoT datasets.

    Parameters
    ----------
    element_length : length ``l`` of each element (paper Section 4.1).
    rot_n, pos_n   : parameters for ``f_rot`` / ``f_pos`` base transformations.
    alphabet       : override the default A-Z alphabet.
    """

    def __init__(
        self,
        element_length: int,
        rot_n: int = 13,
        pos_n: int = 1,
        alphabet: str = ALPHABET,
    ) -> None:
        self.element_length = int(element_length)
        self.rot_n = int(rot_n)
        self.pos_n = int(pos_n)
        self.alphabet = alphabet
        self.total_sequences = len(self.alphabet) ** self.element_length

    def _iter_elements(self) -> Iterable[List[str]]:
        for tup in product(self.alphabet, repeat=self.element_length):
            yield list(tup)

    def _iter_sampled_elements(
        self, n: int, seed: int = 42
    ) -> Iterable[List[str]]:
        import random

        rng = random.Random(seed)
        seen: set = set()
        while len(seen) < n:
            tup = tuple(rng.choices(self.alphabet, k=self.element_length))
            if tup in seen:
                continue
            seen.add(tup)
            yield list(tup)

    def generate(
        self,
        transformation_tokens: Sequence[str],
        filepath: str,
        use_cot: bool = True,
        downsample_to: int | None = None,
    ) -> int:
        """Apply ``transformation_tokens`` to every element and write JSONL.

        Returns the number of lines written. ``filepath`` should end in
        ``.jsonl``; a warning is logged if it does not.
        """
        if not filepath.endswith(".jsonl"):
            print(
                f"WARNING: generate() writes JSONL but {filepath} does not end "
                f"in .jsonl"
            )
        composition = composition_from_tokens(
            transformation_tokens, rot_n=self.rot_n, pos_n=self.pos_n
        )
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        if downsample_to is not None and downsample_to < self.total_sequences:
            element_iter = self._iter_sampled_elements(downsample_to)
        else:
            element_iter = self._iter_elements()

        written = 0
        with open(filepath, "w") as f:
            for record in self._iter_records(
                composition, transformation_tokens, element_iter, use_cot=use_cot
            ):
                f.write(json.dumps(record) + "\n")
                written += 1
        return written

    def _iter_records(
        self,
        composition: Composition,
        transformation_tokens: Sequence[str],
        element_iter: Iterable[List[str]],
        use_cot: bool,
    ) -> Iterable[Dict[str, Any]]:
        tokens = list(transformation_tokens)
        for element in element_iter:
            yield _build_record(
                element,
                tokens,
                composition.intermediates(element),
                use_cot=use_cot,
            )

    def generate_single_transformation(
        self, transformation_token: str, filepath: str
    ) -> int:
        """Shortcut for a length-1 composition (single base transformation)."""
        return self.generate([transformation_token], filepath, use_cot=False)


def _build_record(
    element: Sequence[str],
    transformation_tokens: Sequence[str],
    intermediates: Sequence[Sequence[str]],
    use_cot: bool,
) -> Dict[str, Any]:
    """Construct one JSONL record from the raw pieces.

    ``intermediates`` is the per-step trace produced by
    :meth:`Composition.intermediates`; its last element is always the final
    answer.
    """
    element_str = " ".join(element)
    transformation_str = " ".join(transformation_tokens)
    answer_str = " ".join(intermediates[-1])

    if use_cot and len(transformation_tokens) > 1:
        instruction = THINK_TOKEN
        reasoning_parts: List[str] = []
        for step_idx in range(len(transformation_tokens) - 1):
            if step_idx > 0:
                reasoning_parts.append(THINK_TOKEN)
            reasoning_parts.append(" ".join(intermediates[step_idx]))
            reasoning_parts.extend(transformation_tokens[step_idx + 1 :])
        reasoning_str = " ".join(reasoning_parts)
        output_str = f"{reasoning_str} {ANSWER_TOKEN} {answer_str}"
    else:
        instruction = ANSWER_TOKEN
        reasoning_str = ""
        output_str = answer_str

    input_str = f"{element_str} {transformation_str} {instruction}"
    return {
        "input": input_str,
        "output": output_str,
        "element": element_str,
        "transformation": transformation_str,
        "instruction": instruction,
        "reasoning": reasoning_str,
        "answer": answer_str,
    }
