"""Pure per-row scoring used by the stage-2 evaluator and its workers.

Isolated from ``evaluator.py`` so ``multiprocessing`` workers can import the
module without triggering the evaluator's state. All symbols here are
picklable: no closures, no bound methods, no non-trivial globals beyond a
single cached nltk smoothing function.
"""

from __future__ import annotations

import editdistance
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from src.constants import ANSWER_TOKEN

__all__ = [
    "reasoning_and_answer",
    "score_pair",
    "score_row",
    "init_worker",
]

_SMOOTHING = SmoothingFunction().method1

# Populated by ``init_worker`` on each worker process; single-process callers
# pass ``split_reasoning`` explicitly to ``score_row``.
_SPLIT_REASONING: bool = False


def init_worker(split_reasoning: bool) -> None:
    global _SPLIT_REASONING
    _SPLIT_REASONING = bool(split_reasoning)


def reasoning_and_answer(text: str) -> tuple[str, str]:
    """Split a free-form completion into (reasoning, answer).

    Whitespace around the answer token is preserved so that length-normalized
    edit distance matches between the generated string and the gold label.
    """
    parts = text.split(ANSWER_TOKEN)
    if len(parts) == 1:
        return "", parts[0]
    return ANSWER_TOKEN.join(parts[:-1]), parts[-1]


def _bleu(gen: str, ref: str) -> float:
    if not gen or not ref:
        return 0.0
    try:
        return float(
            sentence_bleu(
                [ref.split()],
                gen.split(),
                smoothing_function=_SMOOTHING,
            )
        )
    except ZeroDivisionError:
        return 0.0


def score_pair(gen: str, ref: str) -> tuple[int, float, float]:
    em = 1 if gen == ref else 0
    max_len = max(len(gen), len(ref))
    ed = (editdistance.eval(gen, ref) / max_len) if max_len > 0 else 0.0
    return em, ed, _bleu(gen, ref)


def score_row(
    row: dict, split_reasoning: bool | None = None
) -> dict[str, tuple[int, float, float]]:
    """Worker entry point. Returns one ``(em, ed, bleu)`` tuple per split.

    ``split_reasoning`` defaults to the worker-process global set by
    :func:`init_worker`; single-process callers may pass it explicitly.
    """
    if split_reasoning is None:
        split_reasoning = _SPLIT_REASONING
    predict = row["predict"]
    label = row["label"]
    out: dict[str, tuple[int, float, float]] = {"full": score_pair(predict, label)}
    if split_reasoning:
        p_r, p_a = reasoning_and_answer(predict)
        l_r, l_a = reasoning_and_answer(label)
        out["reasoning"] = score_pair(p_r, l_r)
        out["answer"] = score_pair(p_a, l_a)
    return out
