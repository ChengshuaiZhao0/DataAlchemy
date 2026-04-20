"""Streaming accumulator for exact-match, edit-distance, and BLEU.

All three metrics are aggregated as per-sample running sums. BLEU is scored
independently for each ``(prediction, reference)`` pair via
:func:`src.evaluation.scoring._bleu` (nltk ``sentence_bleu`` with method-1
smoothing); ``avg_bleu_score`` is the arithmetic mean across samples.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from src.evaluation.scoring import _bleu, score_pair


class EvaluationMetrics:
    """Accumulator: per-sample running sums for EM, ED, and BLEU."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_samples: int = 0
        self._sum: dict[str, float] = {"em": 0.0, "ed": 0.0, "bleu": 0.0}
        self._sq_sum: dict[str, float] = {"em": 0.0, "ed": 0.0, "bleu": 0.0}

    @staticmethod
    def exact_match(gen: Sequence[str], exp: Sequence[str]) -> list[int]:
        return [1 if g == e else 0 for g, e in zip(gen, exp, strict=True)]

    @staticmethod
    def normalized_edit_distance(
        gen: Sequence[str], exp: Sequence[str]
    ) -> list[float]:
        out: list[float] = []
        for g, e in zip(gen, exp, strict=True):
            _, ed, _ = score_pair(g, e)
            out.append(ed)
        return out

    def bleu(self, gen: Sequence[str], exp: Sequence[str]) -> list[float]:
        """Per-sample BLEU via nltk sentence_bleu (method-1 smoothing)."""
        return [_bleu(g, e) for g, e in zip(gen, exp, strict=True)]

    def update_from_scores(
        self,
        em: Sequence[int],
        ed: Sequence[float],
        bleu: Sequence[float],
    ) -> None:
        """Hot path: accumulate precomputed per-sample metric lists."""
        n = len(em)
        if n == 0:
            return
        self.total_samples += n
        for key, vals in (("em", em), ("ed", ed), ("bleu", bleu)):
            self._sum[key] += sum(vals)
            self._sq_sum[key] += sum(v * v for v in vals)

    def update_batch(
        self,
        generated_texts: Sequence[str],
        expected_texts: Sequence[str],
        prompts: Sequence[str] | None = None,  # noqa: ARG002 — kept for API compat
    ) -> dict[str, Any]:
        em: list[int] = []
        ed: list[float] = []
        bl: list[float] = []
        for g, e in zip(generated_texts, expected_texts, strict=True):
            one_em, one_ed, one_bl = score_pair(g, e)
            em.append(one_em)
            ed.append(one_ed)
            bl.append(one_bl)
        self.update_from_scores(em, ed, bl)
        n = len(em)
        return {
            "batch_size": n,
            "exact_match_accuracy": (sum(em) / n) if n else 0.0,
            "avg_edit_distance": (sum(ed) / n) if n else 0.0,
            "avg_bleu_score": (sum(bl) / n) if n else 0.0,
            "per_sample_em": em,
            "per_sample_ed": ed,
            "per_sample_bleu": bl,
        }

    def get_metrics(self) -> dict[str, float]:
        n = self.total_samples
        if n == 0:
            return {
                "total_samples": 0,
                "exact_match_accuracy": 0.0,
                "avg_edit_distance": 0.0,
                "avg_bleu_score": 0.0,
                "exact_match_accuracy_std": 0.0,
                "edit_distance_std": 0.0,
                "bleu_score_std": 0.0,
            }

        def mean_std(s: float, sq: float) -> tuple[float, float]:
            mean = s / n
            var = max(0.0, sq / n - mean * mean)
            return mean, math.sqrt(var)

        em_m, em_s = mean_std(self._sum["em"], self._sq_sum["em"])
        ed_m, ed_s = mean_std(self._sum["ed"], self._sq_sum["ed"])
        bl_m, bl_s = mean_std(self._sum["bleu"], self._sq_sum["bleu"])
        return {
            "total_samples": n,
            "exact_match_accuracy": em_m,
            "avg_edit_distance": ed_m,
            "avg_bleu_score": bl_m,
            "exact_match_accuracy_std": em_s,
            "edit_distance_std": ed_s,
            "bleu_score_std": bl_s,
        }

    def print_summary(
        self,
        label: str | None = None,
        logger: "logging.Logger | None" = None,
    ) -> None:
        m = self.get_metrics()
        header = "EVALUATION RESULTS"
        if label:
            header = f"{header} \u2014 {label.upper()}"
        lines = [
            "=" * 70,
            header,
            "=" * 70,
            f"Total samples: {m['total_samples']}",
            f"Exact match  : {m['exact_match_accuracy']:.4f} "
            f"(+/- {m['exact_match_accuracy_std']:.4f})",
            f"Edit distance: {m['avg_edit_distance']:.4f} "
            f"(+/- {m['edit_distance_std']:.4f})",
            f"BLEU         : {m['avg_bleu_score']:.4f} "
            f"(+/- {m['bleu_score_std']:.4f})",
            "=" * 70,
        ]
        if logger is not None:
            for line in lines:
                logger.info(line)
        else:
            for line in lines:
                print(line, flush=True)
