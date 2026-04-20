"""Stage-2 evaluation: read a predictions JSONL and compute metric splits.

The paper reports three metric splits:
  * full-chain   : metrics over the entire ``<think> ... <answer> ...`` tail
  * reasoning    : metrics over the ``<think> ... <answer>`` prefix
  * answer       : metrics over the final element after ``<answer>``

Scoring is fused per row (all three splits in one pass) and parallelized
across worker processes when ``num_workers > 1``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import dataclass, field

from tqdm import tqdm

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.scoring import init_worker, reasoning_and_answer, score_row
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["EvalResult", "Evaluation", "write_results_json", "reasoning_and_answer"]


@dataclass
class EvalResult:
    full: dict[str, float] = field(default_factory=dict)
    reasoning: dict[str, float] | None = None
    answer: dict[str, float] | None = None


def _default_num_workers() -> int:
    return min(8, os.cpu_count() or 1)


class Evaluation:
    """Reads a predictions JSONL and produces an :class:`EvalResult`."""

    def __init__(
        self,
        split_reasoning: bool = True,
        num_workers: int = 0,
    ) -> None:
        self.split_reasoning = split_reasoning
        self.num_workers = int(num_workers) if num_workers else _default_num_workers()

    def run(self, predictions_path: str) -> EvalResult:
        rows: list[dict] = []
        with open(predictions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        n = len(rows)
        logger.info(
            "Scoring %d rows from %s (num_workers=%d, split_reasoning=%s)",
            n, predictions_path, self.num_workers, self.split_reasoning,
        )

        full = EvaluationMetrics()
        reasoning = EvaluationMetrics() if self.split_reasoning else None
        answer = EvaluationMetrics() if self.split_reasoning else None

        if n == 0:
            result = EvalResult(full=full.get_metrics())
            if self.split_reasoning:
                result.reasoning = reasoning.get_metrics()
                result.answer = answer.get_metrics()
            return result

        full_em: list[int] = []
        full_ed: list[float] = []
        full_bl: list[float] = []
        reas_em: list[int] = []
        reas_ed: list[float] = []
        reas_bl: list[float] = []
        ans_em: list[int] = []
        ans_ed: list[float] = []
        ans_bl: list[float] = []

        def _collect(per_row: dict) -> None:
            em, ed, bl = per_row["full"]
            full_em.append(em)
            full_ed.append(ed)
            full_bl.append(bl)
            if self.split_reasoning:
                em, ed, bl = per_row["reasoning"]
                reas_em.append(em)
                reas_ed.append(ed)
                reas_bl.append(bl)
                em, ed, bl = per_row["answer"]
                ans_em.append(em)
                ans_ed.append(ed)
                ans_bl.append(bl)

        desc = f"Scoring {os.path.basename(os.path.dirname(predictions_path))}"
        if self.num_workers <= 1:
            for row in tqdm(rows, total=n, desc=desc):
                _collect(score_row(row, split_reasoning=self.split_reasoning))
        else:
            chunksize = max(1, n // (self.num_workers * 8))
            with mp.Pool(
                self.num_workers,
                initializer=init_worker,
                initargs=(self.split_reasoning,),
            ) as pool:
                for out in tqdm(
                    pool.imap_unordered(score_row, rows, chunksize=chunksize),
                    total=n,
                    desc=desc,
                ):
                    _collect(out)

        full.update_from_scores(full_em, full_ed, full_bl)
        full.print_summary(label="full", logger=logger)
        result = EvalResult(full=full.get_metrics())

        if self.split_reasoning:
            reasoning.update_from_scores(reas_em, reas_ed, reas_bl)
            reasoning.print_summary(label="reasoning", logger=logger)
            answer.update_from_scores(ans_em, ans_ed, ans_bl)
            answer.print_summary(label="answer", logger=logger)
            result.reasoning = reasoning.get_metrics()
            result.answer = answer.get_metrics()
        return result


def write_results_json(result: EvalResult, path: str) -> None:
    """Write the per-split metrics dict as a single JSON object.

    Stores raw 0..1 floats (no percentage formatting); downstream tooling
    can format. ``reasoning`` / ``answer`` are omitted when not populated.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: dict[str, object] = {
        "num_samples": int(result.full.get("total_samples", 0)),
        "full": result.full,
    }
    if result.reasoning is not None:
        payload["reasoning"] = result.reasoning
    if result.answer is not None:
        payload["answer"] = result.answer

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
