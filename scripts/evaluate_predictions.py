"""CLI: stage-2 evaluation — score a predictions JSONL into metrics JSON."""

from __future__ import annotations

import os
import sys

# Python auto-prepends scripts/ to sys.path when running this file directly,
# which shadows the HuggingFace ``evaluate`` library that metrics.py imports.
# Strip our own dir before any project imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

from src.evaluation.evaluator import Evaluation, write_results_json  # noqa: E402
from src.utils.cli import parse_and_setup  # noqa: E402
from src.utils.config import parse_test_spec  # noqa: E402
from src.utils.logging import get_logger, setup_run_log  # noqa: E402
from src.utils.run_logging import log_run_footer, log_run_header  # noqa: E402

logger = get_logger("src.evaluate_predictions")


def main() -> None:
    _, cfg, _ = parse_and_setup()
    setup_run_log(stage="Evaluation", cfg=cfg)

    evaluation = Evaluation(
        split_reasoning=bool(getattr(cfg.evaluate, "split_reasoning", True)),
        num_workers=int(getattr(cfg.evaluate, "num_workers", 0)),
    )

    model_dir = cfg.checkpoint.model_dir
    result_dir = getattr(
        cfg.checkpoint, "result_dir", os.path.join(model_dir, "results")
    )
    os.makedirs(result_dir, exist_ok=True)
    info = log_run_header(logger, cfg, result_dir)
    combined_summary: dict[str, object] = {}
    for test_paths in parse_test_spec(cfg.dataset.test):
        tag = "+".join(os.path.splitext(os.path.basename(p))[0] for p in test_paths)
        predictions_path = os.path.join(result_dir, tag, "predictions.jsonl")
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(
                f"Missing predictions for tag '{tag}': {predictions_path}. "
                f"Run scripts/model_inference.py first."
            )
        logger.info("Scoring %s", predictions_path)
        result = evaluation.run(predictions_path)
        write_results_json(result, os.path.join(result_dir, tag, "results.json"))

        combined_summary[tag] = {
            "exact_match": result.full.get("exact_match_accuracy"),
            "avg_edit_distance": result.full.get("avg_edit_distance"),
            "avg_bleu": result.full.get("avg_bleu_score"),
            "num_samples": result.full.get("total_samples"),
        }
    log_run_footer(logger, info, extra=combined_summary)


if __name__ == "__main__":
    main()
