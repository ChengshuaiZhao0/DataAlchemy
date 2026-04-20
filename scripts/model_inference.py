"""CLI: stage-1 model inference — generate predictions JSONL from a trained model."""

from __future__ import annotations

import os
import sys

# Python auto-prepends scripts/ to sys.path when running this file directly,
# which can shadow same-named third-party packages. Strip our own dir before
# any project imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

from src.inference import Inference  # noqa: E402
from src.models import load_tokenizer_and_model  # noqa: E402
from src.utils.cli import parse_and_setup  # noqa: E402
from src.utils.config import parse_test_spec  # noqa: E402
from src.utils.logging import get_logger, setup_run_log  # noqa: E402
from src.utils.run_logging import log_run_footer, log_run_header  # noqa: E402

logger = get_logger("src.model_inference")


def main() -> None:
    args, cfg, device = parse_and_setup(
        add_args=lambda p: p.add_argument(
            "--checkpoint", default=None, help="Override cfg.checkpoint.model_dir"
        )
    )
    setup_run_log(stage="Inference", cfg=cfg)

    model_dir = args.checkpoint or cfg.checkpoint.model_dir
    final_dir = os.path.join(model_dir, "final")
    logger.info("Loading model from %s", final_dir)
    tokenizer, model = load_tokenizer_and_model(
        final_dir,
        cfg.model.arch,
        attn_implementation=str(getattr(cfg.model, "attn_implementation", "sdpa")),
    )
    model.to(device)

    top_k = getattr(cfg.inference, "top_k", None)
    top_p = getattr(cfg.inference, "top_p", None)
    inference = Inference(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(getattr(cfg.inference, "per_device_batch_size", 512)),
        do_sample=bool(getattr(cfg.inference, "do_sample", False)),
        temperature=float(getattr(cfg.inference, "temperature", 1.0)),
        top_k=int(top_k) if top_k is not None else None,
        top_p=float(top_p) if top_p is not None else None,
    )

    result_dir = getattr(
        cfg.checkpoint, "result_dir", os.path.join(model_dir, "results")
    )
    os.makedirs(result_dir, exist_ok=True)
    info = log_run_header(logger, cfg, result_dir)
    test_ratio = float(getattr(cfg.dataset, "test_ratio", 1.0))
    for test_paths in parse_test_spec(cfg.dataset.test):
        tag = "+".join(os.path.splitext(os.path.basename(p))[0] for p in test_paths)
        logger.info("Generating for %s (tag=%s)", ", ".join(test_paths), tag)

        predictions_path = os.path.join(result_dir, tag, "predictions.jsonl")
        inference.run(
            eval_paths=test_paths,
            predictions_path=predictions_path,
            test_ratio=test_ratio,
        )
        logger.info("Wrote predictions to %s", predictions_path)
    log_run_footer(logger, info)


if __name__ == "__main__":
    main()
