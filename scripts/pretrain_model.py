"""CLI: from-scratch pretraining from a YAML config.

Example
-------
    python scripts/pretrain_model.py --config configs/pretrain/task_generalization.yaml
"""

from __future__ import annotations

from src.training.pretrain import pretrain
from src.utils.cli import parse_and_setup
from src.utils.config import parse_train_spec
from src.utils.logging import setup_run_log


def main() -> None:
    _, cfg, _ = parse_and_setup()
    setup_run_log(stage="Pretrain", cfg=cfg)
    pretrain(
        data_files=parse_train_spec(cfg.dataset.train),
        tokenizer_dir=cfg.checkpoint.tokenizer_dir,
        model_dir=cfg.checkpoint.model_dir,
        cfg=cfg,
        log_dir=getattr(cfg.checkpoint, "log_dir", None),
    )


if __name__ == "__main__":
    main()
