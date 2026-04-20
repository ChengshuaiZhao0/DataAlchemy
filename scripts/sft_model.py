"""CLI: supervised fine-tuning on top of a pretrained DataAlchemy model."""

from __future__ import annotations

from src.training.sft import sft
from src.utils.cli import parse_and_setup
from src.utils.config import parse_train_spec
from src.utils.logging import setup_run_log


def main() -> None:
    _, cfg, _ = parse_and_setup()
    setup_run_log(stage="SFT", cfg=cfg)
    sft(
        data_files=parse_train_spec(cfg.dataset.train),
        pretrained_model_dir=cfg.checkpoint.pretrained_model_dir,
        model_dir=cfg.checkpoint.model_dir,
        cfg=cfg,
        log_dir=getattr(cfg.checkpoint, "log_dir", None),
    )


if __name__ == "__main__":
    main()
