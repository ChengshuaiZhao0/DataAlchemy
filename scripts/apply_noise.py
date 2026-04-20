"""CLI: inject token-level noise into one or more CoT dataset files.

Example
-------
    python scripts/apply_noise.py \\
        --inputs data/F1F2.txt \\
        --output data/F1F2_hybrid_0p05.txt \\
        --mode hybrid --p 0.05
"""

from __future__ import annotations

import argparse

from src.data.noise import NoiseMode, generate_noisy_dataset
from src.utils.logging import get_logger, setup_run_log

logger = get_logger("src.apply_noise")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Apply token-level noise to one or more CoT dataset files."
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Source CoT dataset file(s) to perturb.",
    )
    p.add_argument("--output", required=True, help="Destination file.")
    p.add_argument(
        "--mode",
        choices=[m.value for m in NoiseMode],
        required=True,
        help="Noise mode (paper Section 7).",
    )
    p.add_argument(
        "--p",
        type=float,
        required=True,
        help="Per-token perturbation probability in [0, 1].",
    )
    p.add_argument(
        "--domain",
        default="all",
        choices=["all", "element", "transformation", "instruction"],
        help="Which token roles to perturb (paper Figure 7).",
    )
    p.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Required. Pass the same seed used in your training config so noise "
        "generation cannot silently drift from training.",
    )
    args = p.parse_args()

    setup_run_log(stage="Apply Noise")
    n = generate_noisy_dataset(
        source_paths=args.inputs,
        output_path=args.output,
        mode=args.mode,
        p=args.p,
        domain=args.domain,
        seed=args.seed,
    )
    logger.info("Wrote %d noisy lines to %s", n, args.output)


if __name__ == "__main__":
    main()
