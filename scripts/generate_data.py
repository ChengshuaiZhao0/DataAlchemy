"""CLI: generate exhaustive CoT datasets for a given transformation composition.

Output is JSONL — one record per line with the schema documented in
:mod:`src.data.generator`.

Example
-------
    python scripts/generate_data.py \\
        --transformations "[F1]" "[F2]" \\
        --element-length 4 \\
        --output data/F1F2.jsonl
"""

from __future__ import annotations

import argparse
import sys

from src.data.generator import DatasetGenerator
from src.utils.logging import get_logger, setup_run_log

logger = get_logger("src.generate_data")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a DataAlchemy CoT JSONL dataset.")
    p.add_argument(
        "--transformations",
        "--rules",
        dest="transformations",
        nargs="+",
        required=True,
        help="Transformation tokens, e.g. '[F1]' '[F2]'. (--rules kept as alias.)",
    )
    p.add_argument("--element-length", type=int, default=4)
    p.add_argument("--rot-n", type=int, default=13)
    p.add_argument("--pos-n", type=int, default=1)
    p.add_argument(
        "--output", type=str, required=True, help="Destination .jsonl file."
    )
    p.add_argument("--no-cot", action="store_true", help="Omit <think> intermediates.")
    p.add_argument(
        "--downsample-to",
        type=int,
        default=None,
        help="Keep at most this many samples (uniform random).",
    )
    args = p.parse_args()

    if not args.output.endswith(".jsonl"):
        sys.exit(f"--output must end in .jsonl (got: {args.output!r})")

    setup_run_log(stage="Generate Data")
    gen = DatasetGenerator(
        element_length=args.element_length, rot_n=args.rot_n, pos_n=args.pos_n
    )
    n = gen.generate(
        transformation_tokens=args.transformations,
        filepath=args.output,
        use_cot=not args.no_cot,
        downsample_to=args.downsample_to,
    )
    logger.info("Wrote %d JSONL records to %s", n, args.output)


if __name__ == "__main__":
    main()
