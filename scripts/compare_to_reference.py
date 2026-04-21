#!/usr/bin/env python3
"""Compare each results.json against new_reference.json in the same directory."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIRS = [
    "results/task_generalization/F2F2",
    "results/length_generalization/F1F2_len2",
    "results/length_generalization/F1F2_len3",
    "results/format_generalization/F1_hybrid_0p10+F2_hybrid_0p10",
    "results/format_generalization/F1_hybrid_0p20+F2_hybrid_0p20",
    "results/task_sft/F2F2",
]

METRIC_KEYS = ("exact_match_accuracy", "avg_edit_distance", "avg_bleu_score")
TOL = 1e-4


def fmt(x: float) -> str:
    if x is None:
        return "    n/a"
    return f"{x:9.6f}"


def main() -> None:
    any_diff = False
    for rel in RESULT_DIRS:
        d = ROOT / rel
        res_p = d / "results.json"
        ref_p = d / "new_reference.json"
        print(f"\n=== {rel} ===")
        if not res_p.exists() or not ref_p.exists():
            print(f"  missing: results={res_p.exists()} new_reference={ref_p.exists()}")
            continue
        res = json.loads(res_p.read_text())
        ref = json.loads(ref_p.read_text())

        splits = [k for k in ("full", "reasoning", "answer") if k in ref]
        print(f"  num_samples: run={res.get('num_samples')} ref={ref.get('num_samples')}")
        header = f"  {'split':<10} {'metric':<22} {'run':>10} {'ref':>10} {'|diff|':>10}"
        print(header)
        for split in splits:
            for m in METRIC_KEYS:
                rv = res.get(split, {}).get(m)
                fv = ref.get(split, {}).get(m)
                if rv is None or fv is None:
                    continue
                diff = abs(rv - fv)
                flag = " OK" if diff <= TOL else " DIFF"
                if diff > TOL:
                    any_diff = True
                print(f"  {split:<10} {m:<22} {fmt(rv)} {fmt(fv)} {diff:10.2e}{flag}")
    print("\n" + ("Some metrics differ beyond tolerance." if any_diff
                   else "All metrics match reference within tolerance."))


if __name__ == "__main__":
    main()
