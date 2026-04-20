"""Smoke tests for the script entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_EXPECTED_FIELDS = {
    "input", "output", "element", "transformation",
    "instruction", "reasoning", "answer",
}


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        **kwargs,
    )


def test_generate_data_writes_jsonl(tmp_path) -> None:
    out = tmp_path / "F1F2.jsonl"
    proc = _run(
        [
            "scripts/generate_data.py",
            "--transformations", "[F1]", "[F2]",
            "--element-length", "2",
            "--output", str(out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    first = json.loads(out.read_text().splitlines()[0])
    assert set(first) == _EXPECTED_FIELDS


def test_generate_data_rejects_non_jsonl_extension(tmp_path) -> None:
    out = tmp_path / "F1F2.txt"
    proc = _run(
        [
            "scripts/generate_data.py",
            "--transformations", "[F1]", "[F2]",
            "--element-length", "2",
            "--output", str(out),
        ]
    )
    assert proc.returncode != 0
    assert ".jsonl" in proc.stderr


def test_pretrain_help_exits_zero() -> None:
    proc = _run(["scripts/pretrain_model.py", "--help"])
    assert proc.returncode == 0, proc.stderr


def test_sft_help_exits_zero() -> None:
    proc = _run(["scripts/sft_model.py", "--help"])
    assert proc.returncode == 0, proc.stderr


def test_model_inference_help_exits_zero() -> None:
    proc = _run(["scripts/model_inference.py", "--help"])
    assert proc.returncode == 0, proc.stderr


def test_evaluate_predictions_help_exits_zero() -> None:
    # Guards the regression where the evaluation script shadowed the HF
    # `evaluate` library via Python's auto sys.path[0] insertion.
    proc = _run(["scripts/evaluate_predictions.py", "--help"])
    assert proc.returncode == 0, proc.stderr


def test_apply_noise_help_exits_zero() -> None:
    proc = _run(["scripts/apply_noise.py", "--help"])
    assert proc.returncode == 0, proc.stderr
