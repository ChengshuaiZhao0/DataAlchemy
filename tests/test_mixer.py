"""Tests for the deterministic dataset mixer."""

from __future__ import annotations

import pytest

from src.data.mixer import mix_files, sweep_mix


def _write_lines(path: str, prefix: str, n: int) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(f"{prefix}{i}\n")


@pytest.fixture
def two_sources(tmp_path) -> tuple[str, str]:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    _write_lines(str(a), "A", 50)
    _write_lines(str(b), "B", 50)
    return str(a), str(b)


def test_mix_files_writes_expected_line_count(tmp_path, two_sources) -> None:
    a, b = two_sources
    out = tmp_path / "mix.txt"
    written = mix_files([a, b], [0.5, 0.5], str(out), total_lines=40)
    assert written == 40
    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 40


def test_mix_files_respects_weight_ratio(tmp_path, two_sources) -> None:
    a, b = two_sources
    out = tmp_path / "mix.txt"
    mix_files([a, b], [0.75, 0.25], str(out), total_lines=40, shuffle=False)
    with open(out) as f:
        lines = [ln.strip() for ln in f]
    # With shuffle=False the first 30 lines come from `a`, next 10 from `b`.
    assert sum(1 for ln in lines if ln.startswith("A")) == 30
    assert sum(1 for ln in lines if ln.startswith("B")) == 10


def test_mix_files_deterministic_under_fixed_seed(tmp_path, two_sources) -> None:
    a, b = two_sources
    out1 = tmp_path / "mix1.txt"
    out2 = tmp_path / "mix2.txt"
    mix_files([a, b], [0.5, 0.5], str(out1), total_lines=30, seed=7)
    mix_files([a, b], [0.5, 0.5], str(out2), total_lines=30, seed=7)
    assert out1.read_text() == out2.read_text()


def test_mix_files_rejects_length_mismatch(tmp_path, two_sources) -> None:
    a, b = two_sources
    with pytest.raises(ValueError):
        mix_files([a, b], [1.0], str(tmp_path / "mix.txt"))


def test_mix_files_rejects_zero_weights(tmp_path, two_sources) -> None:
    a, b = two_sources
    with pytest.raises(ValueError):
        mix_files([a, b], [0.0, 0.0], str(tmp_path / "mix.txt"))


def test_sweep_mix_produces_one_output_per_ratio(tmp_path, two_sources) -> None:
    a, b = two_sources
    pattern = str(tmp_path / "mix_{p}.txt")
    paths = sweep_mix(a, b, [0.2, 0.5, 0.8], pattern, total_lines=20)
    assert set(paths.keys()) == {0.2, 0.5, 0.8}
    for p, path in paths.items():
        with open(path) as f:
            lines = f.readlines()
        # Float arithmetic means int-truncation can drop at most one line.
        assert 19 <= len(lines) <= 20
        assert f"{p:.1f}".replace(".", "p") in path
