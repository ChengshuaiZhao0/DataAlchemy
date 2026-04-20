"""Tests for the YAML-backed Config loader."""

from __future__ import annotations

import pytest

from src.utils.config import Config, load_config


@pytest.fixture
def sample_yaml(tmp_path) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "train:\n"
        "  num_epochs: 10\n"
        "  learning_rate: 1.0e-3\n"
        "model:\n"
        "  arch: GPT\n"
        "  hidden_size: 256\n"
        "tags: [a, b, c]\n"
    )
    return str(path)


def test_dot_access_reads_nested_scalars(sample_yaml: str) -> None:
    cfg = load_config(sample_yaml)
    assert cfg.train.num_epochs == 10
    assert cfg.model.arch == "GPT"
    assert cfg.tags == ["a", "b", "c"]


def test_dict_roundtrip_preserves_structure(sample_yaml: str) -> None:
    cfg = load_config(sample_yaml)
    plain = cfg.dict()
    assert plain["train"]["learning_rate"] == pytest.approx(1e-3)
    assert plain["model"]["hidden_size"] == 256


def test_cli_override_int_and_float(sample_yaml: str) -> None:
    cfg = load_config(
        sample_yaml,
        overrides=[
            "--train.num_epochs=42",
            "--train.learning_rate=5e-4",
        ],
    )
    assert cfg.train.num_epochs == 42
    assert cfg.train.learning_rate == pytest.approx(5e-4)


def test_cli_override_bool_and_list(sample_yaml: str) -> None:
    cfg = load_config(
        sample_yaml,
        overrides=["--model.tied_weights=True", "--tags=['x', 'y']"],
    )
    assert cfg.model.tied_weights is True
    assert cfg.tags == ["x", "y"]


def test_cli_override_string_fallback(sample_yaml: str) -> None:
    cfg = load_config(sample_yaml, overrides=["--model.arch=Llama"])
    assert cfg.model.arch == "Llama"


def test_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.yaml"))


def test_unknown_attribute_raises_attribute_error(sample_yaml: str) -> None:
    cfg = load_config(sample_yaml)
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent


def test_dump_and_reload_is_lossless(sample_yaml: str, tmp_path) -> None:
    cfg = load_config(sample_yaml)
    dumped = tmp_path / "out.yaml"
    cfg.dump(str(dumped))
    reloaded = Config()
    reloaded.load(str(dumped))
    assert reloaded.dict() == cfg.dict()
