"""Tests for the model / tokenizer factories."""

from __future__ import annotations

import pytest

from src.models import (
    ARCH_GPT,
    ARCH_LLAMA,
    ModelSpec,
    build_model,
    build_tokenizer,
    load_tokenizer_and_model,
    num_trainable_parameters,
)


def test_arch_constants() -> None:
    assert ARCH_GPT == "GPT"
    assert ARCH_LLAMA == "Llama"


def test_build_tokenizer_gpt_from_corpus(tmp_path, toy_rule_files) -> None:
    tok_dir = str(tmp_path / "tok")
    tok = build_tokenizer(
        tok_dir, ARCH_GPT, corpus_files=list(toy_rule_files.values()), vocab_size=512
    )
    assert tok.pad_token is not None
    assert tok.eos_token is not None
    # Reload path.
    tok2 = build_tokenizer(tok_dir, ARCH_GPT)
    assert tok2.vocab_size == tok.vocab_size


def test_build_tokenizer_without_corpus_or_cache_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        build_tokenizer(str(tmp_path / "empty"), ARCH_GPT)


def test_build_model_respects_spec(toy_model_spec, gpt_tokenizer) -> None:
    model = build_model(toy_model_spec, gpt_tokenizer)
    cfg = model.config
    assert cfg.n_embd == toy_model_spec.hidden_size
    assert cfg.n_layer == toy_model_spec.n_layer
    assert cfg.n_head == toy_model_spec.n_head
    assert cfg.n_positions == toy_model_spec.n_positions
    assert cfg.vocab_size == gpt_tokenizer.vocab_size


def test_build_model_defaults_to_sdpa(toy_model_spec, gpt_tokenizer) -> None:
    model = build_model(toy_model_spec, gpt_tokenizer)
    assert model.config._attn_implementation == "sdpa"


def test_build_model_honors_eager_attn_impl(toy_model_spec, gpt_tokenizer) -> None:
    from dataclasses import replace

    spec = replace(toy_model_spec, attn_implementation="eager")
    model = build_model(spec, gpt_tokenizer)
    assert model.config._attn_implementation == "eager"


def test_num_trainable_parameters_matches_sum(toy_gpt_model) -> None:
    expected = sum(p.numel() for p in toy_gpt_model.parameters() if p.requires_grad)
    assert num_trainable_parameters(toy_gpt_model) == expected


def test_build_model_rejects_unknown_arch(gpt_tokenizer) -> None:
    spec = ModelSpec(arch="Mamba", hidden_size=32)
    with pytest.raises(ValueError):
        build_model(spec, gpt_tokenizer)


def test_load_tokenizer_and_model_roundtrip(
    tmp_path, toy_gpt_model, gpt_tokenizer
) -> None:
    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    toy_gpt_model.save_pretrained(save_dir)
    gpt_tokenizer.save_pretrained(save_dir)

    tok, model = load_tokenizer_and_model(str(save_dir), ARCH_GPT)
    assert tok.pad_token is not None
    assert model.config.n_embd == toy_gpt_model.config.n_embd


def test_load_tokenizer_and_model_rejects_unknown_arch(tmp_path) -> None:
    with pytest.raises(ValueError):
        load_tokenizer_and_model(str(tmp_path), "Mamba")
